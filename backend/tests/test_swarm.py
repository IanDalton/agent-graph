"""Tests for the Swarm capability (orchestrated sub-agents) and conversation modes.

All unit tests use a duck-typed fake ArcadeClient and stubbed/Test models, so they need no
database, network, or LLM. Dispatch tests monkeypatch ``run_subagent`` in the swarm module's
namespace — the seam between orchestration (tested here) and delegated execution (tested via
``run_subagent`` with a TestModel).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.schemas.swarm_schemas import (
    AgentTask,
    CreateAgentArgs,
    DeepResearchArgs,
    RunSwarmArgs,
    UpdateAgentArgs,
)
from backend.skills import swarm_capability as swarm_mod
from backend.skills.subagent import SubagentOutcome, capabilities_for, run_subagent
from backend.skills.swarm_capability import (
    build_swarm,
    create_agent,
    deep_research,
    delete_agent,
    run_agent,
    run_swarm,
    update_agent,
)

EXPECTED_TOOLS = {
    "list_agents",
    "create_agent",
    "update_agent",
    "delete_agent",
    "run_agent",
    "run_swarm",
    "deep_research",
}

SPEC_ROW = {
    "agent_id": "a1",
    "name": "market-researcher",
    "role": "Researches markets",
    "instructions": "You research markets thoroughly.",
    "tools": ["web", "documents"],
}


class FakeDb:
    """Duck-typed ArcadeClient: records commands, returns canned query rows."""

    def __init__(self, *, rows: list[dict[str, Any]] | None = None, affected: int = 1) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._rows = rows or []
        self._affected = affected

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        if sql.strip().upper().startswith(("UPDATE", "DELETE")):
            return [{"count": self._affected}]
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        return self._rows


def _deps(db: FakeDb) -> GraphDependencies:
    return GraphDependencies(db=db, user_id="u", conversation_id="c")


def _ctx(db: FakeDb) -> RunContext[GraphDependencies]:
    return RunContext(deps=_deps(db), model=TestModel(), usage=RunUsage())


# --------------------------------------------------------------------------- #
# Tool registration / mode wiring
# --------------------------------------------------------------------------- #
def test_tools_are_registered() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, deps_type=GraphDependencies, capabilities=[*build_swarm()])
    asyncio.run(agent.run("hi", deps=_deps(FakeDb())))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert EXPECTED_TOOLS <= names


def test_build_agent_swarm_mode_adds_orchestrator_tools() -> None:
    from backend.main import build_agent

    agent = build_agent(model="test", mode="swarm")
    model = TestModel(call_tools=[])
    with agent.override(model=model):
        asyncio.run(agent.run("hi", deps=_deps(FakeDb())))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert EXPECTED_TOOLS <= names


def test_build_agent_regular_mode_has_no_swarm_tools() -> None:
    from backend.main import build_agent

    agent = build_agent(model="test", mode="regular")
    model = TestModel(call_tools=[])
    with agent.override(model=model):
        asyncio.run(agent.run("hi", deps=_deps(FakeDb())))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert not (EXPECTED_TOOLS & names)


def test_build_agent_unknown_mode_falls_back_to_regular() -> None:
    from backend.main import build_agent

    agent = build_agent(model="test", mode="council")
    model = TestModel(call_tools=[])
    with agent.override(model=model):
        asyncio.run(agent.run("hi", deps=_deps(FakeDb())))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert "run_swarm" not in names


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
def test_create_agent_args_normalize_name_and_tools() -> None:
    args = CreateAgentArgs(
        name="  Pitch-Deck-Designer ", role="r", instructions="i", tools=["WEB", "web", "documents"]
    )
    assert args.name == "pitch-deck-designer"
    assert args.tools == ["web", "documents"]  # lowered + de-duplicated, order kept


def test_create_agent_args_reject_bad_name_and_unknown_tools() -> None:
    with pytest.raises(ValidationError):
        CreateAgentArgs(name="Not Valid!", role="r", instructions="i")
    with pytest.raises(ValidationError):
        CreateAgentArgs(name="ok-name", role="r", instructions="i", tools=["filesystem"])


def test_run_swarm_args_cap_batch_size() -> None:
    tasks = [AgentTask(agent="a", task="t")] * 9
    with pytest.raises(ValidationError):
        RunSwarmArgs(tasks=tasks)


# --------------------------------------------------------------------------- #
# Roster CRUD
# --------------------------------------------------------------------------- #
def test_create_agent_persists_and_links() -> None:
    db = FakeDb(rows=[])  # no existing agent with that name
    info = asyncio.run(
        create_agent(
            _ctx(db),
            CreateAgentArgs(name="pitch-deck-designer", role="Designs decks", instructions="x"),
        )
    )
    assert info.agent_id
    assert info.tools == ["web", "documents"]  # the default grant
    create = next(p for s, p in db.commands if s.startswith("CREATE VERTEX AgentSpec"))
    assert create["name"] == "pitch-deck-designer" and create["uid"] == "u"
    assert any(s.startswith("CREATE EDGE HAS_AGENT") for s, _ in db.commands)


def test_create_agent_duplicate_name_is_model_retry() -> None:
    db = FakeDb(rows=[SPEC_ROW])
    with pytest.raises(ModelRetry):
        asyncio.run(
            create_agent(
                _ctx(db),
                CreateAgentArgs(name="market-researcher", role="r", instructions="i"),
            )
        )


def test_update_agent_requires_some_change_and_known_agent() -> None:
    with pytest.raises(ModelRetry):
        asyncio.run(update_agent(_ctx(FakeDb(rows=[SPEC_ROW])), UpdateAgentArgs(agent="a1")))
    with pytest.raises(ModelRetry):
        asyncio.run(
            update_agent(_ctx(FakeDb(rows=[])), UpdateAgentArgs(agent="nope", role="new"))
        )


def test_update_agent_revises_in_place() -> None:
    db = FakeDb(rows=[SPEC_ROW])
    msg = asyncio.run(
        update_agent(_ctx(db), UpdateAgentArgs(agent="market-researcher", instructions="sharper"))
    )
    assert "Updated" in msg
    sql, params = next((s, p) for s, p in db.commands if s.startswith("UPDATE AgentSpec"))
    assert params["instructions"] == "sharper" and params["aid"] == "a1"
    assert "role" not in params  # untouched fields are not overwritten


def test_delete_agent_unknown_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        asyncio.run(delete_agent(_ctx(FakeDb(rows=[])), "nope"))


# --------------------------------------------------------------------------- #
# Dispatch (run_agent / run_swarm / deep_research)
# --------------------------------------------------------------------------- #
def test_run_agent_unknown_agent_is_error_report_not_exception() -> None:
    report = asyncio.run(run_agent(_ctx(FakeDb(rows=[])), AgentTask(agent="ghost", task="t")))
    assert report.error and "list_agents" in report.error
    assert report.output == ""


def test_run_agent_maps_subagent_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def stub(deps: Any, *, instructions: str, tool_groups: list[str], prompt: str, **kw: Any) -> SubagentOutcome:
        seen.update(instructions=instructions, tool_groups=tool_groups, prompt=prompt)
        return SubagentOutcome(output="deck outline done")

    monkeypatch.setattr(swarm_mod, "run_subagent", stub)
    report = asyncio.run(
        run_agent(
            _ctx(FakeDb(rows=[SPEC_ROW])),
            AgentTask(agent="market-researcher", task="size the market", context="ACME, B2B"),
        )
    )
    assert report.error is None and report.output == "deck outline done"
    assert report.agent_id == "a1" and report.name == "market-researcher"
    assert seen["tool_groups"] == ["web", "documents"]
    assert "size the market" in seen["prompt"] and "ACME, B2B" in seen["prompt"]
    assert "market-researcher" in seen["instructions"]


def test_run_swarm_runs_tasks_concurrently_and_keeps_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0

    async def stub(deps: Any, *, prompt: str, **kw: Any) -> SubagentOutcome:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return SubagentOutcome(output=f"done: {prompt}")

    monkeypatch.setattr(swarm_mod, "run_subagent", stub)
    tasks = [AgentTask(agent="market-researcher", task=f"t{i}") for i in range(3)]
    result = asyncio.run(run_swarm(_ctx(FakeDb(rows=[SPEC_ROW])), RunSwarmArgs(tasks=tasks)))
    assert [r.output for r in result.reports] == ["done: t0", "done: t1", "done: t2"]
    assert max_active > 1  # tasks genuinely overlapped


def test_run_swarm_isolates_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stub(deps: Any, *, prompt: str, **kw: Any) -> SubagentOutcome:
        if "bad" in prompt:
            return SubagentOutcome(error="model exploded")
        return SubagentOutcome(output="fine")

    monkeypatch.setattr(swarm_mod, "run_subagent", stub)
    tasks = [
        AgentTask(agent="market-researcher", task="good one"),
        AgentTask(agent="market-researcher", task="bad one"),
    ]
    result = asyncio.run(run_swarm(_ctx(FakeDb(rows=[SPEC_ROW])), RunSwarmArgs(tasks=tasks)))
    assert result.reports[0].output == "fine" and result.reports[0].error is None
    assert result.reports[1].error == "model exploded"


def test_deep_research_uses_builtin_researcher(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def stub(deps: Any, *, instructions: str, tool_groups: list[str], prompt: str, **kw: Any) -> SubagentOutcome:
        seen.update(tool_groups=tool_groups, prompt=prompt, instructions=instructions)
        return SubagentOutcome(output="findings digest")

    monkeypatch.setattr(swarm_mod, "run_subagent", stub)
    result = asyncio.run(
        deep_research(
            _ctx(FakeDb()), DeepResearchArgs(question="GLP-1 market size", focus="EU, 2026")
        )
    )
    assert result.report == "findings digest" and result.error is None
    assert seen["tool_groups"] == ["web", "documents"]
    assert "GLP-1 market size" in seen["prompt"] and "EU, 2026" in seen["prompt"]


def test_deep_research_persists_report_when_delegate_made_no_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stub(deps: Any, *, prompt: str, **kw: Any) -> SubagentOutcome:
        return SubagentOutcome(output="A thorough, cited report.")  # no documents

    monkeypatch.setattr(swarm_mod, "run_subagent", stub)
    db = FakeDb()
    result = asyncio.run(
        deep_research(_ctx(db), DeepResearchArgs(question="GLP-1 market size"))
    )
    assert len(result.documents) == 1
    assert result.documents[0].title.startswith("Research: GLP-1")
    saved = next(p for s, p in db.commands if s.startswith("CREATE VERTEX Document"))
    assert saved["content"] == "A thorough, cited report."


def test_deep_research_keeps_delegate_documents_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.schemas.document_schemas import DocumentInfo

    async def stub(deps: Any, *, prompt: str, **kw: Any) -> SubagentOutcome:
        return SubagentOutcome(
            output="digest", documents=[DocumentInfo(document_id="d1", title="Report")]
        )

    monkeypatch.setattr(swarm_mod, "run_subagent", stub)
    db = FakeDb()
    result = asyncio.run(deep_research(_ctx(db), DeepResearchArgs(question="q")))
    assert [d.document_id for d in result.documents] == ["d1"]
    assert not any(s.startswith("CREATE VERTEX Document") for s, _ in db.commands)


# --------------------------------------------------------------------------- #
# Research mode (the instructions overlay + report safeguard)
# --------------------------------------------------------------------------- #
def test_research_mode_safeguard_saves_report_when_model_skips_create_document() -> None:
    from backend.skills.document_capability import build_documents
    from backend.skills.research_capability import build_research

    report = "Executive summary.\n" + "Detailed findings about the topic. " * 20
    db = FakeDb()
    agent = Agent(
        TestModel(call_tools=[], custom_output_text=report),
        deps_type=GraphDependencies,
        capabilities=[*build_documents(), *build_research()],
    )
    asyncio.run(agent.run("research X", deps=_deps(db)))
    saved = [p for s, p in db.commands if s.startswith("CREATE VERTEX Document")]
    assert len(saved) == 1 and saved[0]["content"] == report.strip()


def test_research_mode_safeguard_skips_short_replies() -> None:
    from backend.skills.document_capability import build_documents
    from backend.skills.research_capability import build_research

    db = FakeDb()
    agent = Agent(
        TestModel(call_tools=[], custom_output_text="Which region?"),
        deps_type=GraphDependencies,
        capabilities=[*build_documents(), *build_research()],
    )
    asyncio.run(agent.run("research X", deps=_deps(db)))
    assert not any(s.startswith("CREATE VERTEX Document") for s, _ in db.commands)


def test_research_mode_safeguard_skips_when_model_creates_document() -> None:
    from backend.skills.document_capability import build_documents
    from backend.skills.research_capability import build_research

    db = FakeDb()
    agent = Agent(
        TestModel(call_tools=["create_document"]),
        deps_type=GraphDependencies,
        capabilities=[*build_documents(), *build_research()],
    )
    asyncio.run(agent.run("research X", deps=_deps(db)))
    # Only the model's own create_document call persisted; the safeguard added nothing.
    saved = [p for s, p in db.commands if s.startswith("CREATE VERTEX Document")]
    assert len(saved) == 1


# --------------------------------------------------------------------------- #
# run_subagent (the delegated runner itself)
# --------------------------------------------------------------------------- #
def test_run_subagent_returns_output_with_test_model() -> None:
    outcome = asyncio.run(
        run_subagent(
            _deps(FakeDb()),
            instructions="You are a test agent.",
            tool_groups=["documents"],
            prompt="go",
            model=TestModel(call_tools=[], custom_output_text="done"),
        )
    )
    assert outcome.output == "done" and outcome.error is None


def test_run_subagent_collects_created_documents() -> None:
    # TestModel calls create_document (synthesized args) before answering; the collector hook
    # must surface the persisted document on the outcome.
    outcome = asyncio.run(
        run_subagent(
            _deps(FakeDb()),
            instructions="You are a test agent.",
            tool_groups=["documents"],
            prompt="go",
            model=TestModel(call_tools=["create_document"]),
        )
    )
    assert outcome.error is None
    assert len(outcome.documents) == 1 and outcome.documents[0].document_id


def test_run_subagent_swallows_model_failures() -> None:
    from pydantic_ai.models.function import FunctionModel

    def boom(messages: Any, info: Any) -> Any:
        raise RuntimeError("provider down")

    outcome = asyncio.run(
        run_subagent(
            _deps(FakeDb()),
            instructions="x",
            tool_groups=[],
            prompt="go",
            model=FunctionModel(boom),
        )
    )
    assert outcome.error and "provider down" in outcome.error


def test_capabilities_for_skips_unknown_groups() -> None:
    caps = capabilities_for(["web", "made-up-group"])
    assert len(caps) == 1 and caps[0].id == "WebSearch"


# --------------------------------------------------------------------------- #
# Conversation modes (repository)
# --------------------------------------------------------------------------- #
def test_create_conversation_stores_mode() -> None:
    db = FakeDb(rows=[])
    asyncio.run(repo.create_conversation(db, "u", "c", mode="swarm"))
    create = next(p for s, p in db.commands if s.startswith("CREATE VERTEX Conversation"))
    assert create["mode"] == "swarm"


def test_set_conversation_mode_updates_in_place() -> None:
    db = FakeDb()
    asyncio.run(repo.set_conversation_mode(db, "c", "research"))
    sql, params = next((s, p) for s, p in db.commands if s.startswith("UPDATE Conversation"))
    assert params["mode"] == "research" and params["cid"] == "c"


def test_get_conversation_mode_defaults_to_regular() -> None:
    assert asyncio.run(repo.get_conversation_mode(FakeDb(rows=[]), "c")) == "regular"
    assert asyncio.run(repo.get_conversation_mode(FakeDb(rows=[{"mode": None}]), "c")) == "regular"
    assert asyncio.run(repo.get_conversation_mode(FakeDb(rows=[{"mode": "swarm"}]), "c")) == "swarm"


def test_list_conversations_reports_stored_mode_with_fallback() -> None:
    db = FakeDb(
        rows=[
            {"conversation_id": "c1", "mode": "research"},
            {"conversation_id": "c2", "mode": None},  # pre-modes conversation
        ]
    )
    rows = asyncio.run(repo.list_conversations(db, "u"))
    assert [r["mode"] for r in rows] == ["research", "regular"]


# --------------------------------------------------------------------------- #
# stream_run's document frames for delegated work
# --------------------------------------------------------------------------- #
def test_document_events_from_run_agent_report() -> None:
    from backend.main import _document_events
    from backend.schemas.document_schemas import DocumentInfo
    from backend.schemas.swarm_schemas import AgentRunReport

    report = AgentRunReport(
        agent_id="a1",
        documents=[DocumentInfo(document_id="d1", title="Deck", mime_type="text/markdown")],
    )
    events = _document_events("run_agent", report, args=None)
    assert [(e["document_id"], e["action"]) for e in events] == [("d1", "created")]


def test_document_events_from_run_swarm_reports() -> None:
    from backend.main import _document_events
    from backend.schemas.document_schemas import DocumentInfo
    from backend.schemas.swarm_schemas import AgentRunReport, SwarmRunResult

    result = SwarmRunResult(
        reports=[
            AgentRunReport(
                agent_id="a1",
                documents=[DocumentInfo(document_id="d1", title="A", mime_type="text/markdown")],
            ),
            AgentRunReport(agent_id="a2", error="failed"),
            AgentRunReport(
                agent_id="a3",
                documents=[DocumentInfo(document_id="d2", title="B", mime_type="text/csv")],
            ),
        ]
    )
    events = _document_events("run_swarm", result, args=None)
    assert [e["document_id"] for e in events] == ["d1", "d2"]

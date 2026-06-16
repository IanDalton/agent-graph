"""Tests for the Swarm capability (orchestrated sub-agents) and conversation modes.

All unit tests use a duck-typed fake ArcadeClient and stubbed/Test models, so they need no
database, network, or LLM. Communication tests monkeypatch ``run_subagent`` in the *subagent*
module's namespace (where ``dispatch_message`` calls it) — the seam between agency routing (tested
here) and delegated execution (tested via ``run_subagent`` with a TestModel). ``deep_research``
still calls ``run_subagent`` from the swarm module's namespace, so its tests patch there.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.schemas.swarm_schemas import (
    CreateAgentArgs,
    DeepResearchArgs,
    SendMessageArgs,
    SendMessagesArgs,
    UpdateAgentArgs,
)
from backend.skills import subagent as subagent_mod
from backend.skills import swarm_capability as swarm_mod
from backend.skills.subagent import (
    SWARM_MAX_DEPTH,
    SubagentOutcome,
    _can_delegate,
    capabilities_for,
    dispatch_message,
    dispatch_messages,
    run_subagent,
)
from backend.skills.swarm_capability import (
    DEFAULT_SWARM_AGENTS,
    build_swarm,
    create_agent,
    deep_research,
    delete_agent,
    send_message,
    send_messages,
    update_agent,
)

EXPECTED_TOOLS = {
    "list_agents",
    "create_agent",
    "update_agent",
    "delete_agent",
    "send_message",
    "send_messages",
    "deep_research",
}

SPEC_ROW = {
    "agent_id": "a1",
    "name": "market-researcher",
    "role": "Researches markets",
    "instructions": "You research markets thoroughly.",
    "tools": ["web", "documents"],
    "recipients": [],
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
    assert "send_messages" not in names


# --------------------------------------------------------------------------- #
# Pure-orchestrator wiring + default-roster seeding
# --------------------------------------------------------------------------- #
def test_swarm_orchestrator_has_no_work_tools() -> None:
    """The swarm orchestrator can only route/synthesize — no web, sandbox, ontology, documents."""
    from backend.main import build_agent

    agent = build_agent(model="test", mode="swarm")
    model = TestModel(call_tools=[])
    with agent.override(model=model):
        asyncio.run(agent.run("hi", deps=_deps(FakeDb())))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    # Has the swarm/communication tools...
    assert EXPECTED_TOOLS <= names
    # ...but NOT the "doing" tools (it must delegate those to specialists).
    work_tools = {
        "web_search", "fetch_url", "run_python",
        "create_vertex_type", "create_node",
        "create_document", "list_documents",
    }
    assert not (work_tools & names), f"orchestrator unexpectedly has: {work_tools & names}"


def test_swarm_seeds_default_roster_when_empty() -> None:
    from backend.main import build_agent

    db = FakeDb(rows=[])  # empty roster
    agent = build_agent(model="test", mode="swarm")
    with agent.override(model=TestModel(call_tools=[])):
        asyncio.run(agent.run("hi", deps=_deps(db)))
    creates = [p for s, p in db.commands if s.startswith("CREATE VERTEX AgentSpec")]
    created = {p["name"] for p in creates}
    assert created == {a["name"] for a in DEFAULT_SWARM_AGENTS}
    # The seeded team-lead is a sub-orchestrator: its recipients are the worker specialists.
    assert "team-lead" in created
    team_lead = next(p for p in creates if p["name"] == "team-lead")
    workers = created - {"team-lead"}
    assert set(team_lead["recipients"]) == workers
    # Workers are leaves (no chart edges).
    web_researcher = next(p for p in creates if p["name"] == "web-researcher")
    assert web_researcher["recipients"] == []


def test_swarm_does_not_seed_when_roster_present() -> None:
    from backend.main import build_agent

    db = FakeDb(rows=[SPEC_ROW])  # user already has an agent
    agent = build_agent(model="test", mode="swarm")
    with agent.override(model=TestModel(call_tools=[])):
        asyncio.run(agent.run("hi", deps=_deps(db)))
    assert not any(s.startswith("CREATE VERTEX AgentSpec") for s, _ in db.commands)


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


def test_create_agent_args_normalize_and_reject_recipients() -> None:
    # The communication chart edges are kebab-cased, lowered and de-duplicated.
    args = CreateAgentArgs(
        name="lead", role="r", instructions="i", recipients=["Market-Researcher", "market-researcher"]
    )
    assert args.recipients == ["market-researcher"]
    # Bad recipient names are rejected (same shape rule as agent names).
    with pytest.raises(ValidationError):
        CreateAgentArgs(name="lead", role="r", instructions="i", recipients=["Not Valid!"])
    # recipients default to empty (a leaf specialist).
    assert CreateAgentArgs(name="lead", role="r", instructions="i").recipients == []


def test_send_messages_args_cap_batch_size() -> None:
    messages = [SendMessageArgs(recipient="a", message="t")] * 9
    with pytest.raises(ValidationError):
        SendMessagesArgs(messages=messages)


# --------------------------------------------------------------------------- #
# Roster CRUD
# --------------------------------------------------------------------------- #
def test_create_agent_persists_and_links() -> None:
    db = FakeDb(rows=[])  # no existing agent with that name
    info = asyncio.run(
        create_agent(
            _ctx(db),
            CreateAgentArgs(
                name="pitch-deck-designer",
                role="Designs decks",
                instructions="x",
                recipients=["market-researcher"],
            ),
        )
    )
    assert info.agent_id
    assert info.tools == ["web", "documents"]  # the default grant
    assert info.recipients == ["market-researcher"]  # the chart edge round-trips
    create = next(p for s, p in db.commands if s.startswith("CREATE VERTEX AgentSpec"))
    assert create["name"] == "pitch-deck-designer" and create["uid"] == "u"
    assert create["recipients"] == ["market-researcher"]
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
# Communication (send_message / send_messages / deep_research)
# --------------------------------------------------------------------------- #
def test_send_message_unknown_recipient_is_error_report_not_exception() -> None:
    report = asyncio.run(
        send_message(_ctx(FakeDb(rows=[])), SendMessageArgs(recipient="ghost", message="t"))
    )
    assert report.error and "list_agents" in report.error
    assert report.output == ""


def test_send_message_maps_subagent_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def stub(deps: Any, *, instructions: str, tool_groups: list[str], prompt: str, **kw: Any) -> SubagentOutcome:
        seen.update(instructions=instructions, tool_groups=tool_groups, prompt=prompt, kw=kw)
        return SubagentOutcome(output="deck outline done")

    # dispatch_message lives in the subagent module and calls run_subagent there.
    monkeypatch.setattr(subagent_mod, "run_subagent", stub)
    report = asyncio.run(
        send_message(
            _ctx(FakeDb(rows=[SPEC_ROW])),
            SendMessageArgs(
                recipient="market-researcher", message="size the market", context="ACME, B2B"
            ),
        )
    )
    assert report.error is None and report.output == "deck outline done"
    assert report.agent_id == "a1" and report.name == "market-researcher"
    assert seen["tool_groups"] == ["web", "documents"]
    assert "size the market" in seen["prompt"] and "ACME, B2B" in seen["prompt"]
    assert "market-researcher" in seen["instructions"]
    assert seen["kw"]["recipients"] == []  # the spec's chart edges flow to run_subagent


def test_dispatch_messages_runs_concurrently_and_is_tolerant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shared fan-out used by BOTH the orchestrator and sub-orchestrators: concurrent, ordered,
    # and one failure never aborts the batch.
    active = 0
    max_active = 0

    async def stub(deps: Any, *, prompt: str, **kw: Any) -> SubagentOutcome:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        if "bad" in prompt:
            return SubagentOutcome(error="boom")
        return SubagentOutcome(output=f"done: {prompt}")

    monkeypatch.setattr(subagent_mod, "run_subagent", stub)
    messages = [
        SendMessageArgs(recipient="market-researcher", message="t0"),
        SendMessageArgs(recipient="market-researcher", message="bad"),
        SendMessageArgs(recipient="market-researcher", message="t2"),
    ]
    reports = asyncio.run(dispatch_messages(_deps(FakeDb(rows=[SPEC_ROW])), messages))
    assert [r.task for r in reports] == ["t0", "bad", "t2"]  # order preserved
    assert reports[0].output == "done: t0" and reports[2].output == "done: t2"
    assert reports[1].error == "boom"  # isolated failure
    assert max_active > 1  # genuinely overlapped


def test_send_message_enforces_communication_chart() -> None:
    # A dispatched specialist (agency_recipients set) may only message its charted teammates.
    deps = replace(_deps(FakeDb(rows=[SPEC_ROW])), agency_recipients=["someone-else"])
    report = asyncio.run(dispatch_message(deps, "market-researcher", "do a thing"))
    assert report.error and "communication chart" in report.error
    assert report.output == ""  # never dispatched


def test_send_messages_run_concurrently_and_keep_order(
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

    monkeypatch.setattr(subagent_mod, "run_subagent", stub)
    messages = [SendMessageArgs(recipient="market-researcher", message=f"t{i}") for i in range(3)]
    result = asyncio.run(
        send_messages(_ctx(FakeDb(rows=[SPEC_ROW])), SendMessagesArgs(messages=messages))
    )
    assert [r.output for r in result.reports] == ["done: t0", "done: t1", "done: t2"]
    assert max_active > 1  # messages genuinely overlapped


def test_send_messages_isolates_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stub(deps: Any, *, prompt: str, **kw: Any) -> SubagentOutcome:
        if "bad" in prompt:
            return SubagentOutcome(error="model exploded")
        return SubagentOutcome(output="fine")

    monkeypatch.setattr(subagent_mod, "run_subagent", stub)
    messages = [
        SendMessageArgs(recipient="market-researcher", message="good one"),
        SendMessageArgs(recipient="market-researcher", message="bad one"),
    ]
    result = asyncio.run(
        send_messages(_ctx(FakeDb(rows=[SPEC_ROW])), SendMessagesArgs(messages=messages))
    )
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
# Multi-hop agency chart: when a delegate gets its own send_message tool
# --------------------------------------------------------------------------- #
def _subagent_tool_names(deps: GraphDependencies, recipients: list[str]) -> set[str]:
    """Tool names a delegate is built with, for the given chart edges/depth."""
    model = TestModel(call_tools=[])
    asyncio.run(
        run_subagent(
            deps,
            instructions="x",
            tool_groups=[],
            prompt="go",
            recipients=recipients,
            model=model,
        )
    )
    return {t.name for t in model.last_model_request_parameters.function_tools}


def test_run_subagent_grants_both_comms_within_depth() -> None:
    # A delegate dispatched from the entry point (depth 0) with chart edges becomes a
    # sub-orchestrator: it gets BOTH send_message and the parallel send_messages.
    names = _subagent_tool_names(_deps(FakeDb()), recipients=["teammate"])
    assert {"send_message", "send_messages"} <= names


def test_run_subagent_withholds_comms_at_depth_ceiling() -> None:
    # Past the depth ceiling the delegate is a leaf even though it has chart edges.
    deep = replace(_deps(FakeDb()), agency_depth=SWARM_MAX_DEPTH)
    names = _subagent_tool_names(deep, recipients=["teammate"])
    assert not ({"send_message", "send_messages"} & names)


def test_run_subagent_leaf_has_no_comms() -> None:
    # No chart edges → no delegation tools, regardless of depth.
    names = _subagent_tool_names(_deps(FakeDb()), recipients=[])
    assert not ({"send_message", "send_messages"} & names)


# --------------------------------------------------------------------------- #
# Per-conversation swarm bounds (max parallel / max depth)
# --------------------------------------------------------------------------- #
def test_can_delegate_honors_explicit_max_depth() -> None:
    # max_depth=1 ⇒ even a depth-0 dispatch is a leaf; max_depth=2 ⇒ it may delegate.
    assert _can_delegate(0, ["x"], max_depth=1) is False
    assert _can_delegate(0, ["x"], max_depth=2) is True


def test_run_subagent_respects_conversation_max_depth() -> None:
    # A conversation override of swarm_max_depth=1 makes a depth-0 delegate a leaf (no comms),
    # even though it has chart edges and the env default would have granted them.
    deps = replace(_deps(FakeDb()), swarm_max_depth=1)
    assert not ({"send_message", "send_messages"} & _subagent_tool_names(deps, recipients=["t"]))


def test_dispatch_messages_respects_conversation_max_parallel(
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
        return SubagentOutcome(output="ok")

    monkeypatch.setattr(subagent_mod, "run_subagent", stub)
    deps = replace(_deps(FakeDb(rows=[SPEC_ROW])), swarm_max_parallel=1)
    messages = [SendMessageArgs(recipient="market-researcher", message=f"t{i}") for i in range(3)]
    asyncio.run(dispatch_messages(deps, messages))
    assert max_active == 1  # the conversation's cap of 1 serialized the batch


def test_set_conversation_swarm_settings_updates_only_provided() -> None:
    db = FakeDb()
    asyncio.run(repo.set_conversation_swarm_settings(db, "c", max_depth=2))
    sql, params = next((s, p) for s, p in db.commands if s.startswith("UPDATE Conversation"))
    assert params["md"] == 2 and params["cid"] == "c"
    assert "mp" not in params  # max_parallel was not sent → not in the SET clause


def test_get_conversation_swarm_settings_reads_values_and_nulls() -> None:
    got = asyncio.run(
        repo.get_conversation_swarm_settings(
            FakeDb(rows=[{"swarm_max_parallel": 3, "swarm_max_depth": None}]), "c"
        )
    )
    assert got == {"max_parallel": 3, "max_depth": None}
    assert asyncio.run(repo.get_conversation_swarm_settings(FakeDb(rows=[]), "c")) == {
        "max_parallel": None,
        "max_depth": None,
    }


def test_update_conversation_validates_swarm_ranges() -> None:
    from backend.api import UpdateConversation

    assert UpdateConversation(swarm_max_parallel=4, swarm_max_depth=3).swarm_max_depth == 3
    with pytest.raises(ValidationError):
        UpdateConversation(swarm_max_parallel=99)  # above the allowed range
    with pytest.raises(ValidationError):
        UpdateConversation(swarm_max_depth=0)  # below the allowed range


# --------------------------------------------------------------------------- #
# run_subagent live-trace streaming (event_sink path)
# --------------------------------------------------------------------------- #
def _drain(queue: "asyncio.Queue[dict[str, Any]]") -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while not queue.empty():
        frames.append(queue.get_nowait())
    return frames


def test_run_subagent_streams_tagged_frames_to_sink() -> None:
    from dataclasses import replace

    sink: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue()
    deps = replace(_deps(FakeDb()), event_sink=sink)
    outcome = asyncio.run(
        run_subagent(
            deps,
            instructions="You are a test agent.",
            tool_groups=["documents"],
            prompt="go",
            model=TestModel(call_tools=["create_document"], custom_output_text="done"),
            agent_id="a1",
            agent_name="market-researcher",
            instance_id="abc123",
        )
    )
    assert outcome.error is None and outcome.output == "done"

    frames = _drain(sink)
    types = [f["type"] for f in frames]
    assert types[0] == "agent_start" and types[-1] == "agent_end"
    assert {"tool_call", "tool_result", "text"} <= set(types)
    # Every frame is tagged with this delegate's identity.
    assert all(
        f["agent_id"] == "a1"
        and f["name"] == "market-researcher"
        and f["instance_id"] == "abc123"
        for f in frames
    )
    # Tool ids are instance-namespaced so the UI never collides them with the parent/siblings.
    call = next(f for f in frames if f["type"] == "tool_call")
    result = next(f for f in frames if f["type"] == "tool_result")
    assert call["tool_call_id"].startswith("abc123:")
    assert result["tool_call_id"].startswith("abc123:")
    # Documents ride back on the report (orchestrator _document_events), not the live trace.
    assert "document" not in types


def test_run_subagent_sink_path_swallows_failure_but_closes_bubble() -> None:
    from dataclasses import replace

    from pydantic_ai.models.function import FunctionModel

    async def boom_stream(messages: Any, info: Any) -> Any:
        raise RuntimeError("provider down")
        yield ""  # pragma: no cover — marks this an async generator (a streaming model)

    sink: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue()
    deps = replace(_deps(FakeDb()), event_sink=sink)
    outcome = asyncio.run(
        run_subagent(
            deps,
            instructions="x",
            tool_groups=[],
            prompt="go",
            model=FunctionModel(stream_function=boom_stream),
            agent_id="a1",
            agent_name="r",
            instance_id="i1",
        )
    )
    assert outcome.error and "provider down" in outcome.error
    types = [f["type"] for f in _drain(sink)]
    # The bubble always opens and closes, even when the delegate blows up mid-run.
    assert types[0] == "agent_start" and types[-1] == "agent_end"


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
            {"conversation_id": "c1", "mode": "research", "system_prompt": "be terse"},
            {"conversation_id": "c2", "mode": None},  # pre-modes conversation
        ]
    )
    rows = asyncio.run(repo.list_conversations(db, "u"))
    assert [r["mode"] for r in rows] == ["research", "regular"]
    assert [r["system_prompt"] for r in rows] == ["be terse", ""]


# --------------------------------------------------------------------------- #
# Per-conversation system prompt
# --------------------------------------------------------------------------- #
def test_set_conversation_system_prompt_updates_in_place() -> None:
    db = FakeDb()
    asyncio.run(repo.set_conversation_system_prompt(db, "c", "Answer in French."))
    sql, params = next((s, p) for s, p in db.commands if s.startswith("UPDATE Conversation"))
    assert params["sp"] == "Answer in French." and params["cid"] == "c"


def test_get_conversation_system_prompt_defaults_to_empty() -> None:
    assert asyncio.run(repo.get_conversation_system_prompt(FakeDb(rows=[]), "c")) == ""
    assert (
        asyncio.run(repo.get_conversation_system_prompt(FakeDb(rows=[{"system_prompt": None}]), "c"))
        == ""
    )
    assert (
        asyncio.run(repo.get_conversation_system_prompt(FakeDb(rows=[{"system_prompt": "hi"}]), "c"))
        == "hi"
    )


def test_compose_instructions_appends_custom_prompt() -> None:
    from backend.main import compose_instructions
    from backend.skills.system_prompt import BASE_SYSTEM_PROMPT

    # No custom prompt → base unchanged (also for whitespace-only input).
    assert compose_instructions("") == BASE_SYSTEM_PROMPT
    assert compose_instructions(None) == BASE_SYSTEM_PROMPT
    assert compose_instructions("   ") == BASE_SYSTEM_PROMPT

    composed = compose_instructions("Always answer in French.")
    assert composed.startswith(BASE_SYSTEM_PROMPT)
    assert "Always answer in French." in composed
    assert "ADDITIONAL INSTRUCTIONS" in composed


# --------------------------------------------------------------------------- #
# stream_run's document frames for delegated work
# --------------------------------------------------------------------------- #
def test_document_events_from_send_message_report() -> None:
    from backend.main import _document_events
    from backend.schemas.document_schemas import DocumentInfo
    from backend.schemas.swarm_schemas import AgentRunReport

    report = AgentRunReport(
        agent_id="a1",
        documents=[DocumentInfo(document_id="d1", title="Deck", mime_type="text/markdown")],
    )
    events = _document_events("send_message", report, args=None)
    assert [(e["document_id"], e["action"]) for e in events] == [("d1", "created")]


def test_document_events_from_send_messages_reports() -> None:
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
    events = _document_events("send_messages", result, args=None)
    assert [e["document_id"] for e in events] == ["d1", "d2"]

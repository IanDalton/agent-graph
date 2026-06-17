"""Tests for Projects + conversation lifecycle (delete/archive/pin) + global documents.

All unit tests use duck-typed fake ArcadeClients, so they need no database. Repo functions are
plain coroutines; the capability tools are called with a hand-built RunContext like the other
capability tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.main import build_agent, compose_instructions, BASE_SYSTEM_PROMPT
from backend.skills.document_capability import (
    build_documents,
    list_project_documents,
    read_project_document,
    search_project_documents,
)
from backend.skills.system_prompt import project_documents_block


class FakeDb:
    """Duck-typed ArcadeClient: records commands, returns canned query rows (same for every query)."""

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


class RoutedDb:
    """Fake that returns different rows per query, keyed by a substring of the SQL (first match)."""

    def __init__(self, routes: dict[str, list[dict[str, Any]]], affected: int = 1) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._routes = routes
        self._affected = affected

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        if sql.strip().upper().startswith(("UPDATE", "DELETE")):
            return [{"count": self._affected}]
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        for needle, rows in self._routes.items():
            if needle in sql:
                return rows
        return []


def _ctx(db: Any, *, project_id: str | None = None) -> RunContext[GraphDependencies]:
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c", project_id=project_id)
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


# --------------------------------------------------------------------------- #
# Repository: projects
# --------------------------------------------------------------------------- #
def test_create_project_persists_and_links() -> None:
    db = FakeDb(rows=[{"n": 0}])  # existence check → not present
    asyncio.run(repo.create_project(db, "u", "p1", title="Acme", system_prompt="Be terse."))
    create = next(p for s, p in db.commands if s.startswith("CREATE VERTEX Project"))
    assert create["pid"] == "p1" and create["uid"] == "u"
    assert create["title"] == "Acme" and create["sp"] == "Be terse."
    assert any(s.startswith("CREATE EDGE HAS_PROJECT") for s, _ in db.commands)


def test_create_project_idempotent() -> None:
    db = FakeDb(rows=[{"n": 1}])  # already exists
    asyncio.run(repo.create_project(db, "u", "p1"))
    assert not any(s.startswith("CREATE VERTEX Project") for s, _ in db.commands)


def test_delete_project_cascades_and_spares_globals() -> None:
    db = FakeDb(rows=[{"conversation_id": "c1"}], affected=2)
    result = asyncio.run(repo.delete_project(db, "u", "p1"))
    # Member conversation cascaded (its children + the conversation vertex deleted).
    assert any("DELETE VERTEX FROM (SELECT FROM Message" in s for s, _ in db.commands)
    assert any("DELETE VERTEX FROM (SELECT FROM Conversation" in s for s, _ in db.commands)
    # Non-global project documents deleted; global docs un-scoped (project_id cleared), not deleted.
    assert any(
        "DELETE VERTEX FROM (SELECT FROM Document" in s and "is_global" in s
        for s, _ in db.commands
    )
    assert any(
        s.startswith("UPDATE Document SET project_id = null") and "is_global = true" in s
        for s, _ in db.commands
    )
    assert any("DELETE VERTEX FROM (SELECT FROM Project" in s for s, _ in db.commands)
    assert result["conversations"] == 1


def test_delete_conversation_cascades_children() -> None:
    db = FakeDb(affected=1)
    asyncio.run(repo.delete_conversation(db, "u", "c1"))
    deleted = [s for s, _ in db.commands if s.startswith("DELETE VERTEX")]
    # Messages, RunMessages, Documents, LogEntries, then the conversation itself.
    assert any("FROM Message" in s for s in deleted)
    assert any("FROM RunMessages" in s for s in deleted)
    assert any("FROM Document" in s for s in deleted)
    assert any("FROM LogEntry" in s for s in deleted)
    assert any("FROM Conversation" in s for s in deleted)


# --------------------------------------------------------------------------- #
# Repository: conversation listing + lifecycle
# --------------------------------------------------------------------------- #
def test_list_conversations_orders_pinned_first_and_hides_archived() -> None:
    db = FakeDb(rows=[{"conversation_id": "c1", "pinned": True, "archived": None}])
    rows = asyncio.run(repo.list_conversations(db, "u"))
    sql, _ = db.queries[0]
    assert "ORDER BY pinned DESC, started_at DESC" in sql
    assert "archived IS NULL OR archived = false" in sql  # default hides archived
    assert rows[0]["pinned"] is True and rows[0]["archived"] is False  # coerced to bool


def test_list_conversations_include_archived_drops_filter() -> None:
    db = FakeDb(rows=[])
    asyncio.run(repo.list_conversations(db, "u", include_archived=True))
    sql, _ = db.queries[0]
    assert "archived IS NULL OR archived = false" not in sql  # no WHERE filter on archived


def test_create_conversation_stamps_project() -> None:
    db = FakeDb(rows=[{"n": 0}])
    asyncio.run(repo.create_conversation(db, "u", "c1", project_id="p1"))
    create = next(p for s, p in db.commands if s.startswith("CREATE VERTEX Conversation"))
    assert create["pid"] == "p1"


def test_set_conversation_lifecycle_setters() -> None:
    db = FakeDb()
    asyncio.run(repo.set_conversation_pinned(db, "c", True))
    asyncio.run(repo.set_conversation_archived(db, "c", True))
    asyncio.run(repo.set_conversation_project_id(db, "c", None))
    sqls = [s for s, _ in db.commands]
    assert any("SET pinned" in s for s in sqls)
    assert any("SET archived" in s for s in sqls)
    assert any("SET project_id" in s for s in sqls)


# --------------------------------------------------------------------------- #
# Repository: document scoping + global + search
# --------------------------------------------------------------------------- #
def test_create_document_project_scope_anchors_edge_to_project() -> None:
    db = FakeDb()
    asyncio.run(
        repo.create_document(
            db, "u", title="spec", content="body", project_id="p1", is_global=False,
            embedding=[0.1, 0.2],
        )
    )
    create = next(p for s, p in db.commands if s.startswith("CREATE VERTEX Document"))
    assert create["pid"] == "p1" and create["glob"] is False and create["emb"] == [0.1, 0.2]
    assert "conversation_id" not in create  # project-scoped, not conversation-scoped
    edge = next(s for s, _ in db.commands if s.startswith("CREATE EDGE HAS_DOCUMENT"))
    assert "FROM Project" in edge


def test_create_document_global_anchors_edge_to_user() -> None:
    db = FakeDb()
    asyncio.run(repo.create_document(db, "u", title="ref", content="x", is_global=True))
    edge = next(s for s, _ in db.commands if s.startswith("CREATE EDGE HAS_DOCUMENT"))
    assert "FROM User" in edge  # global docs link to the user, never orphaned


def test_list_documents_project_and_global_scope() -> None:
    db = FakeDb(rows=[{"document_id": "d1", "is_global": True}])
    asyncio.run(repo.list_documents(db, "u", project_id="p1", include_global=True))
    sql, params = db.queries[0]
    assert "project_id = :pid" in sql and "is_global = true" in sql and " OR " in sql
    assert params["pid"] == "p1"


def test_search_documents_like_fallback_scope() -> None:
    db = FakeDb(rows=[{"document_id": "d1", "title": "t", "content": "hello world"}])
    hits = asyncio.run(
        repo.search_documents(db, "u", "hello", project_id="p1", include_global=True)
    )
    assert [h["document_id"] for h in hits] == ["d1"]
    sql, params = db.queries[0]
    assert "content LIKE :pat" in sql and "project_id = :pid" in sql and "is_global = true" in sql


def test_search_documents_no_scope_returns_empty() -> None:
    db = FakeDb(rows=[{"document_id": "d1"}])
    hits = asyncio.run(
        repo.search_documents(db, "u", "x", project_id=None, include_global=False)
    )
    assert hits == [] and db.queries == []  # nothing to scope to → no query at all


def test_set_document_global_updates_flag() -> None:
    db = FakeDb(affected=1)
    n = asyncio.run(repo.set_document_global(db, "u", "d1", True))
    assert n == 1
    sql, params = db.commands[0]
    assert sql.startswith("UPDATE Document SET is_global") and params["g"] is True


# --------------------------------------------------------------------------- #
# Capability: project document tools
# --------------------------------------------------------------------------- #
def test_project_doc_tools_registered() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, deps_type=GraphDependencies, capabilities=[*build_documents()])
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c")
    asyncio.run(agent.run("hi", deps=deps))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert {"list_project_documents", "read_project_document", "search_project_documents"} <= names


def test_list_project_documents_scopes_to_project_and_global() -> None:
    db = FakeDb(rows=[{"document_id": "d1", "title": "spec", "is_global": False}])
    docs = asyncio.run(list_project_documents(_ctx(db, project_id="p1")))
    assert [d.document_id for d in docs] == ["d1"]
    _, params = db.queries[0]
    assert params["pid"] == "p1"


def test_read_project_document_unknown_id_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        asyncio.run(read_project_document(_ctx(FakeDb(rows=[])), "nope"))


def test_search_project_documents_is_tolerant() -> None:
    class BoomDb(FakeDb):
        async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            raise RuntimeError("db down")

    hits = asyncio.run(search_project_documents(_ctx(BoomDb(), project_id="p1"), "q"))
    assert hits == []  # never raises


# --------------------------------------------------------------------------- #
# System prompt: layering + manifest
# --------------------------------------------------------------------------- #
def test_compose_instructions_base_only_unchanged() -> None:
    assert compose_instructions() == BASE_SYSTEM_PROMPT
    assert compose_instructions("", "") == BASE_SYSTEM_PROMPT


def test_compose_instructions_layers_project_then_conversation() -> None:
    out = compose_instructions("Conversation rule.", "Project rule.")
    assert out.startswith(BASE_SYSTEM_PROMPT)
    assert "PROJECT INSTRUCTIONS" in out and "Project rule." in out
    assert "ADDITIONAL INSTRUCTIONS" in out and "Conversation rule." in out
    # Project layer comes before the conversation layer (more specific last).
    assert out.index("PROJECT INSTRUCTIONS") < out.index("ADDITIONAL INSTRUCTIONS")


def test_compose_instructions_omits_empty_layers() -> None:
    out = compose_instructions("Only conversation.", "")
    assert "PROJECT INSTRUCTIONS" not in out and "Only conversation." in out


def test_project_documents_block_manifest() -> None:
    db = RoutedDb(
        {
            "FROM Document": [
                {"document_id": "d1", "title": "Spec", "is_global": False},
                {"document_id": "d2", "title": "Policy", "is_global": True},
            ],
            "FROM Project": [{"project_id": "p1", "title": "Acme"}],
        }
    )
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c", project_id="p1")
    block = asyncio.run(project_documents_block(deps))
    assert 'project "Acme"' in block
    assert "Spec (d1)" in block and "Policy (d2) [global]" in block


def test_project_documents_block_empty_when_no_docs() -> None:
    deps = GraphDependencies(db=FakeDb(rows=[]), user_id="u", conversation_id="c", project_id="p1")
    assert asyncio.run(project_documents_block(deps)) == ""


def test_project_documents_block_tolerant_on_error() -> None:
    class BoomDb(FakeDb):
        async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            raise RuntimeError("db down")

    deps = GraphDependencies(db=BoomDb(), user_id="u", conversation_id="c", project_id="p1")
    assert asyncio.run(project_documents_block(deps)) == ""


# --------------------------------------------------------------------------- #
# main.build_agent wiring
# --------------------------------------------------------------------------- #
def test_build_agent_regular_mode_has_project_doc_tools() -> None:
    agent = build_agent(model="test", mode="regular")
    model = TestModel(call_tools=[])
    with agent.override(model=model):
        deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c")
        asyncio.run(agent.run("hi", deps=deps))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert {"list_project_documents", "search_project_documents"} <= names

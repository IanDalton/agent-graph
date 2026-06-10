"""Tests for the ArcadeDB conversation-memory capability.

The unit tests use a recording fake client and need no database. The single
integration test talks to a running ArcadeDB and is skipped automatically when
the server is unreachable (start it with ``docker compose up -d arcadedb``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient
from backend.db.dependencies import GraphDependencies
from backend.skills.graph_capability import build_memory

EXPECTED_TOOLS = {"search_memory", "get_conversation_history", "store_fact", "run_query"}


class RecordingClient:
    """Duck-typed stand-in for ArcadeClient that records commands instead of executing them."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        # create_conversation checks an existence count; return 0 so it proceeds.
        if "count(" in sql.lower():
            return [{"n": 0}]
        return []


def _make_agent(model: TestModel) -> Agent:
    return Agent(model, deps_type=GraphDependencies, capabilities=[*build_memory()])


def test_tools_are_registered() -> None:
    model = TestModel(call_tools=[])
    agent = _make_agent(model)
    db = RecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    asyncio.run(agent.run("hi", deps=deps))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert EXPECTED_TOOLS <= names


def test_hooks_persist_turn() -> None:
    """before_run creates the conversation; after_run appends user + assistant messages."""
    model = TestModel(call_tools=[])
    agent = _make_agent(model)
    db = RecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    asyncio.run(agent.run("hello", deps=deps))

    created_msgs = [
        params for sql, params in db.commands if sql.startswith("CREATE VERTEX Message")
    ]
    roles = {p["role"]: p["content"] for p in created_msgs}
    assert roles.get("user") == "hello"
    assert "assistant" in roles  # TestModel's reply was persisted
    # The conversation vertex was created exactly once (before_run, idempotent).
    assert sum(1 for sql, _ in db.commands if sql.startswith("CREATE VERTEX Conversation")) == 1


@pytest.mark.parametrize(
    "query,allowed",
    [
        ("SELECT FROM Message", True),
        ("  select from Fact  ", True),
        ("MATCH {type: Message} RETURN $elements", True),
        ("TRAVERSE out() FROM Message", True),
        ("DELETE FROM Message", False),
        ("UPDATE Message SET x = 1", False),
        ("CREATE VERTEX Message", False),
        ("", False),
    ],
)
def test_run_query_read_only_guard(query: str, allowed: bool) -> None:
    """The raw-query escape hatch must only permit read-only statements."""
    from backend.skills.graph_capability import is_read_only

    assert is_read_only(query) is allowed


# --------------------------------------------------------------------------- #
# Integration test (requires a running ArcadeDB)
# --------------------------------------------------------------------------- #
def _db_reachable() -> bool:
    try:
        return httpx.get("http://localhost:2480/api/v1/ready", timeout=1).status_code in (200, 204)
    except Exception:
        return False


@pytest.mark.skipif(not _db_reachable(), reason="ArcadeDB not running on localhost:2480")
def test_repository_roundtrip_integration() -> None:
    async def main():
        async with ArcadeClient() as db:
            await db.ensure_schema()
            cid = "itest-conv"
            await repo.create_conversation(db, "itest-user", cid)
            await repo.append_message(db, "itest-user", cid, "user", "integration recoleta probe")
            msgs = await repo.get_recent_messages(db, cid)
            assert any("recoleta probe" in m["content"] for m in msgs)
            hits = await repo.search_messages(db, "itest-user", "recoleta probe")
            assert hits

    asyncio.run(main())

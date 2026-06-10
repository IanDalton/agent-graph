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
from backend.db.arcade_db import ArcadeClient, database_name_for_user
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


def test_database_name_for_user_is_isolated_and_safe() -> None:
    """Each user maps to a distinct, ArcadeDB-safe database name."""
    name = database_name_for_user("u1", base="AgentMemory")
    # Only ArcadeDB-safe characters.
    assert name.replace("_", "").isalnum()
    assert name.startswith("AgentMemory_")
    # Different users -> different databases.
    assert database_name_for_user("u1") != database_name_for_user("u2")
    # Same user -> stable name (so the DB is reused across runs).
    assert database_name_for_user("u1") == database_name_for_user("u1")
    # Ids that collapse to the same sanitized form stay distinct via the hash.
    assert database_name_for_user("a.b") != database_name_for_user("a-b")


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


def test_post_retries_transient_503_then_succeeds() -> None:
    """A burst of writes can make ArcadeDB answer 503; _post retries until it clears."""

    async def main() -> None:
        async with ArcadeClient(database="db", max_retries=5, retry_base_delay=0) as db:
            calls = {"n": 0}
            req = httpx.Request("POST", "http://localhost:2480/api/v1/command/db")

            async def fake_post(path: str, json: dict[str, Any]) -> httpx.Response:
                calls["n"] += 1
                # Fail the first two attempts with 503, then succeed.
                status = 503 if calls["n"] <= 2 else 200
                payload = b'{"result": [{"ok": 1}]}' if status == 200 else b""
                return httpx.Response(status, content=payload, request=req)

            db._client.post = fake_post  # type: ignore[assignment]
            result = await db.command("CREATE VERTEX Fact SET text = 'x'")
            assert calls["n"] == 3  # two 503s, then a success
            assert result == [{"ok": 1}]

    asyncio.run(main())


def test_post_raises_after_exhausting_retries() -> None:
    """If 503 never clears, the original HTTP error surfaces rather than hanging."""

    async def main() -> None:
        async with ArcadeClient(database="db", max_retries=2, retry_base_delay=0) as db:
            req = httpx.Request("POST", "http://localhost:2480/api/v1/command/db")

            async def always_503(path: str, json: dict[str, Any]) -> httpx.Response:
                return httpx.Response(503, request=req)

            db._client.post = always_503  # type: ignore[assignment]
            with pytest.raises(httpx.HTTPStatusError):
                await db.command("CREATE VERTEX Fact SET text = 'x'")

    asyncio.run(main())


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
    """End-to-end against a real, per-user database that is created then dropped."""
    db_name = database_name_for_user("itest-user", base="AgentMemoryTest")

    async def main():
        async with ArcadeClient(database=db_name) as db:
            try:
                await db.ensure_database()
                assert await db.database_exists()
                await db.ensure_schema()
                cid = "itest-conv"
                await repo.create_conversation(db, "itest-user", cid)
                await repo.append_message(db, "itest-user", cid, "user", "integration recoleta probe")
                msgs = await repo.get_recent_messages(db, cid)
                assert any("recoleta probe" in m["content"] for m in msgs)
                hits = await repo.search_messages(db, "itest-user", "recoleta probe")
                assert hits
            finally:
                # Self-cleaning: drop the throwaway per-user database.
                await db._server_command(f"drop database {db_name}")

    asyncio.run(main())

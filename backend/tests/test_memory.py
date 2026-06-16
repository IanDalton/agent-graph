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
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient, database_name_for_user
from backend.db.dependencies import GraphDependencies
from backend import title_generation
from backend.skills.graph_capability import build_memory, delete_fact, update_fact

EXPECTED_TOOLS = {
    "search_memory",
    "get_conversation_history",
    "store_fact",
    "run_query",
    "update_fact",
    "delete_fact",
}


class RecordingClient:
    """Duck-typed stand-in for ArcadeClient that records commands instead of executing them."""

    def __init__(self, fact_count: int = 1) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._fact_count = fact_count  # rows UPDATE/DELETE on Fact should report as affected

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        # UPDATE/DELETE report affected rows as [{"count": N}].
        if sql.strip().upper().startswith(("UPDATE", "DELETE")):
            return [{"count": self._fact_count}]
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        # create_conversation checks an existence count; return 0 so it proceeds.
        if "count(" in sql.lower():
            return [{"n": 0}]
        return []


class Failing503Client(RecordingClient):
    """Like RecordingClient, but every write (command) fails with a 503, as during DB overload."""

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        req = httpx.Request("POST", "http://localhost:2480/api/v1/command/db")
        raise httpx.HTTPStatusError("503 Service Unavailable", request=req, response=httpx.Response(503, request=req))


class TitleRecordingClient(RecordingClient):
    """Recording client with canned reads for title generation."""

    def __init__(self) -> None:
        super().__init__()
        # Repo.get_recent_messages reverses query results, so this is newest-first.
        self._message_rows = [
            {"role": "assistant", "content": "Let's narrow it down.", "created_at": "2"},
            {"role": "user", "content": "I need a hotel in Tokyo", "created_at": "1"},
        ]

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        normalized = sql.upper()
        if "SELECT COUNT(*) AS N FROM CONVERSATION" in normalized:
            return [{"n": 0}]
        if "SELECT TITLE FROM CONVERSATION" in normalized:
            return [{"title": ""}]
        if "CONTENT, CREATED_AT" in normalized and "FROM MESSAGE" in normalized:
            return self._message_rows
        if "COUNT(" in normalized:
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


def test_hooks_generate_title_after_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the first completed turn, a blank conversation title is generated automatically."""
    monkeypatch.setattr(
        title_generation,
        "select_model",
        lambda *_args, **_kwargs: TestModel(custom_output_text="Tokyo hotel search"),
    )
    model = TestModel(call_tools=[])
    agent = _make_agent(model)
    db = TitleRecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    asyncio.run(agent.run("hello", deps=deps))

    updates = [params for sql, params in db.commands if sql.startswith("UPDATE Conversation SET title")]
    assert updates and updates[-1]["title"] == "Tokyo hotel search"


def test_title_model_label_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TITLE_MODEL", raising=False)
    monkeypatch.setenv("TITLE_OLLAMA_MODEL", "supra-title-350m-exp")
    assert title_generation.title_model_label() == "ollama/supra-title-350m-exp"

    monkeypatch.setenv("TITLE_MODEL", "openai:gpt-5.2")
    assert title_generation.title_model_label() == "openai:gpt-5.2"


def test_run_messages_blob_round_trips_tool_calls() -> None:
    """Regression: replayed history must carry tool calls + returns, not just text.

    The agent used to re-doubt completed tool work because only role/content text was replayed.
    after_run now also persists the run's serialized messages; _to_message_history must rebuild
    them faithfully, ToolCallPart included.
    """
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    from backend.main import _to_message_history

    # TestModel calls the named tool once, producing a ToolCallPart + tool return in the run.
    model = TestModel(call_tools=["get_conversation_history"])
    agent = _make_agent(model)
    db = RecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    asyncio.run(agent.run("hi", deps=deps))

    blobs = [p["raw"] for sql, p in db.commands if sql.startswith("CREATE VERTEX RunMessages")]
    assert blobs, "after_run must persist a RunMessages blob"

    history = _to_message_history([{"raw": blobs[0]}])
    tool_calls = [
        part
        for msg in history
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    assert any(tc.tool_name == "get_conversation_history" for tc in tool_calls)


def test_persistence_failure_does_not_crash_the_run() -> None:
    """A DB write failure in the persistence hooks must be swallowed, not crash the agent loop."""
    model = TestModel(call_tools=[])
    agent = _make_agent(model)
    db = Failing503Client()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    # Every command() raises 503; the run must still complete and return the model's output.
    result = asyncio.run(agent.run("hello", deps=deps))
    assert result.output
    # The hooks did attempt the writes (so persistence is best-effort, not skipped).
    assert db.commands


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


def _ctx(deps: GraphDependencies) -> RunContext[GraphDependencies]:
    """Minimal RunContext for invoking memory tool coroutines directly in unit tests."""
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def _http_error(status: int, body: bytes) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://localhost:2480/api/v1/query/db")
    resp = httpx.Response(status, content=body, request=req)
    return httpx.HTTPStatusError(f"{status}", request=req, response=resp)


class _QueryRaisingClient(RecordingClient):
    """RecordingClient whose query() raises a preset HTTPStatusError (simulates a DB query error)."""

    def __init__(self, error: httpx.HTTPStatusError) -> None:
        super().__init__()
        self._error = error

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise self._error


def test_run_query_missing_type_returns_no_records_not_retry() -> None:
    """A SELECT against a not-yet-created type (ArcadeDB 500 SchemaException) must come back as a
    graceful 'no records' result, NOT a ModelRetry — run_query has max_retries=1, so retrying would
    crash the run with UnexpectedModelBehavior on the second such error. Querying a missing type is
    the normal 'check before create' path, and the truthful answer is 'there are none'."""
    from backend.schemas.graph_schemas import RawQuery
    from backend.skills.graph_capability import run_query

    body = (
        b'{"error":"Error on transaction commit",'
        b'"detail":"Type with name \'Company\' was not found",'
        b'"exception":"com.arcadedb.exception.SchemaException"}'
    )
    deps = GraphDependencies(db=_QueryRaisingClient(_http_error(500, body)), user_id="u", conversation_id="c")
    result = asyncio.run(
        run_query(_ctx(deps), RawQuery(query="SELECT FROM Company WHERE name = 'DoorLink'", rationale="check existence"))
    )
    # A normal result the model can read, not an exception.
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["result"] == "no_records"
    assert "Company" in result[0]["detail"]  # the DB's detail is surfaced
    assert "list_vertex_types" in result[0]["hint"]


def test_run_query_transient_503_propagates() -> None:
    """A 503 is a genuine outage the model can't fix; it must surface, not turn into a ModelRetry."""
    from backend.schemas.graph_schemas import RawQuery
    from backend.skills.graph_capability import run_query

    deps = GraphDependencies(db=_QueryRaisingClient(_http_error(503, b"")), user_id="u", conversation_id="c")
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run_query(_ctx(deps), RawQuery(query="SELECT FROM Message", rationale="read")))


def test_update_fact_revises_in_place() -> None:
    db = RecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    msg = asyncio.run(update_fact(_ctx(deps), "fid123", "loves Recoleta"))
    updates = [(sql, p) for sql, p in db.commands if sql.startswith("UPDATE Fact SET text")]
    assert updates
    sql, params = updates[0]
    assert "WHERE fact_id = :fid AND user_id = :uid" in sql
    assert params["text"] == "loves Recoleta" and params["fid"] == "fid123" and params["uid"] == "u"
    assert "Updated fact fid123" in msg


def test_update_fact_missing_raises() -> None:
    db = RecordingClient(fact_count=0)  # nothing matched the fact_id
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    with pytest.raises(ModelRetry):
        asyncio.run(update_fact(_ctx(deps), "nope", "x"))


def test_delete_fact_removes_by_id() -> None:
    db = RecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    msg = asyncio.run(delete_fact(_ctx(deps), "fid123"))
    assert any(sql.startswith("DELETE VERTEX FROM (SELECT FROM Fact") for sql, _ in db.commands)
    assert "Deleted fact fid123" in msg


# --------------------------------------------------------------------------- #
# Fact importance (curated context) + message embeddings
# --------------------------------------------------------------------------- #
def test_store_fact_tool_marks_important_by_default() -> None:
    from backend.schemas.graph_schemas import StoreFactArgs
    from backend.skills.graph_capability import store_fact

    db = RecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    asyncio.run(store_fact(_ctx(deps), StoreFactArgs(text="likes Recoleta")))
    created = [p for sql, p in db.commands if sql.startswith("CREATE VERTEX Fact SET")]
    assert created and created[0]["imp"] is True and created[0]["text"] == "likes Recoleta"


def test_store_fact_tool_respects_important_false() -> None:
    from backend.schemas.graph_schemas import StoreFactArgs
    from backend.skills.graph_capability import store_fact

    db = RecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    asyncio.run(store_fact(_ctx(deps), StoreFactArgs(text="incidental", important=False)))
    created = [p for sql, p in db.commands if sql.startswith("CREATE VERTEX Fact SET")]
    assert created and created[0]["imp"] is False


def test_update_fact_tool_can_flag_importance() -> None:
    db = RecordingClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    asyncio.run(update_fact(_ctx(deps), "fid123", "revised", important=False))
    updates = [(sql, p) for sql, p in db.commands if sql.startswith("UPDATE Fact SET text")]
    assert updates
    sql, params = updates[0]
    assert "important = :imp" in sql and params["imp"] is False


def test_set_fact_importance_repo_builds_scoped_update() -> None:
    db = RecordingClient()
    n = asyncio.run(repo.set_fact_importance(db, "u", "fid123", True))
    updates = [(sql, p) for sql, p in db.commands if sql.startswith("UPDATE Fact SET important")]
    assert updates and n == 1
    sql, params = updates[0]
    assert "WHERE fact_id = :fid AND user_id = :uid" in sql
    assert params["imp"] is True and params["fid"] == "fid123" and params["uid"] == "u"


def test_list_facts_important_only_filters() -> None:
    # important_only adds the `important <> false` guard (NULL legacy rows count as important).
    db = RecordingClient()
    asyncio.run(repo.list_facts(db, "u", important_only=True))
    asyncio.run(repo.list_facts(db, "u"))
    sqls = [sql for sql, _ in db.queries if sql.startswith("SELECT fact_id, text, important")]
    assert len(sqls) == 2
    assert "important <> false" in sqls[0]
    assert "important <> false" not in sqls[1]


class VectorRoutingClient(RecordingClient):
    """Routes Message reads: vectorNeighbors → ``vector`` rows, LIKE → ``like`` rows."""

    def __init__(self, vector: list[dict[str, Any]], like: list[dict[str, Any]]) -> None:
        super().__init__()
        self._vector = vector
        self._like = like

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        if "vectorNeighbors" in sql:
            return list(self._vector)
        if "LIKE" in sql.upper():
            return list(self._like)
        return []


def test_search_messages_uses_vector_path_when_it_has_hits() -> None:
    db = VectorRoutingClient(vector=[{"content": "semantic hit"}], like=[{"content": "like hit"}])
    hits = asyncio.run(repo.search_messages(db, "u", "probe", embedding=[0.1, 0.2]))
    assert hits == [{"content": "semantic hit"}]


def test_search_messages_falls_back_to_like_when_vector_empty() -> None:
    db = VectorRoutingClient(vector=[], like=[{"content": "like hit"}])
    hits = asyncio.run(repo.search_messages(db, "u", "probe", embedding=[0.1, 0.2]))
    assert hits == [{"content": "like hit"}]


def test_search_messages_like_only_without_embedding() -> None:
    db = VectorRoutingClient(vector=[{"content": "should-not-appear"}], like=[{"content": "like hit"}])
    hits = asyncio.run(repo.search_messages(db, "u", "probe"))
    assert hits == [{"content": "like hit"}]
    assert not any("vectorNeighbors" in sql for sql, _ in db.queries)


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

                # Facts default to important; list_facts surfaces them and set_fact_importance toggles.
                await repo.store_fact(db, "itest-user", "integration likes empanadas")
                facts = await repo.list_facts(db, "itest-user")
                assert facts and facts[0]["important"] is True
                fid = facts[0]["fact_id"]
                assert await repo.set_fact_importance(db, "itest-user", fid, False) == 1
                assert await repo.list_facts(db, "itest-user", important_only=True) == []
            finally:
                # Self-cleaning: drop the throwaway per-user database.
                await db._server_command(f"drop database {db_name}")

    asyncio.run(main())

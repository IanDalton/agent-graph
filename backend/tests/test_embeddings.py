"""Tests for semantic (vector) memory search: the Embedder, the repository vector path, and the
schema wiring. All unit tests run without a database or network (the embedder's httpx call is
monkeypatched; the repository uses a recording fake client).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient
from backend.db.dependencies import GraphDependencies
from backend.embeddings import Embedder, embedding_dimension, embeddings_enabled
from backend.schemas.graph_schemas import StoreFactArgs
from backend.skills.graph_capability import search_memory, store_fact


class FakeEmbedder:
    """Deterministic stand-in for Embedder: returns a fixed vector (or None to simulate disabled)."""

    def __init__(self, vector: list[float] | None = (0.1, 0.2, 0.3)) -> None:
        self._vector = list(vector) if vector is not None else None
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float] | None:
        self.calls.append(text)
        return list(self._vector) if self._vector is not None else None


class FactSearchClient:
    """Records queries/commands; returns preset vector rows, optionally raising on the vector query."""

    def __init__(self, vector_rows: list[dict[str, Any]] | None = None, vector_error: Exception | None = None) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._vector_rows = vector_rows
        self._vector_error = vector_error

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        if "vectorNeighbors" in sql:
            if self._vector_error:
                raise self._vector_error
            return self._vector_rows or []
        # The LIKE fallback query.
        return [{"fact_id": "like1", "text": "from like search", "created_at": "t"}]


def _ctx(deps: GraphDependencies) -> RunContext[GraphDependencies]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def test_embeddings_enabled_and_dimension_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("EMBED_DIM", raising=False)
    assert embeddings_enabled() is False
    assert embedding_dimension() is None

    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBED_DIM", "768")
    assert embeddings_enabled() is True
    assert embedding_dimension() == 768

    monkeypatch.setenv("EMBED_DIM", "not-a-number")
    assert embedding_dimension() is None


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #
def test_embedder_disabled_returns_none() -> None:
    async def main() -> None:
        async with Embedder(model=None) as e:
            assert e.enabled is False
            assert await e.embed("anything") is None

    asyncio.run(main())


def test_embedder_parses_ollama_response() -> None:
    async def main() -> None:
        async with Embedder(model="nomic-embed-text", provider="ollama") as e:
            async def fake_post(url: str, json: dict[str, Any]) -> httpx.Response:
                assert url.endswith("/api/embeddings")
                assert json["prompt"] == "hello"
                return httpx.Response(200, json={"embedding": [1.0, 2.0, 3.0]}, request=httpx.Request("POST", url))

            e._ensure_client().post = fake_post  # type: ignore[assignment]
            assert await e.embed("hello") == [1.0, 2.0, 3.0]

    asyncio.run(main())


def test_embedder_parses_openai_response() -> None:
    async def main() -> None:
        async with Embedder(model="text-embedding-3-small", provider="openai", api_key="k") as e:
            async def fake_post(url: str, json: dict[str, Any]) -> httpx.Response:
                assert url.endswith("/embeddings")
                assert json["input"] == "hi"
                return httpx.Response(
                    200, json={"data": [{"embedding": [0.5, 0.6]}]}, request=httpx.Request("POST", url)
                )

            e._ensure_client().post = fake_post  # type: ignore[assignment]
            assert await e.embed("hi") == [0.5, 0.6]

    asyncio.run(main())


def test_embedder_returns_none_on_error() -> None:
    """Any failure degrades to None (so recall falls back to LIKE), never raises."""

    async def main() -> None:
        async with Embedder(model="m", provider="ollama", max_retries=0) as e:
            async def boom(url: str, json: dict[str, Any]) -> httpx.Response:
                raise httpx.ConnectError("no route")

            e._ensure_client().post = boom  # type: ignore[assignment]
            assert await e.embed("hello") is None

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# repository.search_facts vector path + fallback
# --------------------------------------------------------------------------- #
def test_search_facts_uses_vector_path_when_embedding_given() -> None:
    rows = [{"fact_id": "v1", "text": "semantic hit", "created_at": "t"}]
    db = FactSearchClient(vector_rows=rows)
    out = asyncio.run(repo.search_facts(db, "u", "query text", embedding=[0.1, 0.2, 0.3]))
    assert out == rows
    vquery = [q for q in db.queries if "vectorNeighbors" in q[0]]
    assert vquery and vquery[0][1]["qvec"] == [0.1, 0.2, 0.3]
    assert vquery[0][1]["uid"] == "u"


def test_search_facts_without_embedding_uses_like() -> None:
    db = FactSearchClient()
    out = asyncio.run(repo.search_facts(db, "u", "query text"))
    assert out[0]["fact_id"] == "like1"
    assert not any("vectorNeighbors" in q[0] for q in db.queries)


def test_search_facts_falls_back_to_like_on_vector_error() -> None:
    req = httpx.Request("POST", "http://localhost:2480/api/v1/query/db")
    err = httpx.HTTPStatusError("boom", request=req, response=httpx.Response(500, request=req))
    db = FactSearchClient(vector_error=err)
    out = asyncio.run(repo.search_facts(db, "u", "query", embedding=[0.1, 0.2, 0.3]))
    assert out[0]["fact_id"] == "like1"  # fell back to LIKE despite an embedding


def test_search_facts_falls_back_to_like_when_vector_empty() -> None:
    db = FactSearchClient(vector_rows=[])  # no embedded facts yet
    out = asyncio.run(repo.search_facts(db, "u", "query", embedding=[0.1, 0.2, 0.3]))
    assert out[0]["fact_id"] == "like1"


# --------------------------------------------------------------------------- #
# repository.store_fact embedding wiring
# --------------------------------------------------------------------------- #
def test_store_fact_includes_embedding_when_given() -> None:
    db = FactSearchClient()
    asyncio.run(repo.store_fact(db, "u", "likes cats", embedding=[0.1, 0.2, 0.3]))
    creates = [(sql, p) for sql, p in db.commands if sql.startswith("CREATE VERTEX Fact SET")]
    assert creates
    sql, params = creates[0]
    assert "embedding = :emb" in sql
    assert params["emb"] == [0.1, 0.2, 0.3]


def test_store_fact_omits_embedding_when_none() -> None:
    db = FactSearchClient()
    asyncio.run(repo.store_fact(db, "u", "likes cats"))
    creates = [(sql, p) for sql, p in db.commands if sql.startswith("CREATE VERTEX Fact SET")]
    assert creates and "embedding" not in creates[0][0]


# --------------------------------------------------------------------------- #
# Capability: search_memory / store_fact embed via deps.embedder
# --------------------------------------------------------------------------- #
def test_search_memory_embeds_query_and_uses_vector_path() -> None:
    rows = [{"fact_id": "v1", "text": "semantic hit", "created_at": "t"}]
    db = FactSearchClient(vector_rows=rows)
    embedder = FakeEmbedder(vector=[0.1, 0.2, 0.3])
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c", embedder=embedder)
    result = asyncio.run(search_memory(_ctx(deps), "where do I live"))
    # The embedder was consulted and the vector query ran.
    assert embedder.calls == ["where do I live"]
    assert any("vectorNeighbors" in q[0] for q in db.queries)
    assert any(h.content == "semantic hit" for h in result.hits)


def test_store_fact_tool_embeds_text() -> None:
    db = FactSearchClient()
    embedder = FakeEmbedder(vector=[0.4, 0.5, 0.6])
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c", embedder=embedder)
    asyncio.run(store_fact(_ctx(deps), StoreFactArgs(text="allergic to peanuts")))
    assert embedder.calls == ["allergic to peanuts"]
    creates = [(sql, p) for sql, p in db.commands if sql.startswith("CREATE VERTEX Fact SET")]
    assert creates and creates[0][1]["emb"] == [0.4, 0.5, 0.6]


def test_search_memory_without_embedder_uses_like() -> None:
    db = FactSearchClient()
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")  # no embedder
    asyncio.run(search_memory(_ctx(deps), "anything"))
    assert not any("vectorNeighbors" in q[0] for q in db.queries)


# --------------------------------------------------------------------------- #
# Schema: vector property + index added only when EMBED_DIM is configured
# --------------------------------------------------------------------------- #
def test_ensure_schema_adds_vector_index_when_dim_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_DIM", "3")

    async def main() -> None:
        # Unique db name so the process-level _ensured cache doesn't short-circuit.
        db = ArcadeClient(database="VecSchemaProbe_unit")
        recorded: list[str] = []

        async def fake_command(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            recorded.append(sql)
            return []

        db.command = fake_command  # type: ignore[assignment]
        await db.ensure_schema()
        await db.aclose()
        assert any("Fact.embedding" in s and "ARRAY_OF_FLOATS" in s for s in recorded)
        assert any("LSM_VECTOR" in s and '"dimensions": 3' in s for s in recorded)

    asyncio.run(main())


def test_ensure_schema_skips_vector_index_when_dim_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBED_DIM", raising=False)

    async def main() -> None:
        db = ArcadeClient(database="NoVecSchemaProbe_unit")
        recorded: list[str] = []

        async def fake_command(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            recorded.append(sql)
            return []

        db.command = fake_command  # type: ignore[assignment]
        await db.ensure_schema()
        await db.aclose()
        assert not any("LSM_VECTOR" in s for s in recorded)

    asyncio.run(main())

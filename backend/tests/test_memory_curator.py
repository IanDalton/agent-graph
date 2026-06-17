"""Tests for the background memory curator (facts + durable user profile).

Unit tests use recording/duck-typed fakes and need no database, network, or embedder. The single
integration test talks to a running ArcadeDB and skips automatically when it is unreachable.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend import memory_curator
from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient, database_name_for_user
from backend.db.dependencies import GraphDependencies
from backend.skills.system_prompt import user_profile_block


class CuratorClient:
    """Duck-typed ArcadeClient stand-in for the curator's reads/writes.

    Routes the handful of reads ``curate_memory``/``maybe_curate_memory`` issue (message count,
    curation watermark, recent messages, facts, profile) and records every command so the test can
    assert what was written.
    """

    def __init__(
        self,
        *,
        message_count: int = 0,
        watermark: int = 0,
        messages: list[dict[str, Any]] | None = None,
        facts: list[dict[str, Any]] | None = None,
        profile: str = "",
    ) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._message_count = message_count
        self._watermark = watermark
        # get_recent_messages reverses the rows, so order here is newest-first.
        self._messages = messages if messages is not None else []
        self._facts = facts if facts is not None else []
        self._profile = profile

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        if sql.strip().upper().startswith(("UPDATE", "DELETE")):
            return [{"count": 1}]
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        upper = sql.upper()
        if "COUNT(*) AS N FROM MESSAGE" in upper:
            return [{"n": self._message_count}]
        if "MEMORY_CURATED_MESSAGE_COUNT FROM CONVERSATION" in upper:
            return [{"memory_curated_message_count": self._watermark}]
        if "FROM MESSAGE" in upper:
            return list(self._messages)
        if "FROM FACT" in upper:
            return list(self._facts)
        if "PROFILE, PROFILE_UPDATED_AT FROM USER" in upper:
            return [{"profile": self._profile, "profile_updated_at": None}]
        return []


def _deps(db: Any) -> GraphDependencies:
    return GraphDependencies(db=db, user_id="u", conversation_id="c")


# --------------------------------------------------------------------------- #
# Threshold gate
# --------------------------------------------------------------------------- #
def test_maybe_curate_skips_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    async def _spy(_deps: GraphDependencies) -> None:
        called["n"] += 1

    monkeypatch.setattr(memory_curator, "curate_memory", _spy)
    monkeypatch.setattr(memory_curator, "MEMORY_CURATION_EVERY_N_MESSAGES", 8)
    # 4 new messages since the last curation (watermark 2, count 6) — below the threshold of 8.
    db = CuratorClient(message_count=6, watermark=2)
    asyncio.run(memory_curator.maybe_curate_memory(_deps(db)))
    assert called["n"] == 0


def test_maybe_curate_runs_at_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    async def _spy(_deps: GraphDependencies) -> None:
        called["n"] += 1

    monkeypatch.setattr(memory_curator, "curate_memory", _spy)
    monkeypatch.setattr(memory_curator, "MEMORY_CURATION_EVERY_N_MESSAGES", 8)
    # 8 new messages since the last curation (watermark 0, count 8) — meets the threshold.
    db = CuratorClient(message_count=8, watermark=0)
    asyncio.run(memory_curator.maybe_curate_memory(_deps(db)))
    assert called["n"] == 1


def test_maybe_curate_disabled_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    async def _spy(_deps: GraphDependencies) -> None:
        called["n"] += 1

    monkeypatch.setattr(memory_curator, "curate_memory", _spy)
    monkeypatch.setattr(memory_curator, "MEMORY_CURATION_ENABLED", False)
    db = CuratorClient(message_count=100, watermark=0)
    asyncio.run(memory_curator.maybe_curate_memory(_deps(db)))
    assert called["n"] == 0


def test_maybe_curate_skips_empty_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    async def _spy(_deps: GraphDependencies) -> None:
        called["n"] += 1

    monkeypatch.setattr(memory_curator, "curate_memory", _spy)
    db = CuratorClient(message_count=0, watermark=0)
    asyncio.run(memory_curator.maybe_curate_memory(_deps(db)))
    assert called["n"] == 0


# --------------------------------------------------------------------------- #
# The curator agent drives the tools (facts + profile) and advances the watermark
# --------------------------------------------------------------------------- #
def test_curate_memory_drives_fact_and_profile_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scripted FunctionModel: store a fact, rewrite the profile, then finish.

    Asserts the curator persists the fact and the new profile and advances the curation watermark.
    """
    # Use a real model (TestModel/FunctionModel) instead of resolving env/Ollama.
    monkeypatch.setattr(memory_curator, "resolve_model", lambda _label: _curator_script())

    db = CuratorClient(
        message_count=12,
        watermark=0,
        messages=[{"role": "user", "content": "I'm a backend engineer in Buenos Aires."}],
        facts=[{"fact_id": "f1", "text": "old fact"}],
        profile="",
    )
    asyncio.run(memory_curator.curate_memory(_deps(db)))

    # The fact was stored and the profile was rewritten via the tools.
    assert any(sql.startswith("CREATE VERTEX Fact SET") for sql, _ in db.commands)
    profile_updates = [p for sql, p in db.commands if sql.startswith("UPDATE User SET profile")]
    assert profile_updates and "backend engineer" in profile_updates[0]["p"]
    # The watermark advanced to the current message count so the gate moves forward.
    wm = [p for sql, p in db.commands if sql.startswith("UPDATE Conversation SET memory_curated_message_count")]
    assert wm and wm[0]["n"] == 12


def _curator_script() -> FunctionModel:
    """A FunctionModel that calls store_fact then update_user_profile, then replies done.

    Stateful across the agent's model calls: it inspects how many tool returns it has already
    seen to decide the next action, so it emits exactly one of each tool call and then stops.
    """

    def respond(messages: list[ModelMessage], _info: Any) -> ModelResponse:
        returns = sum(
            1
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if p.__class__.__name__ == "ToolReturnPart"
        )
        if returns == 0:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="store_fact", args={"args": {"text": "is a backend engineer in Buenos Aires"}})]
            )
        if returns == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="update_user_profile",
                        args={"profile": "## Who\nA backend engineer in Buenos Aires."},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="Curated: 1 fact, profile updated.")])

    return FunctionModel(respond)


def test_update_user_profile_tool_writes_full_profile() -> None:
    """The profile tool replaces the profile in full (and upserts the User vertex)."""
    db = CuratorClient()
    ctx: RunContext[GraphDependencies] = RunContext(deps=_deps(db), model=TestModel(), usage=RunUsage())
    msg = asyncio.run(memory_curator.update_user_profile(ctx, "  the whole profile  "))
    updates = [p for sql, p in db.commands if sql.startswith("UPDATE User SET profile")]
    assert updates and updates[0]["p"] == "the whole profile"  # trimmed
    assert "Updated user profile" in msg


# --------------------------------------------------------------------------- #
# user_profile_block (system-prompt injection) — formatting + tolerance
# --------------------------------------------------------------------------- #
def test_user_profile_block_formats_profile() -> None:
    db = CuratorClient(profile="Backend engineer; prefers concise answers.")
    block = asyncio.run(user_profile_block(_deps(db)))
    assert block.startswith("What we know about this user")
    assert "Backend engineer" in block


def test_user_profile_block_empty_when_no_profile() -> None:
    assert asyncio.run(user_profile_block(_deps(CuratorClient(profile="")))) == ""


class _RaisingProfileClient(CuratorClient):
    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise RuntimeError("db down")


def test_user_profile_block_tolerant_of_db_errors() -> None:
    assert asyncio.run(user_profile_block(_deps(_RaisingProfileClient()))) == ""


# --------------------------------------------------------------------------- #
# Integration test (requires a running ArcadeDB)
# --------------------------------------------------------------------------- #
def _db_reachable() -> bool:
    try:
        return httpx.get("http://localhost:2480/api/v1/ready", timeout=1).status_code in (200, 204)
    except Exception:
        return False


@pytest.mark.skipif(not _db_reachable(), reason="ArcadeDB not running on localhost:2480")
def test_profile_and_watermark_roundtrip_integration() -> None:
    db_name = database_name_for_user("itest-curator", base="AgentMemoryTest")

    async def main() -> None:
        async with ArcadeClient(database=db_name) as db:
            try:
                await db.ensure_database()
                await db.ensure_schema()
                cid = "itest-curator-conv"
                await repo.create_conversation(db, "itest-curator", cid)

                # Profile round-trips and replaces in full.
                assert (await repo.get_user_profile(db, "itest-curator"))["profile"] == ""
                await repo.set_user_profile(db, "itest-curator", "first profile")
                assert (await repo.get_user_profile(db, "itest-curator"))["profile"] == "first profile"
                await repo.set_user_profile(db, "itest-curator", "second profile")
                got = await repo.get_user_profile(db, "itest-curator")
                assert got["profile"] == "second profile" and got["profile_updated_at"]

                # Watermark round-trips (0 when unset).
                assert await repo.get_memory_curation_watermark(db, cid) == 0
                await repo.set_memory_curation_watermark(db, cid, 8)
                assert await repo.get_memory_curation_watermark(db, cid) == 8
            finally:
                await db._server_command(f"drop database {db_name}")

    asyncio.run(main())

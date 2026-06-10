"""HTTP/SSE API over the conversation-memory agent.

A thin wrapper that exposes the existing CLI machinery to a web frontend. It adds **no** new
database or agent logic: every handler calls into :mod:`backend.db.repository` and
:func:`backend.main.stream_run`, opening a short-lived :class:`ArcadeClient` pointed at the
caller's per-user database (mirroring :func:`backend.main.run`).

Run it with::

    uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import main, summarization
from backend.db import repository as repo
from backend.db.arcade_db import (
    DEFAULT_URL,
    ArcadeClient,
    database_name_for_user,
)

logger = logging.getLogger("agent_graph.api")

# Pool of long-lived clients, one per user database, reused across requests. Constructing an
# httpx client costs ~200ms (transport/SSL init) and opening a fresh connection per request adds
# more — so a per-request client made conversation loads slow (~700ms). Reusing a pooled,
# keep-alive client drops a message load to a single ~40ms query. Closed on shutdown via lifespan.
_clients: dict[str, ArcadeClient] = {}


def _get_client(user_id: str) -> ArcadeClient:
    """Return the pooled client for this user's database, creating it on first use."""
    dbname = database_name_for_user(user_id)
    client = _clients.get(dbname)
    if client is None:
        client = ArcadeClient(database=dbname)
        _clients[dbname] = client
    return client


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Pre-warm the default user's client on startup: construct the (slow ~200ms) httpx client,
    # open the connection, and ensure the schema once — so the first conversation load is already
    # fast (~40ms) instead of paying that init cost. Best-effort: a DB that's down must not block
    # startup; the client is created either way and warms lazily on first use.
    try:
        await _get_client("default").ensure_database()
        await _get_client("default").ensure_schema()
    except Exception:  # noqa: BLE001
        logger.warning("client pre-warm failed; will warm on first request", exc_info=True)
    yield
    for client in _clients.values():
        await client.aclose()
    _clients.clear()


app = FastAPI(title="agent-graph API", lifespan=_lifespan)

# The Vite dev server runs on a different origin; allow it (and common localhost variants)
# so the browser can call the API and read the SSE stream during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def _client_for(user_id: str, *, ensure: bool = False) -> AsyncIterator[ArcadeClient]:
    """Yield a pooled ArcadeDB client for this user (NOT closed here — it lives for the app).

    Reads default to ``ensure=False`` so a conversation load is a single fast query with no schema
    round-trips. Write paths pass ``ensure=True`` to guarantee the database/schema exist; that call
    is cheap after the first one thanks to ArcadeClient's process-level ensure cache.
    """
    client = _get_client(user_id)
    if ensure:
        await client.ensure_database()
        await client.ensure_schema()
    yield client


# --------------------------------------------------------------------------- config


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    """Read-only view of the runtime configuration (no secrets)."""
    agent_model = os.getenv("AGENT_MODEL")
    model_label = agent_model or f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen3')}"
    return {
        "model": model_label,
        "model_source": "AGENT_MODEL" if agent_model else "OLLAMA_MODEL (local fallback)",
        "arcade_url": os.getenv("ARCADE_URL", DEFAULT_URL),
        "searxng_url": os.getenv("SEARXNG_URL", "http://localhost:8085"),
        "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
    }


# --------------------------------------------------------------------- conversations


class NewConversation(BaseModel):
    user_id: str = "default"
    title: str | None = None


@app.get("/api/conversations")
async def list_conversations(user_id: str = "default") -> list[dict[str, Any]]:
    # Pure read, no schema ensure. Tolerant: a brand-new user whose database doesn't exist yet
    # simply has no conversations, so a query error maps to an empty list rather than a 500.
    try:
        async with _client_for(user_id) as db:
            return await repo.list_conversations(db, user_id)
    except Exception:  # noqa: BLE001
        logger.warning("list_conversations failed", exc_info=True)
        return []


@app.post("/api/conversations")
async def create_conversation(body: NewConversation) -> dict[str, Any]:
    conversation_id = uuid.uuid4().hex
    # The one common write path: ensure the database/schema exist before the first insert.
    async with _client_for(body.user_id, ensure=True) as db:
        await repo.create_conversation(db, body.user_id, conversation_id, title=body.title)
    return {
        "conversation_id": conversation_id,
        "title": body.title,
        "mode": "regular",
    }


@app.get("/api/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, user_id: str = "default") -> list[dict[str, Any]]:
    # Pure read, no schema ensure — a single ~40ms query on the pooled client. Tolerant: a missing
    # database/type (e.g. nothing written yet) yields an empty conversation, not an error.
    try:
        async with _client_for(user_id) as db:
            return await repo.get_recent_messages(db, conversation_id)
    except Exception:  # noqa: BLE001
        logger.warning("get_messages failed", exc_info=True)
        return []


@app.get("/api/conversations/{conversation_id}/summary")
async def get_summary(conversation_id: str, user_id: str = "default") -> dict[str, str]:
    """Return the cached conversation summary (a fast DB read).

    The summary is generated at write time — only every N messages — by the after-run hook
    (see :func:`backend.summarization.maybe_refresh_summary`), so this endpoint never runs an LLM
    and loads instantly. Tolerant: a read error yields an empty summary so the pane never breaks.
    """
    try:
        async with _client_for(user_id) as db:
            meta = await repo.get_conversation_summary(db, conversation_id)
        return {"summary": meta.get("summary") or ""}
    except Exception:  # noqa: BLE001 — the summary pane must never break the page.
        logger.warning("summary read failed", exc_info=True)
        return {"summary": ""}


@app.post("/api/conversations/{conversation_id}/summary")
async def refresh_summary(conversation_id: str, user_id: str = "default") -> dict[str, str]:
    """Force regeneration of the summary now (the only path that runs the LLM on demand).

    Backs the manual "refresh" button. Unlike the after-run hook, this ignores the message-count
    threshold and always regenerates, then stores and returns the new summary.
    """
    try:
        async with _client_for(user_id) as db:
            summary = await summarization.generate_summary(db, conversation_id)
        return {"summary": summary}
    except Exception:  # noqa: BLE001 — the summary pane must never break the page.
        logger.warning("summary refresh failed", exc_info=True)
        return {"summary": ""}


# ----------------------------------------------------------------------------- chat


class ChatRequest(BaseModel):
    user_id: str = "default"
    conversation_id: str
    prompt: str


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    """Stream one agent turn as Server-Sent Events.

    Each event from :func:`backend.main.stream_run` is written as one ``data:`` frame. Any
    error becomes a final ``{"type": "error"}`` frame so the client always gets a clean end
    instead of a dropped connection.
    """

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in main.stream_run(
                body.prompt, user_id=body.user_id, conversation_id=body.conversation_id
            ):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 — surface the failure to the client, don't 500 mid-stream.
            logger.error("chat stream failed: %s: %s", type(exc).__name__, exc)
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

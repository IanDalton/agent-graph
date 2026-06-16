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
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
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
from backend.embeddings import embeddings_enabled
from backend.model_selection import available_models, default_model_label

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
    """View of the runtime configuration (no secrets).

    ``model`` is the default model (used when a chat request sends no override); ``models`` is the
    selectable set for the UI dropdown (see :func:`backend.model_selection.available_models`). The
    chosen model is sent per-request on ``/api/chat/stream``, not stored server-side.
    """
    agent_model = os.getenv("AGENT_MODEL")
    return {
        "model": default_model_label(),
        "models": available_models(),
        "model_source": "AGENT_MODEL" if agent_model else "OLLAMA_MODEL (local fallback)",
        "effort": main.DEFAULT_EFFORT,
        "efforts": main.THINKING_EFFORTS,
        # Conversation modes (agent profiles) selectable at conversation creation.
        "modes": main.MODES,
        "arcade_url": os.getenv("ARCADE_URL", DEFAULT_URL),
        "searxng_url": os.getenv("SEARXNG_URL", "http://localhost:8085"),
        "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
        # Semantic fact search: on when an embedding model is configured, else substring matching.
        "embeddings": embeddings_enabled(),
        "embed_model": os.getenv("EMBED_MODEL") or None,
    }


# --------------------------------------------------------------------- conversations


class NewConversation(BaseModel):
    user_id: str = "default"
    title: str | None = None
    # The conversation's agent profile at creation (see backend.main.MODES). Changeable later
    # via PATCH /api/conversations/{id}.
    mode: Literal["regular", "research", "swarm"] = "regular"


class UpdateConversation(BaseModel):
    user_id: str = "default"
    # The agent profile to switch this conversation to (see backend.main.MODES).
    mode: Literal["regular", "research", "swarm"]


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
        await repo.create_conversation(
            db, body.user_id, conversation_id, title=body.title, mode=body.mode
        )
    return {
        "conversation_id": conversation_id,
        "title": body.title,
        "mode": body.mode,
    }


@app.patch("/api/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str, body: UpdateConversation
) -> dict[str, Any]:
    """Switch a conversation's agent mode mid-thread; the change persists for later turns."""
    async with _client_for(body.user_id, ensure=True) as db:
        await repo.set_conversation_mode(db, conversation_id, body.mode)
    return {"conversation_id": conversation_id, "mode": body.mode}


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


# ------------------------------------------------------------------------- documents


class DocumentEdit(BaseModel):
    """A user edit from the Documents pane. Only the provided fields change."""

    user_id: str = "default"
    title: str | None = None
    content: str | None = None


@app.get("/api/conversations/{conversation_id}/documents")
async def get_documents(conversation_id: str, user_id: str = "default") -> list[dict[str, Any]]:
    """List a conversation's documents (metadata only — fetch a body via /api/documents/{id}).

    Pure read, tolerant like the other list endpoints: a missing database/type (no documents
    written yet) yields an empty list rather than a 500.
    """
    try:
        async with _client_for(user_id) as db:
            return await repo.list_documents(db, user_id, conversation_id=conversation_id)
    except Exception:  # noqa: BLE001
        logger.warning("list_documents failed", exc_info=True)
        return []


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str, user_id: str = "default") -> dict[str, Any]:
    """Return one document with its full content (404 if it doesn't exist for this user)."""
    try:
        async with _client_for(user_id) as db:
            row = await repo.get_document(db, user_id, document_id)
    except Exception:  # noqa: BLE001
        logger.warning("get_document failed", exc_info=True)
        row = None
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    return row


@app.put("/api/documents/{document_id}")
async def edit_document(document_id: str, body: DocumentEdit) -> dict[str, Any]:
    """Apply a user edit (title and/or content) to a document and return the updated record.

    This is what makes text documents editable in the UI. A write path, so the database/schema
    are ensured first; 404 when the document doesn't belong to this user.
    """
    if body.title is None and body.content is None:
        raise HTTPException(status_code=400, detail="nothing to update")
    async with _client_for(body.user_id, ensure=True) as db:
        updated = await repo.update_document(
            db, body.user_id, document_id, title=body.title, content=body.content
        )
        if not updated:
            raise HTTPException(status_code=404, detail="document not found")
        row = await repo.get_document(db, body.user_id, document_id)
    return row or {"document_id": document_id}


@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str, user_id: str = "default") -> dict[str, Any]:
    """Delete a document (404 if it doesn't exist for this user)."""
    async with _client_for(user_id, ensure=True) as db:
        deleted = await repo.delete_document(db, user_id, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="document not found")
    return {"deleted": document_id}


# ---------------------------------------------------------------------------- graph


@app.get("/api/graph")
async def get_graph(user_id: str = "default", limit: int = 100) -> dict[str, Any]:
    """Return the user's agent-built knowledge graph as ``{nodes, edges}`` for visualization.

    Pure read on the pooled client (no schema ensure). Tolerant: a missing database/type (nothing
    built yet) or any error yields an empty graph rather than a 500, so the UI pane never breaks.
    """
    try:
        async with _client_for(user_id) as db:
            return await repo.get_user_graph(db, user_id, limit=limit)
    except Exception:  # noqa: BLE001 — the graph pane must never break the page.
        logger.warning("get_graph failed", exc_info=True)
        return {"nodes": [], "edges": []}


# ----------------------------------------------------------------------------- chat


class ChatRequest(BaseModel):
    user_id: str = "default"
    conversation_id: str
    prompt: str
    # Optional per-request model override (a label from /api/config "models"). When omitted, the
    # agent uses the env-configured default. Selection lives in the browser, not on the server.
    model: str | None = None
    # Optional per-request thinking-effort override (a value from /api/config "efforts"). When
    # omitted/unknown, the agent uses DEFAULT_EFFORT. Also browser-side, not stored on the server.
    effort: str | None = None


def _sse(event: dict[str, Any]) -> str:
    # default=str is a safety net: stream_run._jsonable already coerces tool payloads, but a
    # stray non-JSON value must degrade to its repr, never kill the stream mid-turn.
    return f"data: {json.dumps(event, default=str)}\n\n"


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
                body.prompt,
                user_id=body.user_id,
                conversation_id=body.conversation_id,
                model=body.model,
                effort=body.effort,
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

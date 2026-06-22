"""HTTP/SSE API over the conversation-memory agent.

A thin wrapper that exposes the existing CLI machinery to a web frontend. It adds **no** new
database or agent logic: every handler calls into :mod:`backend.db.repository` and
:func:`backend.main.stream_run`, opening a short-lived :class:`ArcadeClient` pointed at the
caller's per-user database (mirroring :func:`backend.main.run`).

Run it with::

    uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend import kb_compiler, main, marketplace, summarization
from backend.db import repository as repo
from backend.schemas.swarm_schemas import (
    TOOL_GROUPS,
    CreateAgentArgs,
    _valid_recipients,
    _valid_skills,
    _valid_tools,
)
from backend.skills.subagent import (
    SWARM_MAX_DEPTH,
    SWARM_MAX_DEPTH_RANGE,
    SWARM_MAX_PARALLEL,
    SWARM_MAX_PARALLEL_RANGE,
)

# kebab-case skill name (lowercase letters/digits/hyphens), for user-authored skills.
_SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
from backend.db.arcade_db import (
    DEFAULT_URL,
    ArcadeClient,
    database_name_for_user,
)
from backend.embeddings import Embedder, embeddings_enabled
from backend.model_selection import available_models, context_window_for, default_model_label
from backend.token_count import count_tokens

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
        # The tool bundles a swarm specialist can be granted (name -> description); backs the
        # roster editor's tool checkboxes.
        "tool_groups": TOOL_GROUPS,
        # The fixed base system prompt; a conversation's custom prompt is appended to it. Shown
        # read-only in the UI so the user knows what their custom instructions add to.
        "base_system_prompt": main.BASE_SYSTEM_PROMPT,
        # Swarm bounds: the env defaults + the allowed override ranges. The Configuration card
        # renders these (swarm mode only) and a conversation may override within range.
        "swarm": {
            "max_parallel": SWARM_MAX_PARALLEL,
            "max_depth": SWARM_MAX_DEPTH,
            "max_parallel_range": list(SWARM_MAX_PARALLEL_RANGE),
            "max_depth_range": list(SWARM_MAX_DEPTH_RANGE),
        },
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
    # Optional owning project: the chat is created inside this project (inherits its system prompt
    # + reference documents, shows under its sidebar group). Omit for an ungrouped chat.
    project_id: str | None = None


class UpdateConversation(BaseModel):
    user_id: str = "default"
    # Partial update: only the fields actually sent are applied (tracked via model_fields_set, so
    # system_prompt="" can clear the prompt while an omitted field is left untouched).
    mode: Literal["regular", "research", "swarm"] | None = None
    # The conversation's custom system prompt, appended to the base prompt at run time.
    system_prompt: str | None = None
    # Per-conversation swarm bounds (swarm mode), validated to the ranges in /api/config.
    swarm_max_parallel: int | None = Field(
        None, ge=SWARM_MAX_PARALLEL_RANGE[0], le=SWARM_MAX_PARALLEL_RANGE[1]
    )
    swarm_max_depth: int | None = Field(
        None, ge=SWARM_MAX_DEPTH_RANGE[0], le=SWARM_MAX_DEPTH_RANGE[1]
    )
    # Marketplace skills enabled for this conversation (by skill name). Empty list clears them.
    enabled_skills: list[str] | None = None
    # Move the conversation into a project (id) or out of one (null = ungrouped).
    project_id: str | None = None
    # Lifecycle flags: archive hides it from the default list; pin floats it to the top.
    archived: bool | None = None
    pinned: bool | None = None


@app.get("/api/conversations")
async def list_conversations(
    user_id: str = "default", include_archived: bool = False
) -> list[dict[str, Any]]:
    # Pure read, no schema ensure. Tolerant: a brand-new user whose database doesn't exist yet
    # simply has no conversations, so a query error maps to an empty list rather than a 500.
    # Archived conversations are omitted unless include_archived is set (the "Show archived" toggle).
    try:
        async with _client_for(user_id) as db:
            return await repo.list_conversations(
                db, user_id, include_archived=include_archived
            )
    except Exception:  # noqa: BLE001
        logger.warning("list_conversations failed", exc_info=True)
        return []


@app.post("/api/conversations")
async def create_conversation(body: NewConversation) -> dict[str, Any]:
    conversation_id = uuid.uuid4().hex
    # The one common write path: ensure the database/schema exist before the first insert.
    async with _client_for(body.user_id, ensure=True) as db:
        await repo.create_conversation(
            db, body.user_id, conversation_id, title=body.title, mode=body.mode,
            project_id=body.project_id,
        )
    return {
        "conversation_id": conversation_id,
        "title": body.title,
        "mode": body.mode,
        "project_id": body.project_id,
    }


@app.patch("/api/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str, body: UpdateConversation
) -> dict[str, Any]:
    """Update a conversation's mode, custom prompt, swarm bounds, project, or lifecycle flags.

    All changes persist and take effect on the next turn. Only the fields explicitly sent are
    applied, so the client can change one without disturbing the others.
    """
    fields = body.model_fields_set
    updated: dict[str, Any] = {"conversation_id": conversation_id}
    async with _client_for(body.user_id, ensure=True) as db:
        if "mode" in fields and body.mode is not None:
            await repo.set_conversation_mode(db, conversation_id, body.mode)
            updated["mode"] = body.mode
        if "system_prompt" in fields:
            await repo.set_conversation_system_prompt(db, conversation_id, body.system_prompt or "")
            updated["system_prompt"] = body.system_prompt or ""
        if "swarm_max_parallel" in fields or "swarm_max_depth" in fields:
            await repo.set_conversation_swarm_settings(
                db,
                conversation_id,
                max_parallel=body.swarm_max_parallel,
                max_depth=body.swarm_max_depth,
            )
            if "swarm_max_parallel" in fields:
                updated["swarm_max_parallel"] = body.swarm_max_parallel
            if "swarm_max_depth" in fields:
                updated["swarm_max_depth"] = body.swarm_max_depth
        if "enabled_skills" in fields:
            await repo.set_conversation_enabled_skills(
                db, conversation_id, body.enabled_skills or []
            )
            updated["enabled_skills"] = body.enabled_skills or []
        if "project_id" in fields:
            await repo.set_conversation_project_id(db, conversation_id, body.project_id)
            updated["project_id"] = body.project_id
        if "archived" in fields and body.archived is not None:
            await repo.set_conversation_archived(db, conversation_id, body.archived)
            updated["archived"] = body.archived
        if "pinned" in fields and body.pinned is not None:
            await repo.set_conversation_pinned(db, conversation_id, body.pinned)
            updated["pinned"] = body.pinned
    return updated


@app.delete("/api/conversations/{conversation_id}")
async def remove_conversation(
    conversation_id: str, user_id: str = "default"
) -> dict[str, Any]:
    """Permanently delete a conversation and all its messages/documents (404 if not the caller's)."""
    async with _client_for(user_id, ensure=True) as db:
        # delete_conversation is tolerant of an already-gone conversation; report the id regardless.
        await repo.delete_conversation(db, user_id, conversation_id)
    return {"deleted": conversation_id}


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


@app.get("/api/conversations/{conversation_id}/context")
async def get_context_usage(
    conversation_id: str,
    user_id: str = "default",
    model: str = "",
    mode: str = "",
) -> dict[str, Any]:
    """Estimate how much of the model's context window this conversation consumes, by component.

    Breaks the window into **system prompt** (base + the conversation's custom prompt — the per-turn
    dynamic facts/date block is variable and excluded), **tool definitions** (the exact tools the
    conversation's ``mode`` exposes), and **messages** (the faithful run history reloaded each turn).
    ``model`` is the UI-selected label (defaults to the server default); ``mode`` is read from the DB
    when not supplied. Token counts use :mod:`backend.token_count` (precise tiktoken, heuristic
    fallback). Tolerant: any failure returns zeros so the config pane never breaks.
    """
    label = model or default_model_label()
    window = context_window_for(label)
    try:
        async with _client_for(user_id) as db:
            resolved_mode = mode or await repo.get_conversation_mode(db, conversation_id)
            custom_prompt = await repo.get_conversation_system_prompt(db, conversation_id)
            history_rows = await repo.get_run_history(db, conversation_id)

        sys_tokens, counter = count_tokens(main.compose_instructions(custom_prompt), label)
        tool_tokens, _ = count_tokens(main.tool_definitions_json(resolved_mode), label)
        msg_tokens, _ = count_tokens(main.message_history_text(history_rows), label)
        used = sys_tokens + tool_tokens + msg_tokens
        return {
            "model": label,
            "context_window": window,
            "counter": counter,
            "components": {
                "system_prompt": sys_tokens,
                "tools": tool_tokens,
                "messages": msg_tokens,
            },
            "used": used,
            "free": max(0, window - used),
            "percent": round(used / window * 100, 1) if window else 0.0,
        }
    except Exception:  # noqa: BLE001 — the context meter must never break the page.
        logger.warning("context usage failed", exc_info=True)
        return {
            "model": label,
            "context_window": window,
            "counter": "unavailable",
            "components": {"system_prompt": 0, "tools": 0, "messages": 0},
            "used": 0,
            "free": window,
            "percent": 0.0,
        }


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


class DocumentGlobal(BaseModel):
    """A toggle from the project documents UI: whether a document is global (survives cascade)."""

    user_id: str = "default"
    is_global: bool


@app.post("/api/documents/{document_id}/global")
async def set_document_global(document_id: str, body: DocumentGlobal) -> dict[str, Any]:
    """Mark a document global (available everywhere, exempt from project cascade-delete) or not.

    404 when the document doesn't belong to this user.
    """
    async with _client_for(body.user_id, ensure=True) as db:
        updated = await repo.set_document_global(
            db, body.user_id, document_id, body.is_global
        )
    if not updated:
        raise HTTPException(status_code=404, detail="document not found")
    return {"document_id": document_id, "is_global": body.is_global}


# -------------------------------------------------------------------------- projects


class NewProject(BaseModel):
    user_id: str = "default"
    title: str | None = None
    system_prompt: str = ""


class UpdateProject(BaseModel):
    user_id: str = "default"
    # Partial update (model_fields_set): only the sent fields change. title="" / system_prompt=""
    # clear the respective field.
    title: str | None = None
    system_prompt: str | None = None


@app.get("/api/projects")
async def list_projects(user_id: str = "default") -> list[dict[str, Any]]:
    """List the user's projects (metadata) for the sidebar groups. Tolerant: errors → empty list."""
    try:
        async with _client_for(user_id) as db:
            return await repo.list_projects(db, user_id)
    except Exception:  # noqa: BLE001 — the sidebar must never break the page.
        logger.warning("list_projects failed", exc_info=True)
        return []


@app.post("/api/projects")
async def create_project(body: NewProject) -> dict[str, Any]:
    project_id = uuid.uuid4().hex
    async with _client_for(body.user_id, ensure=True) as db:
        await repo.create_project(
            db, body.user_id, project_id, title=body.title, system_prompt=body.system_prompt
        )
    return {
        "project_id": project_id,
        "title": body.title,
        "system_prompt": body.system_prompt,
    }


@app.patch("/api/projects/{project_id}")
async def update_project(project_id: str, body: UpdateProject) -> dict[str, Any]:
    """Update a project's title and/or system prompt (404 if not the caller's)."""
    fields = body.model_fields_set
    title = body.title if "title" in fields else None
    system_prompt = body.system_prompt if "system_prompt" in fields else None
    async with _client_for(body.user_id, ensure=True) as db:
        updated = await repo.update_project(
            db, body.user_id, project_id, title=title, system_prompt=system_prompt
        )
        if not updated:
            raise HTTPException(status_code=404, detail="project not found")
        row = await repo.get_project(db, body.user_id, project_id)
    return row or {"project_id": project_id}


@app.delete("/api/projects/{project_id}")
async def remove_project(project_id: str, user_id: str = "default") -> dict[str, Any]:
    """Cascade-delete a project: its conversations + non-global documents. Globals are spared.

    Returns counts of what was deleted (for the UI's confirmation toast).
    """
    async with _client_for(user_id, ensure=True) as db:
        counts = await repo.delete_project(db, user_id, project_id)
    return {"deleted": project_id, **counts}


@app.get("/api/projects/{project_id}/documents")
async def get_project_documents(
    project_id: str, user_id: str = "default"
) -> list[dict[str, Any]]:
    """List a project's reference documents plus the user's global ones (metadata only). Tolerant."""
    try:
        async with _client_for(user_id) as db:
            return await repo.list_documents(
                db, user_id, project_id=project_id, include_global=True
            )
    except Exception:  # noqa: BLE001 — the project pane must never break the page.
        logger.warning("list project documents failed", exc_info=True)
        return []


class ProjectUpload(BaseModel):
    """Upload one reference file into a project's document set (base64 bytes + name + mime)."""

    user_id: str = "default"
    filename: str = ""
    mime_type: str
    data: str  # base64-encoded file bytes (no "data:" URL prefix)


@app.post("/api/projects/{project_id}/documents")
async def upload_project_document(
    project_id: str, body: ProjectUpload
) -> dict[str, Any]:
    """Persist an uploaded file as a project reference document (decoded + embedded like a chat upload).

    Mirrors stream_run's upload-persist: binary mimes stay base64, text is decoded and (when
    embeddings are configured) embedded so search_project_documents can rank it.
    """
    try:
        size = len(base64.b64decode(body.data, validate=True))
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="attachment is not valid base64")
    if size > _MAX_ATTACHMENT_BYTES:
        raise HTTPException(status_code=400, detail="attachment exceeds the size limit")

    mime = body.mime_type or "application/octet-stream"
    name = body.filename or "upload"
    if main._is_binary_attachment(mime):
        body_text, encoding = body.data, "base64"
    else:
        body_text = base64.b64decode(body.data).decode("utf-8", errors="replace")
        encoding = "text"
    embedding = None
    if encoding == "text" and embeddings_enabled():
        try:
            async with Embedder.from_env() as embedder:
                embedding = await embedder.embed(body_text)
        except Exception:  # noqa: BLE001 — embedding is best-effort; degrade to LIKE search.
            logger.warning("embedding project upload %r failed", name, exc_info=True)
    async with _client_for(body.user_id, ensure=True) as db:
        document_id = await repo.create_document(
            db,
            body.user_id,
            title=name,
            content=body_text,
            mime_type=mime,
            encoding=encoding,
            project_id=project_id,
            embedding=embedding,
            vertex_type="KbSource",
        )
    # Auto-compile this project's knowledge base (debounced + best-effort; coalesces bulk uploads).
    kb_compiler.schedule_kb_compile(body.user_id, project_id, _client_for)
    return {
        "document_id": document_id,
        "project_id": project_id,
        "title": name,
        "mime_type": mime,
        "encoding": encoding,
    }


@app.get("/api/projects/{project_id}/kb")
async def get_project_kb(project_id: str, user_id: str = "default") -> dict[str, Any]:
    """Return the project's knowledge-base status + its compiled pages (metadata only). Tolerant."""
    try:
        async with _client_for(user_id) as db:
            status = await repo.get_project_kb_status(db, user_id, project_id)
            pages = await repo.list_documents(
                db, user_id, project_id=project_id, from_type="KbPage", limit=200
            )
        return {**status, "pages": pages}
    except Exception:  # noqa: BLE001 — the project pane must never break the page.
        logger.warning("get project kb failed", exc_info=True)
        return {"status": "idle", "compiled_at": None, "pages": []}


@app.post("/api/projects/{project_id}/kb/rebuild")
async def rebuild_project_kb(project_id: str, user_id: str = "default") -> dict[str, Any]:
    """Kick a full knowledge-base rebuild for a project in the background. Returns the new status."""
    kb_compiler.schedule_kb_compile(user_id, project_id, _client_for, full=True, delay=0)
    return {"status": "compiling" if kb_compiler.KB_COMPILE_ENABLED else "idle"}


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


# ----------------------------------------------------------------------------- facts


class FactImportance(BaseModel):
    """A toggle from the Facts pane: whether a fact is included in the agent's context."""

    user_id: str = "default"
    important: bool


@app.get("/api/user/profile")
async def get_user_profile(user_id: str = "default") -> dict[str, Any]:
    """Return the user's durable, curator-maintained profile (a fast DB read).

    The profile is rewritten at write time by the background memory curator (every few turns; see
    :func:`backend.memory_curator.maybe_curate_memory`), so this endpoint never runs an LLM. Tolerant
    like the summary/facts reads: any error (or no profile yet) yields an empty profile, never a 500.
    """
    try:
        async with _client_for(user_id) as db:
            return await repo.get_user_profile(db, user_id)
    except Exception:  # noqa: BLE001 — the profile card must never break the page.
        logger.warning("user profile read failed", exc_info=True)
        return {"profile": "", "profile_updated_at": None}


@app.get("/api/facts")
async def list_facts(user_id: str = "default", limit: int = 200) -> list[dict[str, Any]]:
    """List the user's stored facts (newest first) for the Facts tab.

    Pure read, tolerant like the graph/documents lists: a missing database/type (no facts yet) or
    any error yields an empty list rather than a 500, so the pane never breaks.
    """
    try:
        async with _client_for(user_id) as db:
            return await repo.list_facts(db, user_id, limit=limit)
    except Exception:  # noqa: BLE001 — the facts pane must never break the page.
        logger.warning("list_facts failed", exc_info=True)
        return []


@app.patch("/api/facts/{fact_id}")
async def update_fact_importance(fact_id: str, body: FactImportance) -> dict[str, Any]:
    """Toggle whether a fact is included in the agent's context (404 if not the caller's)."""
    async with _client_for(body.user_id, ensure=True) as db:
        updated = await repo.set_fact_importance(db, body.user_id, fact_id, body.important)
    if not updated:
        raise HTTPException(status_code=404, detail="fact not found")
    return {"fact_id": fact_id, "important": body.important}


# ---------------------------------------------------------------------------- skills


class SyncSkills(BaseModel):
    """A request to sync marketplace skills into the user's database."""

    user_id: str = "default"
    # Specific skill names to sync; omit/null to sync the whole catalog.
    names: list[str] | None = None


@app.get("/api/skills")
async def list_skills(user_id: str = "default") -> list[dict[str, Any]]:
    """List the marketplace skills this user has synced (metadata only — no body/files).

    Backs the Configuration card's skill picker. Pure read, tolerant like the other list endpoints:
    a missing database/type (nothing synced yet) yields an empty list rather than a 500.
    """
    try:
        async with _client_for(user_id) as db:
            return await repo.list_skills(db, user_id)
    except Exception:  # noqa: BLE001 — the skills picker must never break the page.
        logger.warning("list_skills failed", exc_info=True)
        return []


@app.get("/api/skills/catalog")
async def skills_catalog(user_id: str = "default") -> list[dict[str, Any]]:
    """Browse the live Anthropic marketplace catalog (name + description + `installed` flag).

    Backs the Skill Marketplace dialog. The catalog (name/description) comes from GitHub
    (cached in-process); the `installed` flag is merged fresh from the user's synced library, so it
    flips to true immediately after an install. Tolerant: any failure (GitHub unreachable, no DB
    yet) yields an empty list rather than a 500.
    """
    try:
        async with _client_for(user_id) as db:
            installed = {s.get("name") for s in await repo.list_skills(db, user_id)}
        items = await marketplace.catalog()
        return [{**item, "installed": item["name"] in installed} for item in items]
    except Exception:  # noqa: BLE001 — the marketplace dialog must never break the page.
        logger.warning("skills catalog failed", exc_info=True)
        return []


@app.delete("/api/skills/{name}")
async def uninstall_skill(name: str, user_id: str = "default") -> dict[str, Any]:
    """Remove a skill from the user's library (404 if they don't have it). Does not touch chats.

    A conversation that still lists the skill in its `enabled_skills` simply won't find it on the
    next turn (load_skill returns "re-sync" guidance) — the same tolerant contract as a stale id.
    """
    async with _client_for(user_id, ensure=True) as db:
        deleted = await repo.delete_skill(db, user_id, name)
    if not deleted:
        raise HTTPException(status_code=404, detail="skill not found")
    return {"deleted": name}


@app.post("/api/skills/sync")
async def sync_skills(body: SyncSkills) -> dict[str, Any]:
    """Sync skills from the Anthropic marketplace into the user's database.

    A write path (it upserts Skill vertices), so the database/schema are ensured first. Tolerant:
    the sync collects per-skill failures into the returned summary rather than raising. A total
    failure (e.g. GitHub unreachable) returns the same shape with an error entry, not a 500.
    """
    try:
        async with _client_for(body.user_id, ensure=True) as db:
            return await marketplace.sync(db, body.user_id, names=body.names)
    except Exception as exc:  # noqa: BLE001 — never 500; report as a summary error.
        logger.warning("skills sync failed", exc_info=True)
        return {"synced": [], "errors": [{"name": "*", "error": str(exc)}], "source": ""}


class CreateSkill(BaseModel):
    """A user-authored skill (name + description + instructions body). Upsert-by-name = edit."""

    user_id: str = "default"
    name: str
    description: str = ""
    body: str = ""
    # Optional bundled files (relpath -> {content, encoding}); usually empty for hand-authored skills.
    files: dict[str, dict[str, str]] | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        low = v.strip().lower()
        if not _SKILL_NAME_RE.match(low):
            raise ValueError(
                "name must be kebab-case (lowercase letters, digits, hyphens), e.g. 'my-skill'."
            )
        return low


@app.post("/api/skills")
async def create_user_skill(body: CreateSkill) -> dict[str, Any]:
    """Create (or edit, by name) a user-authored skill in the library. `source` is `"user"`.

    Distinct from POST /api/skills/sync (which pulls from GitHub). Upsert-by-name means re-posting
    the same name edits it. The skill then participates in everything a synced skill does
    (auto-enable account-wide, swarm assignment, the use-notification).
    """
    async with _client_for(body.user_id, ensure=True) as db:
        await repo.upsert_skill(
            db,
            body.user_id,
            name=body.name,
            description=body.description,
            body=body.body,
            files=body.files or {},
            source="user",
        )
    return {"name": body.name, "description": body.description, "source": "user"}


@app.get("/api/skills/{name}/content")
async def get_skill_content(name: str, user_id: str = "default") -> dict[str, Any]:
    """Return a skill's full record (body + files) so the editor can load it (404 if missing)."""
    try:
        async with _client_for(user_id) as db:
            row = await repo.get_skill(db, user_id, name)
    except Exception:  # noqa: BLE001
        logger.warning("get_skill_content failed", exc_info=True)
        row = None
    if row is None:
        raise HTTPException(status_code=404, detail="skill not found")
    return row


# ----------------------------------------------------------------------------- agents


class NewAgent(CreateAgentArgs):
    """Create a swarm roster agent via the REST API (CreateAgentArgs + the owning user)."""

    user_id: str = "default"


class EditAgent(BaseModel):
    """Partial update of a roster agent (only the fields sent change). Name is immutable."""

    user_id: str = "default"
    role: str | None = None
    instructions: str | None = None
    tools: list[str] | None = None
    skills: list[str] | None = None
    recipients: list[str] | None = None

    @field_validator("tools")
    @classmethod
    def _check_tools(cls, v: list[str] | None) -> list[str] | None:
        return _valid_tools(v) if v is not None else None

    @field_validator("skills")
    @classmethod
    def _check_skills(cls, v: list[str] | None) -> list[str] | None:
        return _valid_skills(v) if v is not None else None

    @field_validator("recipients")
    @classmethod
    def _check_recipients(cls, v: list[str] | None) -> list[str] | None:
        return _valid_recipients(v) if v is not None else None


@app.get("/api/agents")
async def list_agents(user_id: str = "default") -> list[dict[str, Any]]:
    """List the user's swarm roster (full specs incl. tools/skills/recipients). Tolerant → []."""
    try:
        async with _client_for(user_id) as db:
            return await repo.list_agent_specs(db, user_id)
    except Exception:  # noqa: BLE001 — the roster editor must never break the page.
        logger.warning("list_agents failed", exc_info=True)
        return []


@app.post("/api/agents")
async def create_agent(body: NewAgent) -> dict[str, Any]:
    """Create a roster agent (409 if the name is taken — edit it with PATCH instead)."""
    async with _client_for(body.user_id, ensure=True) as db:
        existing = await repo.get_agent_spec(db, body.user_id, body.name)
        if existing is not None:
            raise HTTPException(status_code=409, detail="an agent with that name already exists")
        agent_id = await repo.create_agent_spec(
            db,
            body.user_id,
            name=body.name,
            role=body.role,
            instructions=body.instructions,
            tools=body.tools,
            recipients=body.recipients,
            skills=body.skills,
        )
        row = await repo.get_agent_spec(db, body.user_id, agent_id)
    return row or {"agent_id": agent_id, "name": body.name}


@app.patch("/api/agents/{agent_id}")
async def update_agent(agent_id: str, body: EditAgent) -> dict[str, Any]:
    """Update a roster agent's role/instructions/tools/skills/recipients (404 if not the caller's)."""
    fields = body.model_fields_set
    async with _client_for(body.user_id, ensure=True) as db:
        updated = await repo.update_agent_spec(
            db,
            body.user_id,
            agent_id,
            role=body.role if "role" in fields else None,
            instructions=body.instructions if "instructions" in fields else None,
            tools=body.tools if "tools" in fields else None,
            recipients=body.recipients if "recipients" in fields else None,
            skills=body.skills if "skills" in fields else None,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="agent not found")
        row = await repo.get_agent_spec(db, body.user_id, agent_id)
    return row or {"agent_id": agent_id}


@app.delete("/api/agents/{agent_id}")
async def remove_agent(agent_id: str, user_id: str = "default") -> dict[str, Any]:
    """Delete a roster agent (404 if not the caller's)."""
    async with _client_for(user_id, ensure=True) as db:
        deleted = await repo.delete_agent_spec(db, user_id, agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="agent not found")
    return {"deleted": agent_id}


# ----------------------------------------------------------------------------- chat


# Upload limits. Enforced here (authoritative) as well as client-side (fast UX feedback).
_MAX_ATTACHMENTS = 5
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 20 MB decoded per file


class Attachment(BaseModel):
    """One uploaded file: base64-encoded bytes plus its name and mime type."""

    filename: str = ""
    mime_type: str
    data: str  # base64-encoded file bytes (no "data:" URL prefix)


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
    # Files the user attached to this message (images/PDFs the agent reads as multimodal content,
    # text files inlined). Empty for a plain text turn. See backend.main.build_user_content.
    attachments: list[Attachment] = Field(default_factory=list)


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

    # Validate uploads up front (before the stream opens) so an oversized/garbled file is a clean
    # 400, not a mid-stream failure.
    if len(body.attachments) > _MAX_ATTACHMENTS:
        raise HTTPException(
            status_code=400, detail=f"too many attachments (max {_MAX_ATTACHMENTS})"
        )
    for att in body.attachments:
        try:
            size = len(base64.b64decode(att.data, validate=True))
        except (binascii.Error, ValueError):
            raise HTTPException(
                status_code=400, detail=f"attachment {att.filename!r} is not valid base64"
            )
        if size > _MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=400, detail=f"attachment {att.filename!r} exceeds the size limit"
            )

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in main.stream_run(
                body.prompt,
                user_id=body.user_id,
                conversation_id=body.conversation_id,
                model=body.model,
                effort=body.effort,
                attachments=[a.model_dump() for a in body.attachments],
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

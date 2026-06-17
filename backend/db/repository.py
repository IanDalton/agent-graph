"""Persistence layer for conversations, messages, facts and logs.

This is the single source of truth for all database access: both the agent
tools (in ``backend.skills.graph_capability``) and the automatic persistence
hooks call into these functions. All statements are parameterized ArcadeDB SQL
and scoped by ``user_id`` so each user's memory stays isolated.

Conversation history is stored in two parallel records: ``Message`` vertices keep
human-readable role/content text for substring search, while ``RunMessages``
vertices keep each run's serialized Pydantic AI messages (tool calls + returns
included) so a later turn can be replayed *faithfully* — see ``append_run_messages``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.db.arcade_db import ArcadeClient

logger = logging.getLogger("agent_graph.repository")

# Server-generated record ids look like ``#12:0``; only such values are interpolated into DDL.
_RID_RE = re.compile(r"^#\d+:\d+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def _affected(result: list[dict[str, Any]]) -> int:
    """Rows changed by an UPDATE/DELETE. ArcadeDB returns ``[{"count": N}]`` for these."""
    if result and isinstance(result[0], dict) and "count" in result[0]:
        return int(result[0]["count"])
    return len(result)


async def create_conversation(
    db: ArcadeClient,
    user_id: str,
    conversation_id: str,
    title: str | None = None,
    mode: str = "regular",
    project_id: str | None = None,
) -> None:
    """Ensure the User and Conversation vertices (and their link) exist. Idempotent.

    ``mode`` selects the agent profile for this conversation ('regular'/'research'/'swarm' — see
    ``backend.main.MODES``). It is fixed at creation: the persistence hook's idempotent call (which
    always passes the default) cannot overwrite a mode the API set, because an existing
    conversation returns early. ``project_id`` (when given) stamps the owning project at creation
    (the conversation shows under that project's group and inherits its system prompt + documents).
    """
    existing = await db.query(
        "SELECT count(*) AS n FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    if existing and existing[0].get("n", 0) > 0:
        return

    # Upsert the user vertex, then create the conversation and link them.
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    set_clause = (
        "conversation_id = :cid, user_id = :uid, title = :title, mode = :mode, started_at = :ts"
    )
    params: dict[str, Any] = {
        "cid": conversation_id, "uid": user_id, "title": title, "mode": mode, "ts": _now(),
    }
    if project_id is not None:
        set_clause += ", project_id = :pid"
        params["pid"] = project_id
    await db.command("CREATE VERTEX Conversation SET " + set_clause, params)
    await db.command(
        "CREATE EDGE HAS_CONVERSATION "
        "FROM (SELECT FROM User WHERE user_id = :uid) "
        "TO (SELECT FROM Conversation WHERE conversation_id = :cid)",
        {"uid": user_id, "cid": conversation_id},
    )


async def append_message(
    db: ArcadeClient,
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    embedding: list[float] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> None:
    """Persist a single message and link it to its conversation.

    When ``embedding`` is given (semantic search is enabled), it is stored on the Message's vector
    property so :func:`search_messages` can rank by similarity. Omitted ⇒ the message is still
    searchable via substring matching.

    ``attachments`` is the metadata of files uploaded with this (user) message —
    ``[{document_id, filename, mime_type}]`` — stored as a JSON string so a reloaded bubble can
    re-open them (the file bodies live in their own Document vertices). Omitted for a text-only
    message.
    """
    message_id = _new_id()
    set_clause = (
        "message_id = :mid, conversation_id = :cid, "
        "user_id = :uid, role = :role, content = :content, created_at = :ts"
    )
    params: dict[str, Any] = {
        "mid": message_id,
        "cid": conversation_id,
        "uid": user_id,
        "role": role,
        "content": content,
        "ts": _now(),
    }
    if embedding is not None:
        set_clause += ", embedding = :emb"
        params["emb"] = embedding
    if attachments:
        set_clause += ", attachments = :att"
        params["att"] = json.dumps(attachments)
    await db.command(f"CREATE VERTEX Message SET {set_clause}", params)
    await db.command(
        "CREATE EDGE HAS_MESSAGE "
        "FROM (SELECT FROM Conversation WHERE conversation_id = :cid) "
        "TO (SELECT FROM Message WHERE message_id = :mid)",
        {"cid": conversation_id, "mid": message_id},
    )


async def get_recent_messages(
    db: ArcadeClient,
    conversation_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return the most recent messages of a conversation in chronological order.

    Each row carries ``role``/``content``/``created_at`` and, for messages saved with uploaded
    files, an ``attachments`` list (parsed from its stored JSON; absent/empty otherwise).
    """
    rows = await db.query(
        "SELECT role, content, created_at, attachments FROM Message "
        "WHERE conversation_id = :cid ORDER BY created_at DESC LIMIT :limit",
        {"cid": conversation_id, "limit": limit},
    )
    for row in rows:
        att = row.get("attachments")
        if isinstance(att, str) and att:
            try:
                row["attachments"] = json.loads(att)
            except json.JSONDecodeError:
                row["attachments"] = []
    return list(reversed(rows))


async def count_messages(db: ArcadeClient, conversation_id: str) -> int:
    """Number of messages in a conversation (fast indexed count)."""
    rows = await db.query(
        "SELECT count(*) AS n FROM Message WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    return int(rows[0].get("n", 0)) if rows else 0


async def get_conversation_summary(db: ArcadeClient, conversation_id: str) -> dict[str, Any]:
    """Return the cached summary + the message count it was generated at.

    Defaults (``""``/``0``) when the conversation has no summary yet, so callers can compare the
    current message count against ``summary_message_count`` to decide whether a refresh is due.
    """
    rows = await db.query(
        "SELECT summary, summary_message_count FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    row = rows[0] if rows else {}
    return {
        "summary": row.get("summary") or "",
        "summary_message_count": int(row.get("summary_message_count") or 0),
    }


async def get_conversation_title(db: ArcadeClient, conversation_id: str) -> str:
    """Return the current conversation title, or an empty string when unset."""
    rows = await db.query(
        "SELECT title FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    return str((rows[0].get("title") if rows else "") or "")


async def set_conversation_title(db: ArcadeClient, conversation_id: str, title: str) -> None:
    """Store a generated or user-edited conversation title in place."""
    await db.command(
        "UPDATE Conversation SET title = :title WHERE conversation_id = :cid",
        {"title": title, "cid": conversation_id},
    )


async def set_conversation_summary(
    db: ArcadeClient,
    conversation_id: str,
    summary: str,
    message_count: int,
) -> None:
    """Store a freshly generated summary and the message count it reflects on the Conversation."""
    await db.command(
        "UPDATE Conversation SET summary = :s, summary_message_count = :n, "
        "summary_updated_at = :ts WHERE conversation_id = :cid",
        {"s": summary, "n": message_count, "ts": _now(), "cid": conversation_id},
    )


async def get_memory_curation_watermark(db: ArcadeClient, conversation_id: str) -> int:
    """Return the message count at which the background memory curator last ran for this thread.

    The watermark gates curation the same way ``summary_message_count`` gates the summary:
    callers compare it against the current message count to decide whether a fresh pass is due.
    Defaults to ``0`` for conversations that have never been curated.
    """
    rows = await db.query(
        "SELECT memory_curated_message_count FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    row = rows[0] if rows else {}
    return int(row.get("memory_curated_message_count") or 0)


async def set_memory_curation_watermark(
    db: ArcadeClient, conversation_id: str, message_count: int
) -> None:
    """Record the message count reached when the memory curator last ran for this conversation."""
    await db.command(
        "UPDATE Conversation SET memory_curated_message_count = :n, "
        "memory_curated_at = :ts WHERE conversation_id = :cid",
        {"n": message_count, "ts": _now(), "cid": conversation_id},
    )


async def get_user_profile(db: ArcadeClient, user_id: str) -> dict[str, Any]:
    """Return the persistent, curator-maintained profile of the user (durable cross-conversation context).

    Distinct from the per-conversation summary: this is a single rolling synopsis of who the user
    is, rewritten by the background memory curator (see ``backend.memory_curator``) and injected into
    every main-agent turn. Defaults to ``""``/``None`` for a user without a profile yet.
    """
    rows = await db.query(
        "SELECT profile, profile_updated_at FROM User WHERE user_id = :uid",
        {"uid": user_id},
    )
    row = rows[0] if rows else {}
    return {
        "profile": row.get("profile") or "",
        "profile_updated_at": row.get("profile_updated_at") or None,
    }


async def set_user_profile(db: ArcadeClient, user_id: str, profile: str) -> None:
    """Replace the user's curated profile in full (upserting the User vertex first)."""
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    await db.command(
        "UPDATE User SET profile = :p, profile_updated_at = :ts WHERE user_id = :uid",
        {"p": profile, "ts": _now(), "uid": user_id},
    )


async def list_conversations(
    db: ArcadeClient,
    user_id: str,
    limit: int = 50,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    """Return this user's conversations, pinned first then most recently started.

    Backs the UI's left-pane conversation list. Scoped by ``user_id`` (defense-in-depth on
    top of the per-user database). ``mode`` is the stored agent profile; conversations created
    before modes existed have none and report ``"regular"``. ``system_prompt`` is the custom
    per-conversation prompt (``""`` when unset), so the UI's config card has it without a separate
    fetch. ``project_id`` is the owning project (``None`` ⇒ ungrouped); ``pinned``/``archived`` are
    the lifecycle flags (default ``False`` for rows created before they existed). Archived
    conversations are omitted unless ``include_archived`` is set.
    """
    where = "user_id = :uid"
    if not include_archived:
        # Default-false semantics: a missing/NULL archived flag reads as not-archived, so existing
        # rows need no backfill. ArcadeDB has no boolean coercion in WHERE, so test both forms.
        where += " AND (archived IS NULL OR archived = false)"
    rows = await db.query(
        "SELECT conversation_id, title, mode, system_prompt, enabled_skills, "
        "swarm_max_parallel, swarm_max_depth, project_id, pinned, archived, started_at "
        f"FROM Conversation WHERE {where} "
        "ORDER BY pinned DESC, started_at DESC LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )
    for row in rows:
        row["mode"] = row.get("mode") or "regular"
        row["system_prompt"] = row.get("system_prompt") or ""
        row["enabled_skills"] = _parse_skill_names(row.get("enabled_skills"))
        row["pinned"] = bool(row.get("pinned"))
        row["archived"] = bool(row.get("archived"))
        # project_id and swarm overrides are left as-is (None when unset).
    return rows


async def set_conversation_mode(db: ArcadeClient, conversation_id: str, mode: str) -> None:
    """Update a conversation's agent mode ('regular'/'research'/'swarm') in place.

    Lets the user switch agent profile mid-conversation; the change persists, so every subsequent
    turn (which re-reads the stored mode) uses the new profile. Validate ``mode`` at the API edge.
    """
    await db.command(
        "UPDATE Conversation SET mode = :mode WHERE conversation_id = :cid",
        {"mode": mode, "cid": conversation_id},
    )


async def get_conversation_mode(db: ArcadeClient, conversation_id: str) -> str:
    """Return the conversation's agent mode ('regular'/'research'/'swarm').

    Defaults to ``"regular"`` for an unknown conversation or one created before modes existed,
    so callers never have to special-case missing data.
    """
    rows = await db.query(
        "SELECT mode FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    return (rows[0].get("mode") if rows else None) or "regular"


async def set_conversation_system_prompt(
    db: ArcadeClient, conversation_id: str, prompt: str
) -> None:
    """Store the conversation's custom system prompt (appended to the base prompt at run time).

    User-set from the web UI's Configuration card. An empty string clears it. Like the other
    Conversation properties this needs no DDL — ArcadeDB sets the field on write.
    """
    await db.command(
        "UPDATE Conversation SET system_prompt = :sp WHERE conversation_id = :cid",
        {"sp": prompt, "cid": conversation_id},
    )


async def get_conversation_system_prompt(db: ArcadeClient, conversation_id: str) -> str:
    """Return the conversation's custom system prompt, or ``""`` when unset/unknown."""
    rows = await db.query(
        "SELECT system_prompt FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    return str((rows[0].get("system_prompt") if rows else "") or "")


async def get_conversation_project_id(db: ArcadeClient, conversation_id: str) -> str | None:
    """Return the conversation's owning project id, or ``None`` when ungrouped/unknown.

    Read each turn by ``stream_run`` to layer in the project's system prompt + reference documents.
    """
    rows = await db.query(
        "SELECT project_id FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    return (rows[0].get("project_id") if rows else None) or None


async def set_conversation_project_id(
    db: ArcadeClient, conversation_id: str, project_id: str | None
) -> None:
    """Move a conversation into a project, or out of one. ``None`` clears membership (ungrouped)."""
    await db.command(
        "UPDATE Conversation SET project_id = :pid WHERE conversation_id = :cid",
        {"pid": project_id, "cid": conversation_id},
    )


async def set_conversation_archived(
    db: ArcadeClient, conversation_id: str, archived: bool
) -> None:
    """Archive (hide from the default list) or unarchive a conversation. Needs no DDL."""
    await db.command(
        "UPDATE Conversation SET archived = :a WHERE conversation_id = :cid",
        {"a": bool(archived), "cid": conversation_id},
    )


async def set_conversation_pinned(db: ArcadeClient, conversation_id: str, pinned: bool) -> None:
    """Pin a conversation to the top of the list, or unpin it. Needs no DDL."""
    await db.command(
        "UPDATE Conversation SET pinned = :p WHERE conversation_id = :cid",
        {"p": bool(pinned), "cid": conversation_id},
    )


async def delete_conversation(db: ArcadeClient, user_id: str, conversation_id: str) -> None:
    """Permanently delete a conversation and all of its children, if it belongs to ``user_id``.

    Cascades: the conversation's Messages, RunMessages, conversation-scoped Documents and LogEntries
    are deleted first (their HAS_* edges go with them via ``DELETE VERTEX``), then the Conversation
    vertex itself. Project- and global-scoped documents have no ``conversation_id`` and are left
    untouched. Best-effort per child set — a failure deleting one set must not strand the rest, so
    callers wrap this tolerantly.
    """
    for child in ("Message", "RunMessages", "Document", "LogEntry"):
        await db.command(
            f"DELETE VERTEX FROM (SELECT FROM {child} WHERE conversation_id = :cid)",
            {"cid": conversation_id},
        )
    await db.command(
        "DELETE VERTEX FROM (SELECT FROM Conversation "
        "WHERE conversation_id = :cid AND user_id = :uid)",
        {"cid": conversation_id, "uid": user_id},
    )


# --- Projects: containers grouping conversations + reference documents ---------------------------


async def create_project(
    db: ArcadeClient,
    user_id: str,
    project_id: str,
    title: str | None = None,
    system_prompt: str = "",
) -> None:
    """Ensure the User and Project vertices (and their link) exist. Idempotent like
    :func:`create_conversation`: an existing project returns early.
    """
    existing = await db.query(
        "SELECT count(*) AS n FROM Project WHERE project_id = :pid",
        {"pid": project_id},
    )
    if existing and existing[0].get("n", 0) > 0:
        return
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    await db.command(
        "CREATE VERTEX Project SET project_id = :pid, user_id = :uid, "
        "title = :title, system_prompt = :sp, created_at = :ts",
        {"pid": project_id, "uid": user_id, "title": title, "sp": system_prompt, "ts": _now()},
    )
    await db.command(
        "CREATE EDGE HAS_PROJECT "
        "FROM (SELECT FROM User WHERE user_id = :uid) "
        "TO (SELECT FROM Project WHERE project_id = :pid)",
        {"uid": user_id, "pid": project_id},
    )


async def list_projects(
    db: ArcadeClient, user_id: str, limit: int = 50
) -> list[dict[str, Any]]:
    """Return this user's projects, most recently created first (metadata for the sidebar groups).

    Defensively skips any vertex with a NULL/empty ``project_id`` (legacy orphans that can't be
    addressed by the id-based update/delete paths and would otherwise render as undeletable
    "Untitled project" rows).
    """
    rows = await db.query(
        "SELECT project_id, title, system_prompt, created_at FROM Project "
        "WHERE user_id = :uid AND project_id IS NOT NULL ORDER BY created_at DESC LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )
    rows = [r for r in rows if r.get("project_id")]
    for row in rows:
        row["system_prompt"] = row.get("system_prompt") or ""
    return rows


async def get_project(
    db: ArcadeClient, user_id: str, project_id: str
) -> dict[str, Any] | None:
    """Return one project's metadata, or ``None`` if it doesn't exist for this user."""
    rows = await db.query(
        "SELECT project_id, title, system_prompt, created_at FROM Project "
        "WHERE project_id = :pid AND user_id = :uid",
        {"pid": project_id, "uid": user_id},
    )
    if not rows:
        return None
    row = rows[0]
    row["system_prompt"] = row.get("system_prompt") or ""
    return row


async def get_project_system_prompt(db: ArcadeClient, project_id: str) -> str:
    """Return the project's system prompt, or ``""`` when unset/unknown (mirrors the conversation one)."""
    rows = await db.query(
        "SELECT system_prompt FROM Project WHERE project_id = :pid",
        {"pid": project_id},
    )
    return str((rows[0].get("system_prompt") if rows else "") or "")


async def update_project(
    db: ArcadeClient,
    user_id: str,
    project_id: str,
    title: str | None = None,
    system_prompt: str | None = None,
) -> int:
    """Revise a project's title and/or system prompt. Returns rows updated (0 if not this user's)."""
    set_clauses: list[str] = []
    params: dict[str, Any] = {"pid": project_id, "uid": user_id}
    if title is not None:
        set_clauses.append("title = :title")
        params["title"] = title
    if system_prompt is not None:
        set_clauses.append("system_prompt = :sp")
        params["sp"] = system_prompt
    if not set_clauses:
        return 0
    result = await db.command(
        "UPDATE Project SET " + ", ".join(set_clauses)
        + " WHERE project_id = :pid AND user_id = :uid",
        params,
    )
    return _affected(result)


async def delete_project(
    db: ArcadeClient, user_id: str, project_id: str
) -> dict[str, int]:
    """Cascade-delete a project: its member conversations and its non-global documents.

    Global documents (``is_global = true``) are *spared* — they are un-scoped from the project
    (``project_id`` cleared) and remain queryable everywhere. Returns counts ``{conversations,
    documents}`` of what was deleted, for the UI's confirmation toast. Scoped to ``user_id``.
    """
    members = await db.query(
        "SELECT conversation_id FROM Conversation WHERE user_id = :uid AND project_id = :pid",
        {"uid": user_id, "pid": project_id},
    )
    for row in members:
        cid = row.get("conversation_id")
        if cid:
            await delete_conversation(db, user_id, cid)
    docs = await db.command(
        "DELETE VERTEX FROM (SELECT FROM Document WHERE user_id = :uid AND project_id = :pid "
        "AND (is_global IS NULL OR is_global = false))",
        {"uid": user_id, "pid": project_id},
    )
    # Spare global docs: drop their project link so they survive as user-global references.
    await db.command(
        "UPDATE Document SET project_id = null "
        "WHERE user_id = :uid AND project_id = :pid AND is_global = true",
        {"uid": user_id, "pid": project_id},
    )
    await db.command(
        "DELETE VERTEX FROM (SELECT FROM Project WHERE project_id = :pid AND user_id = :uid)",
        {"pid": project_id, "uid": user_id},
    )
    return {"conversations": len(members), "documents": _affected(docs)}


async def set_conversation_swarm_settings(
    db: ArcadeClient,
    conversation_id: str,
    max_parallel: int | None = None,
    max_depth: int | None = None,
) -> None:
    """Store per-conversation swarm bounds (max concurrent agents / max orchestration depth).

    User-set from the web UI's Configuration card (swarm mode). Only the fields passed (non-None)
    are updated, so one can change without disturbing the other; ``None`` means "leave it / use the
    env default". Needs no DDL — ArcadeDB sets the field on write.
    """
    set_clauses: list[str] = []
    params: dict[str, Any] = {"cid": conversation_id}
    if max_parallel is not None:
        set_clauses.append("swarm_max_parallel = :mp")
        params["mp"] = max_parallel
    if max_depth is not None:
        set_clauses.append("swarm_max_depth = :md")
        params["md"] = max_depth
    if not set_clauses:
        return
    await db.command(
        "UPDATE Conversation SET " + ", ".join(set_clauses) + " WHERE conversation_id = :cid",
        params,
    )


async def get_conversation_swarm_settings(
    db: ArcadeClient, conversation_id: str
) -> dict[str, int | None]:
    """Return the conversation's swarm overrides, each ``None`` when unset (caller uses env default)."""
    rows = await db.query(
        "SELECT swarm_max_parallel, swarm_max_depth FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    row = rows[0] if rows else {}
    return {
        "max_parallel": row.get("swarm_max_parallel"),
        "max_depth": row.get("swarm_max_depth"),
    }


async def set_conversation_enabled_skills(
    db: ArcadeClient, conversation_id: str, names: list[str]
) -> None:
    """Store the marketplace skills enabled for this conversation (by skill name).

    User-set from the web UI's Configuration card. Stored as a JSON string (ArcadeDB sets the field
    on write — no DDL). An empty list clears the selection. The agent reads these each turn to know
    which skills to offer (descriptions injected) and mount into the sandbox.
    """
    await db.command(
        "UPDATE Conversation SET enabled_skills = :sk WHERE conversation_id = :cid",
        {"sk": json.dumps(list(names)), "cid": conversation_id},
    )


async def get_conversation_enabled_skills(
    db: ArcadeClient, conversation_id: str
) -> list[str]:
    """Return the conversation's enabled skill names, or ``[]`` when unset/unknown."""
    rows = await db.query(
        "SELECT enabled_skills FROM Conversation WHERE conversation_id = :cid",
        {"cid": conversation_id},
    )
    return _parse_skill_names(rows[0].get("enabled_skills") if rows else None)


def _parse_skill_names(raw: Any) -> list[str]:
    """Normalize a stored ``enabled_skills`` value (JSON string or native list) to ``list[str]``."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(n) for n in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [str(n) for n in parsed] if isinstance(parsed, list) else []
    return []


async def _search_messages_like(
    db: ArcadeClient, user_id: str, text: str, limit: int
) -> list[dict[str, Any]]:
    """Substring (LIKE) message search — the default, and the fallback when no embedding is given."""
    return await db.query(
        "SELECT content, created_at FROM Message "
        "WHERE user_id = :uid AND content LIKE :pat ORDER BY created_at DESC LIMIT :limit",
        {"uid": user_id, "pat": f"%{text}%", "limit": limit},
    )


async def search_messages(
    db: ArcadeClient,
    user_id: str,
    text: str,
    limit: int = 10,
    embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Search this user's past messages.

    When ``embedding`` is provided (semantic search enabled), rank by vector similarity via
    ArcadeDB's ``vectorNeighbors`` HNSW index; otherwise — or if the vector query errors / returns
    nothing (e.g. no embedded messages yet) — fall back to substring (LIKE) matching. The vector
    path can never abort the run: any error degrades to LIKE (same contract as :func:`search_facts`).
    """
    if embedding is None:
        return await _search_messages_like(db, user_id, text, limit)
    try:
        hits = await db.query(
            "SELECT content, created_at FROM ("
            "SELECT expand(vectorNeighbors('Message[embedding]', :qvec, :k))"
            ") WHERE user_id = :uid LIMIT :limit",
            {"qvec": embedding, "k": limit, "uid": user_id, "limit": limit},
        )
    except Exception:  # noqa: BLE001 — semantic search is best-effort; degrade to substring search.
        logger.warning("vector message search failed; falling back to LIKE", exc_info=True)
        return await _search_messages_like(db, user_id, text, limit)
    return hits or await _search_messages_like(db, user_id, text, limit)


async def append_run_messages(db: ArcadeClient, conversation_id: str, raw: str) -> None:
    """Persist one run's serialized Pydantic AI messages for *faithful* replay.

    ``raw`` is ``AgentRunResult.new_messages_json()`` (the run's delta: prompt, the assistant's
    text AND its tool calls, plus the tool returns). Stored verbatim so the next turn can rebuild
    the conversation with tool calls/results intact — unlike the human-readable ``Message`` vertices
    (which keep only role/content text for ``search_messages``/``get_recent_messages``). The two
    records serve different consumers and are written side by side.
    """
    run_id = _new_id()
    await db.command(
        "CREATE VERTEX RunMessages SET run_id = :rid, conversation_id = :cid, raw = :raw, created_at = :ts",
        {"rid": run_id, "cid": conversation_id, "raw": raw, "ts": _now()},
    )
    await db.command(
        "CREATE EDGE HAS_RUN_MESSAGES "
        "FROM (SELECT FROM Conversation WHERE conversation_id = :cid) "
        "TO (SELECT FROM RunMessages WHERE run_id = :rid)",
        {"cid": conversation_id, "rid": run_id},
    )


async def get_run_history(
    db: ArcadeClient,
    conversation_id: str,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Return the most recent runs' serialized message blobs, in chronological order.

    Each row's ``raw`` is one run's ``new_messages_json()``; concatenating them (oldest first)
    reconstructs the full faithful message history. ``limit`` bounds the number of *runs* loaded.
    """
    rows = await db.query(
        "SELECT raw, created_at FROM RunMessages "
        "WHERE conversation_id = :cid ORDER BY created_at DESC LIMIT :limit",
        {"cid": conversation_id, "limit": limit},
    )
    return list(reversed(rows))


async def store_fact(
    db: ArcadeClient,
    user_id: str,
    text: str,
    embedding: list[float] | None = None,
    important: bool = True,
) -> None:
    """Store a durable fact about the user and link it to the User vertex.

    When ``embedding`` is given (semantic search is enabled), it is stored on the Fact's vector
    property so :func:`search_facts` can rank by similarity. Omitted ⇒ the fact is still searchable
    via substring matching. ``important`` (default ``True`` — all facts are included by default)
    controls whether the fact is always loaded into the agent's per-turn context.
    """
    fact_id = _new_id()
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    set_clause = "fact_id = :fid, user_id = :uid, text = :text, important = :imp, created_at = :ts"
    params: dict[str, Any] = {
        "fid": fact_id,
        "uid": user_id,
        "text": text,
        "imp": important,
        "ts": _now(),
    }
    if embedding is not None:
        set_clause += ", embedding = :emb"
        params["emb"] = embedding
    await db.command(f"CREATE VERTEX Fact SET {set_clause}", params)
    await db.command(
        "CREATE EDGE KNOWS "
        "FROM (SELECT FROM User WHERE user_id = :uid) "
        "TO (SELECT FROM Fact WHERE fact_id = :fid)",
        {"uid": user_id, "fid": fact_id},
    )


async def _search_facts_like(
    db: ArcadeClient, user_id: str, text: str, limit: int
) -> list[dict[str, Any]]:
    """Substring (LIKE) fact search — the default, and the fallback when no embedding is given."""
    return await db.query(
        "SELECT fact_id, text, created_at FROM Fact "
        "WHERE user_id = :uid AND text LIKE :pat ORDER BY created_at DESC LIMIT :limit",
        {"uid": user_id, "pat": f"%{text}%", "limit": limit},
    )


async def list_facts(
    db: ArcadeClient,
    user_id: str,
    limit: int = 200,
    important_only: bool = False,
) -> list[dict[str, Any]]:
    """List this user's stored facts (newest first) for the UI and the prompt's important set.

    Returns ``fact_id, text, created_at, updated_at, important`` for each fact. Default-true
    semantics: a fact with no ``important`` value (legacy rows) is treated as important — so
    ``important_only`` filters out only those explicitly set to ``false``.
    """
    where = "user_id = :uid"
    if important_only:
        where += " AND important <> false"
    return await db.query(
        f"SELECT fact_id, text, important, created_at, updated_at FROM Fact "
        f"WHERE {where} ORDER BY created_at DESC LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )


async def search_facts(
    db: ArcadeClient,
    user_id: str,
    text: str,
    limit: int = 10,
    embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Search this user's stored facts (each hit includes fact_id for in-place updates).

    When ``embedding`` is provided (semantic search enabled), rank by vector similarity via
    ArcadeDB's ``vectorNeighbors`` HNSW index; otherwise — or if the vector query errors / returns
    nothing (e.g. no embedded facts yet) — fall back to substring (LIKE) matching. The vector path
    can never abort the run: any error degrades to LIKE.
    """
    if embedding is None:
        return await _search_facts_like(db, user_id, text, limit)
    try:
        hits = await db.query(
            "SELECT fact_id, text, created_at FROM ("
            "SELECT expand(vectorNeighbors('Fact[embedding]', :qvec, :k))"
            ") WHERE user_id = :uid LIMIT :limit",
            {"qvec": embedding, "k": limit, "uid": user_id, "limit": limit},
        )
    except Exception:  # noqa: BLE001 — semantic search is best-effort; degrade to substring search.
        logger.warning("vector fact search failed; falling back to LIKE", exc_info=True)
        return await _search_facts_like(db, user_id, text, limit)
    return hits or await _search_facts_like(db, user_id, text, limit)


async def update_fact(
    db: ArcadeClient,
    user_id: str,
    fact_id: str,
    text: str,
    embedding: list[float] | None = None,
    important: bool | None = None,
) -> int:
    """Replace the text of an existing fact (so the agent can revise instead of duplicating).

    Returns the number of facts updated (0 if no such fact for this user). Scoped by user_id and
    matched on the indexed fact_id. When ``embedding`` is given, the stored vector is refreshed too
    so semantic search stays consistent with the revised text. When ``important`` is not ``None``,
    the fact's inclusion flag is updated as well.
    """
    set_clause = "text = :text, updated_at = :ts"
    params: dict[str, Any] = {"text": text, "ts": _now(), "fid": fact_id, "uid": user_id}
    if embedding is not None:
        set_clause += ", embedding = :emb"
        params["emb"] = embedding
    if important is not None:
        set_clause += ", important = :imp"
        params["imp"] = important
    result = await db.command(
        f"UPDATE Fact SET {set_clause} WHERE fact_id = :fid AND user_id = :uid",
        params,
    )
    return _affected(result)


async def set_fact_importance(
    db: ArcadeClient, user_id: str, fact_id: str, important: bool
) -> int:
    """Toggle whether a fact is included in the agent's context. Returns count updated (0 if none).

    User-scoped and matched on the indexed fact_id, mirroring :func:`update_fact`. This is the write
    behind the UI's per-fact toggle (and the agent can reach the same flag via ``update_fact``).
    """
    result = await db.command(
        "UPDATE Fact SET important = :imp, updated_at = :ts WHERE fact_id = :fid AND user_id = :uid",
        {"imp": important, "ts": _now(), "fid": fact_id, "uid": user_id},
    )
    return _affected(result)


async def delete_fact(db: ArcadeClient, user_id: str, fact_id: str) -> int:
    """Delete a fact (and its KNOWS edge) by id, if it belongs to ``user_id``. Returns count deleted."""
    result = await db.command(
        "DELETE VERTEX FROM (SELECT FROM Fact WHERE fact_id = :fid AND user_id = :uid)",
        {"fid": fact_id, "uid": user_id},
    )
    return _affected(result)


async def create_document(
    db: ArcadeClient,
    user_id: str,
    conversation_id: str | None = None,
    title: str = "",
    content: str = "",
    mime_type: str = "text/markdown",
    encoding: str = "text",
    project_id: str | None = None,
    is_global: bool = False,
    embedding: list[float] | None = None,
) -> str:
    """Create a document and link it to its owner. Returns the document_id.

    Documents are durable artifacts (reports, notes, code listings, uploaded reference files). They
    surface in the web UI's Documents pane, where text-based ones can be edited by the user via
    :func:`update_document`. ``encoding`` is ``"text"`` for literal text or ``"base64"`` for binary
    artifacts (PDFs, images).

    Scope (pick one): pass ``conversation_id`` for a conversation-scoped document (the default,
    agent-authored or chat upload), or ``project_id`` for a project reference document the agent can
    query across the whole project. ``is_global`` marks a document available everywhere (and exempt
    from project cascade-delete). When ``embedding`` is given, it is stored for semantic search via
    :func:`search_documents`. The ``HAS_DOCUMENT`` edge is anchored to the Conversation when
    ``conversation_id`` is set, else the Project when ``project_id`` is set, else the User (so a
    global document is never orphaned).
    """
    document_id = _new_id()
    now = _now()
    set_clause = (
        "document_id = :did, user_id = :uid, title = :title, content = :content, "
        "mime_type = :mime, encoding = :enc, is_global = :glob, created_at = :ts, updated_at = :ts"
    )
    params: dict[str, Any] = {
        "did": document_id,
        "uid": user_id,
        "title": title,
        "content": content,
        "mime": mime_type,
        "enc": encoding,
        "glob": bool(is_global),
        "ts": now,
    }
    if conversation_id is not None:
        set_clause += ", conversation_id = :cid"
        params["cid"] = conversation_id
    if project_id is not None:
        set_clause += ", project_id = :pid"
        params["pid"] = project_id
    if embedding is not None:
        set_clause += ", embedding = :emb"
        params["emb"] = embedding
    await db.command("CREATE VERTEX Document SET " + set_clause, params)
    # Anchor the HAS_DOCUMENT edge to the most specific owner that exists.
    if conversation_id is not None:
        anchor = "SELECT FROM Conversation WHERE conversation_id = :aid"
        params_edge = {"aid": conversation_id, "did": document_id}
    elif project_id is not None:
        anchor = "SELECT FROM Project WHERE project_id = :aid"
        params_edge = {"aid": project_id, "did": document_id}
    else:
        anchor = "SELECT FROM User WHERE user_id = :aid"
        params_edge = {"aid": user_id, "did": document_id}
    await db.command(
        f"CREATE EDGE HAS_DOCUMENT FROM ({anchor}) "
        "TO (SELECT FROM Document WHERE document_id = :did)",
        params_edge,
    )
    return document_id


async def update_document(
    db: ArcadeClient,
    user_id: str,
    document_id: str,
    title: str | None = None,
    content: str | None = None,
) -> int:
    """Revise a document's title and/or content in place (agent revisions AND user edits).

    Returns the number of documents updated (0 if no such document for this user). Scoped by
    ``user_id`` and matched on the indexed ``document_id``. Pass only the fields to change.
    """
    set_clauses = ["updated_at = :ts"]
    params: dict[str, Any] = {"ts": _now(), "did": document_id, "uid": user_id}
    if title is not None:
        set_clauses.append("title = :title")
        params["title"] = title
    if content is not None:
        set_clauses.append("content = :content")
        params["content"] = content
    result = await db.command(
        "UPDATE Document SET " + ", ".join(set_clauses) + " WHERE document_id = :did AND user_id = :uid",
        params,
    )
    return _affected(result)


async def get_document(db: ArcadeClient, user_id: str, document_id: str) -> dict[str, Any] | None:
    """Return one document (full content included), or None if it doesn't exist for this user."""
    rows = await db.query(
        "SELECT document_id, conversation_id, project_id, is_global, title, content, mime_type, "
        "encoding, created_at, updated_at "
        "FROM Document WHERE document_id = :did AND user_id = :uid",
        {"did": document_id, "uid": user_id},
    )
    if not rows:
        return None
    row = rows[0]
    row["is_global"] = bool(row.get("is_global"))
    return row


async def set_document_global(
    db: ArcadeClient, user_id: str, document_id: str, is_global: bool
) -> int:
    """Mark a document global (available everywhere, exempt from project cascade-delete) or not.

    User-scoped and matched on the indexed ``document_id``. Returns rows updated (0 if not this
    user's). Backs the per-document "Global" toggle in the project documents UI.
    """
    result = await db.command(
        "UPDATE Document SET is_global = :g, updated_at = :ts "
        "WHERE document_id = :did AND user_id = :uid",
        {"g": bool(is_global), "ts": _now(), "did": document_id, "uid": user_id},
    )
    return _affected(result)


async def list_documents(
    db: ArcadeClient,
    user_id: str,
    conversation_id: str | None = None,
    limit: int = 50,
    project_id: str | None = None,
    include_global: bool = False,
) -> list[dict[str, Any]]:
    """Return this user's documents (metadata only, no content), most recently updated first.

    Scope (the filters compose with OR so the global set can be folded in):
    - ``conversation_id`` — restrict to one conversation (the Documents pane).
    - ``project_id`` — restrict to one project's reference documents.
    - ``include_global`` — also include the user's global documents (``is_global = true``),
      regardless of project/conversation. With no scope at all, returns all of the user's documents.
    Content is excluded so the list stays light — fetch one document's body with :func:`get_document`.
    """
    params: dict[str, Any] = {"uid": user_id, "limit": limit}
    clauses: list[str] = []
    if conversation_id is not None:
        clauses.append("conversation_id = :cid")
        params["cid"] = conversation_id
    if project_id is not None:
        clauses.append("project_id = :pid")
        params["pid"] = project_id
    if include_global:
        clauses.append("is_global = true")
    where = "user_id = :uid"
    if clauses:
        where += " AND (" + " OR ".join(clauses) + ")"
    rows = await db.query(
        "SELECT document_id, conversation_id, project_id, is_global, title, mime_type, encoding, "
        f"created_at, updated_at FROM Document WHERE {where} ORDER BY updated_at DESC LIMIT :limit",
        params,
    )
    for row in rows:
        row["is_global"] = bool(row.get("is_global"))
    return rows


async def _search_documents_like(
    db: ArcadeClient, user_id: str, text: str, limit: int, scope: str, params: dict[str, Any]
) -> list[dict[str, Any]]:
    p = {**params, "uid": user_id, "pat": f"%{text}%", "limit": limit}
    return await db.query(
        "SELECT document_id, title, content, project_id, is_global FROM Document "
        f"WHERE user_id = :uid AND ({scope}) AND content LIKE :pat "
        "ORDER BY updated_at DESC LIMIT :limit",
        p,
    )


async def search_documents(
    db: ArcadeClient,
    user_id: str,
    text: str,
    limit: int = 8,
    embedding: list[float] | None = None,
    project_id: str | None = None,
    include_global: bool = True,
) -> list[dict[str, Any]]:
    """Search a project's (and optionally the user's global) reference documents.

    Mirrors :func:`search_facts`: when ``embedding`` is given, rank by vector similarity via the
    ``vectorNeighbors`` HNSW index, else (or on any vector error / empty result) fall back to
    substring (LIKE) matching. Best-effort — the vector path can never abort the run. Each hit
    carries ``document_id``/``title``/``content`` so the agent can cite it directly.
    """
    # Build the scope predicate shared by both the vector and LIKE paths.
    scope_parts: list[str] = []
    scope_params: dict[str, Any] = {}
    if project_id is not None:
        scope_parts.append("project_id = :pid")
        scope_params["pid"] = project_id
    if include_global:
        scope_parts.append("is_global = true")
    if not scope_parts:
        return []
    scope = " OR ".join(scope_parts)

    if embedding is None:
        return await _search_documents_like(db, user_id, text, limit, scope, scope_params)
    try:
        hits = await db.query(
            "SELECT document_id, title, content, project_id, is_global FROM ("
            "SELECT expand(vectorNeighbors('Document[embedding]', :qvec, :k))"
            f") WHERE user_id = :uid AND ({scope}) LIMIT :limit",
            {"qvec": embedding, "k": limit, "uid": user_id, "limit": limit, **scope_params},
        )
    except Exception:  # noqa: BLE001 — semantic search is best-effort; degrade to substring search.
        logger.warning("vector document search failed; falling back to LIKE", exc_info=True)
        return await _search_documents_like(db, user_id, text, limit, scope, scope_params)
    return hits or await _search_documents_like(db, user_id, text, limit, scope, scope_params)


async def delete_document(db: ArcadeClient, user_id: str, document_id: str) -> int:
    """Delete a document (and its HAS_DOCUMENT edge) by id, if it belongs to ``user_id``."""
    result = await db.command(
        "DELETE VERTEX FROM (SELECT FROM Document WHERE document_id = :did AND user_id = :uid)",
        {"did": document_id, "uid": user_id},
    )
    return _affected(result)


async def create_agent_spec(
    db: ArcadeClient,
    user_id: str,
    name: str,
    role: str,
    instructions: str,
    tools: list[str],
    recipients: list[str] | None = None,
    skills: list[str] | None = None,
) -> str:
    """Create a swarm sub-agent definition, linked to the User. Returns the agent_id.

    AgentSpec vertices are the swarm's persistent roster: the orchestrator defines a specialist
    once (name, role, system prompt, granted tool groups, the teammates it may ``send_message``
    via ``recipients`` — the agency communication chart — and the marketplace ``skills`` it is
    granted) and re-dispatches it across turns and conversations. Name uniqueness per user is
    enforced by the capability layer (it checks :func:`get_agent_spec` first), not by the database.
    """
    agent_id = _new_id()
    now = _now()
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    await db.command(
        "CREATE VERTEX AgentSpec SET agent_id = :aid, user_id = :uid, name = :name, "
        "role = :role, instructions = :instructions, tools = :tools, recipients = :recipients, "
        "skills = :skills, created_at = :ts, updated_at = :ts",
        {
            "aid": agent_id,
            "uid": user_id,
            "name": name,
            "role": role,
            "instructions": instructions,
            "tools": tools,
            "recipients": list(recipients or []),
            "skills": list(skills or []),
            "ts": now,
        },
    )
    await db.command(
        "CREATE EDGE HAS_AGENT "
        "FROM (SELECT FROM User WHERE user_id = :uid) "
        "TO (SELECT FROM AgentSpec WHERE agent_id = :aid)",
        {"uid": user_id, "aid": agent_id},
    )
    return agent_id


def _normalize_agent_spec(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce an AgentSpec row's list fields to real lists.

    ArcadeDB returns ``null`` for a property that was never assigned (e.g. specs seeded before the
    ``skills`` column existed), so a raw row can carry ``skills=None``. Every consumer — the web
    roster UI and the swarm runtime that dispatches the spec — expects lists, so a stray null must
    never leak out: ``None`` becomes ``[]`` here, the single read boundary.
    """
    for field in ("tools", "recipients", "skills"):
        if row.get(field) is None:
            row[field] = []
    return row


async def get_agent_spec(db: ArcadeClient, user_id: str, ref: str) -> dict[str, Any] | None:
    """Return one sub-agent spec by its agent_id OR its name, or None. User-scoped."""
    rows = await db.query(
        "SELECT agent_id, name, role, instructions, tools, recipients, skills, created_at, updated_at "
        "FROM AgentSpec WHERE user_id = :uid AND (agent_id = :ref OR name = :ref)",
        {"uid": user_id, "ref": ref},
    )
    return _normalize_agent_spec(rows[0]) if rows else None


async def list_agent_specs(db: ArcadeClient, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return this user's swarm roster (full specs, instructions included), oldest first."""
    rows = await db.query(
        "SELECT agent_id, name, role, instructions, tools, recipients, skills, created_at, updated_at "
        "FROM AgentSpec WHERE user_id = :uid ORDER BY created_at ASC LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )
    return [_normalize_agent_spec(r) for r in rows]


async def update_agent_spec(
    db: ArcadeClient,
    user_id: str,
    agent_id: str,
    role: str | None = None,
    instructions: str | None = None,
    tools: list[str] | None = None,
    recipients: list[str] | None = None,
    skills: list[str] | None = None,
) -> int:
    """Revise a sub-agent spec in place. Returns the number updated (0 if not this user's).

    Pass only the fields to change; ``name`` is immutable (it is how the orchestrator and the
    stored reports refer to the agent). ``recipients`` rewires this agent's outgoing edges in the
    communication chart; ``skills`` replaces the marketplace skills it is granted.
    """
    set_clauses = ["updated_at = :ts"]
    params: dict[str, Any] = {"ts": _now(), "aid": agent_id, "uid": user_id}
    if role is not None:
        set_clauses.append("role = :role")
        params["role"] = role
    if instructions is not None:
        set_clauses.append("instructions = :instructions")
        params["instructions"] = instructions
    if tools is not None:
        set_clauses.append("tools = :tools")
        params["tools"] = tools
    if recipients is not None:
        set_clauses.append("recipients = :recipients")
        params["recipients"] = list(recipients)
    if skills is not None:
        set_clauses.append("skills = :skills")
        params["skills"] = list(skills)
    result = await db.command(
        "UPDATE AgentSpec SET " + ", ".join(set_clauses) + " WHERE agent_id = :aid AND user_id = :uid",
        params,
    )
    return _affected(result)


async def delete_agent_spec(db: ArcadeClient, user_id: str, agent_id: str) -> int:
    """Delete a sub-agent spec (and its HAS_AGENT edge) by id, if it belongs to ``user_id``."""
    result = await db.command(
        "DELETE VERTEX FROM (SELECT FROM AgentSpec WHERE agent_id = :aid AND user_id = :uid)",
        {"aid": agent_id, "uid": user_id},
    )
    return _affected(result)


async def create_skill(
    db: ArcadeClient,
    user_id: str,
    name: str,
    description: str,
    body: str,
    files: dict[str, dict[str, str]] | None = None,
    source: str = "",
) -> str:
    """Create a synced marketplace skill, linked to the User. Returns the skill_id.

    A Skill is one Anthropic Agent Skill: its ``description`` (frontmatter, injected per-turn for
    progressive disclosure), ``body`` (the SKILL.md instructions, loaded on demand via load_skill),
    and ``files`` — the bundled scripts/assets/references as a map of ``relpath -> {content,
    encoding}`` (``encoding`` is ``"text"`` or ``"base64"``, the Document convention), mounted into
    the run_python sandbox. ``source`` records where it came from (e.g. ``anthropics/skills@main``).
    Name uniqueness per user is maintained by :func:`upsert_skill` (the sync path).
    """
    skill_id = _new_id()
    now = _now()
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    await db.command(
        "CREATE VERTEX Skill SET skill_id = :sid, user_id = :uid, name = :name, "
        "description = :description, body = :body, files = :files, source = :source, "
        "synced_at = :ts",
        {
            "sid": skill_id,
            "uid": user_id,
            "name": name,
            "description": description,
            "body": body,
            "files": json.dumps(files or {}),
            "source": source,
            "ts": now,
        },
    )
    await db.command(
        "CREATE EDGE HAS_SKILL "
        "FROM (SELECT FROM User WHERE user_id = :uid) "
        "TO (SELECT FROM Skill WHERE skill_id = :sid)",
        {"uid": user_id, "sid": skill_id},
    )
    return skill_id


async def upsert_skill(
    db: ArcadeClient,
    user_id: str,
    name: str,
    description: str,
    body: str,
    files: dict[str, dict[str, str]] | None = None,
    source: str = "",
) -> str:
    """Create or replace a user's skill by ``name`` (the sync path). Returns its skill_id.

    Keeps exactly one row per skill name per user: if the named skill already exists it is updated
    in place (so re-syncing refreshes content without duplicating); otherwise it is created.
    """
    existing = await get_skill(db, user_id, name)
    if existing:
        await db.command(
            "UPDATE Skill SET description = :description, body = :body, files = :files, "
            "source = :source, synced_at = :ts WHERE skill_id = :sid AND user_id = :uid",
            {
                "description": description,
                "body": body,
                "files": json.dumps(files or {}),
                "source": source,
                "ts": _now(),
                "sid": existing["skill_id"],
                "uid": user_id,
            },
        )
        return str(existing["skill_id"])
    return await create_skill(db, user_id, name, description, body, files, source)


async def get_skill(db: ArcadeClient, user_id: str, ref: str) -> dict[str, Any] | None:
    """Return one skill (full body + files) by its skill_id OR its name, or None. User-scoped.

    The ``files`` field is parsed back from its stored JSON to a ``relpath -> {content, encoding}``
    dict (``{}`` when absent/corrupt), so callers get a ready-to-use map.
    """
    rows = await db.query(
        "SELECT skill_id, name, description, body, files, source, synced_at "
        "FROM Skill WHERE user_id = :uid AND (skill_id = :ref OR name = :ref)",
        {"uid": user_id, "ref": ref},
    )
    if not rows:
        return None
    row = rows[0]
    raw_files = row.get("files")
    if isinstance(raw_files, str):
        try:
            row["files"] = json.loads(raw_files) if raw_files else {}
        except json.JSONDecodeError:
            row["files"] = {}
    elif not isinstance(raw_files, dict):
        row["files"] = {}
    return row


async def list_skills(db: ArcadeClient, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return this user's synced skills (metadata only — no body/files), name-ordered.

    Backs the Configuration card's skill picker. Content is excluded to keep the list light; fetch
    a skill's body/files with :func:`get_skill`.
    """
    return await db.query(
        "SELECT skill_id, name, description, source, synced_at "
        "FROM Skill WHERE user_id = :uid ORDER BY name ASC LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )


async def delete_skill(db: ArcadeClient, user_id: str, ref: str) -> int:
    """Delete a skill (and its HAS_SKILL edge) by skill_id or name, if it belongs to ``user_id``."""
    result = await db.command(
        "DELETE VERTEX FROM (SELECT FROM Skill WHERE user_id = :uid AND (skill_id = :ref OR name = :ref))",
        {"uid": user_id, "ref": ref},
    )
    return _affected(result)


async def vertex_type_exists(db: ArcadeClient, type_name: str) -> bool:
    """True if a type named ``type_name`` already exists in this database's schema."""
    rows = await db.query("SELECT FROM schema:types")
    return any(r.get("name") == type_name for r in rows)


async def type_category(db: ArcadeClient, type_name: str) -> str | None:
    """Return ``'vertex'``/``'edge'`` for ``type_name``, or ``None`` if no such type exists.

    Lets a drop be routed to the right cleanup (DELETE VERTEX vs DELETE … UNSAFE) and prevents
    dropping a vertex type with the edge tool (or vice versa). ``schema:types`` carries the
    discriminator in its ``type`` field.
    """
    rows = await db.query("SELECT FROM schema:types")
    for row in rows:
        if row.get("name") == type_name:
            return row.get("type")
    return None


async def drop_vertex_type(db: ArcadeClient, type_name: str) -> int:
    """Delete every instance of a vertex type (cleaning their edges), then drop the type itself.

    Returns the number of instances removed. ``type_name`` is interpolated (DDL can't bind
    identifiers), so callers MUST pass a value already validated by ``DropVertexTypeArgs`` and
    confirmed to be a vertex type. ``DELETE VERTEX`` removes connected edges, so no dangling
    references survive; ``DROP TYPE`` then succeeds (it refuses a non-empty type).
    """
    deleted = await db.command(f"DELETE VERTEX FROM {type_name}")
    count = _affected(deleted)
    await db.command(f"DROP TYPE {type_name} IF EXISTS")
    return count


async def drop_edge_type(db: ArcadeClient, type_name: str) -> int:
    """Delete every edge of an edge type, then drop the type itself. Returns the number removed.

    ``type_name`` is interpolated, so callers MUST pass a value validated by ``DropEdgeTypeArgs`` and
    confirmed to be an *edge* type — ``DELETE … UNSAFE`` on a vertex type would strip records without
    cleaning adjacency. On edges, ``UNSAFE`` is required to delete the records and (verified against
    ArcadeDB) does update the endpoints' adjacency, so no dangling references survive; ``DROP TYPE``
    then succeeds.
    """
    deleted = await db.command(f"DELETE FROM {type_name} UNSAFE")
    count = _affected(deleted)
    await db.command(f"DROP TYPE {type_name} IF EXISTS")
    return count


async def list_vertex_types(db: ArcadeClient) -> list[dict[str, Any]]:
    """Return the current ontology: each type's name, usage note, and property names.

    The usage note is the type-level CUSTOM ``description`` set by :func:`create_vertex_type`,
    so callers can read "when to use this type" and reuse an existing generic type.
    """
    rows = await db.query("SELECT FROM schema:types")
    types: list[dict[str, Any]] = []
    for row in rows:
        custom = row.get("custom") or {}
        is_custom = isinstance(custom, dict)
        usage = custom.get("description") if is_custom else None
        kind = custom.get("kind") if is_custom else None
        props = row.get("properties") or []
        prop_names = [p.get("name") for p in props if isinstance(p, dict) and p.get("name")]
        parents = row.get("parentTypes") or []
        parent = parents[0] if isinstance(parents, list) and parents else None
        types.append(
            {
                "name": row.get("name"),
                "usage": usage,
                "parent_type": parent,
                "kind": kind,
                "properties": prop_names,
            }
        )
    return types


async def create_vertex_type(
    db: ArcadeClient,
    type_name: str,
    usage: str,
    properties: dict[str, str] | None = None,
    parent_type: str | None = None,
    kind: str = "semantic",
) -> bool:
    """Create a vertex type, attach its usage instruction, and add its properties (idempotent).

    Returns True if the type was newly created. ``type_name``/property names/``parent_type`` are
    interpolated (DDL cannot bind identifiers); callers MUST pass values already validated by the
    Pydantic models in ``backend.schemas.graph_schemas``. ``usage`` is stored as type-level CUSTOM
    metadata (key ``description``) so future runs can read it via ``schema:types``. When
    ``parent_type`` is given the new type EXTENDS it (single inheritance), so the agent can grow a
    hierarchy (e.g. ``Pet`` extending ``Animal``).

    ``kind`` (``'semantic'``/``'episodic'``) is stored the same way (CUSTOM ``kind``) so retrieval
    and the UI can tell durable state from time-ordered events. An ``episodic`` type also gets an
    ``occurred_at DATETIME`` property — the timestamp of *when the event happened* (distinct from the
    bookkeeping ``created_at`` that ``create_node`` stamps for when it was recorded).
    """
    newly_created = not await vertex_type_exists(db, type_name)
    # `IF NOT EXISTS` is a SUFFIX for types (ArcadeDB quirk; see ensure_schema) and, verified against
    # ArcadeDB, must come BEFORE `EXTENDS` — `... EXTENDS <parent> IF NOT EXISTS` is a parse error.
    extends = f" EXTENDS {parent_type}" if parent_type else ""
    await db.command(f"CREATE VERTEX TYPE {type_name} IF NOT EXISTS{extends}")
    # Type-level documentation. CUSTOM *values* are string literals, so they bind as parameters.
    await db.command(f"ALTER TYPE {type_name} CUSTOM description = :usage", {"usage": usage})
    await db.command(f"ALTER TYPE {type_name} CUSTOM kind = :kind", {"kind": kind})
    if kind == "episodic":
        await db.command(f"CREATE PROPERTY {type_name}.occurred_at IF NOT EXISTS DATETIME")
    for prop_name, prop_type in (properties or {}).items():
        await db.command(f"CREATE PROPERTY {type_name}.{prop_name} IF NOT EXISTS {prop_type}")
    return newly_created


async def create_edge_type(
    db: ArcadeClient,
    type_name: str,
    usage: str,
    properties: dict[str, str] | None = None,
) -> bool:
    """Create an edge (relationship) type, attach its usage note, add its properties (idempotent).

    Returns True if newly created. Mirrors :func:`create_vertex_type` but for edges. ``type_name``
    and property names are interpolated (validated by ``ProposeEdgeArgs``); ``usage`` binds as a
    parameter and is stored as type-level CUSTOM metadata. Type names are unique across the schema,
    so :func:`vertex_type_exists` answers correctly for edge types too.
    """
    newly_created = not await vertex_type_exists(db, type_name)
    await db.command(f"CREATE EDGE TYPE {type_name} IF NOT EXISTS")
    await db.command(f"ALTER TYPE {type_name} CUSTOM description = :usage", {"usage": usage})
    for prop_name, prop_type in (properties or {}).items():
        await db.command(f"CREATE PROPERTY {type_name}.{prop_name} IF NOT EXISTS {prop_type}")
    return newly_created


async def node_type(db: ArcadeClient, rid: str) -> str | None:
    """The type name of the record at ``rid`` (e.g. 'Person'), or None if it doesn't exist.

    A well-formed but non-existent rid (e.g. one whose bucket doesn't exist) makes ArcadeDB answer
    500 rather than an empty result, so any lookup error is treated as "not found" — callers use
    this to give the agent a friendly message, not to diagnose the DB.
    """
    if not _RID_RE.match(rid):
        return None
    try:
        rows = await db.query(f"SELECT @type AS t FROM {rid}")
    except httpx.HTTPStatusError as exc:
        logger.debug("node_type(%s) lookup failed (%s); treating as not found", rid, exc)
        return None
    return rows[0].get("t") if rows else None


async def node_exists(db: ArcadeClient, rid: str) -> bool:
    """True if a record with id ``rid`` (e.g. ``#29:0``) exists."""
    return await node_type(db, rid) is not None


async def update_node(db: ArcadeClient, user_id: str, rid: str, properties: dict[str, Any]) -> int:
    """Set/overwrite ``properties`` on the node at ``rid``. Returns the number of records updated.

    Scoped by ``user_id`` (defense-in-depth). ``rid``/property names are interpolated (validated by
    ``UpdateNodeArgs``); property values bind as parameters.
    """
    if not properties:
        return 0
    set_clauses, params = [], {"uid": user_id}
    for key, value in properties.items():
        set_clauses.append(f"{key} = :p_{key}")
        params[f"p_{key}"] = value
    result = await db.command(
        f"UPDATE {rid} SET " + ", ".join(set_clauses) + " WHERE user_id = :uid", params
    )
    return _affected(result)


async def delete_node(db: ArcadeClient, user_id: str, rid: str) -> int:
    """Delete the node at ``rid`` (and its edges) if it belongs to ``user_id``. Returns count deleted."""
    result = await db.command(
        f"DELETE VERTEX FROM (SELECT FROM {rid} WHERE user_id = :uid)", {"uid": user_id}
    )
    return _affected(result)


async def create_edge(
    db: ArcadeClient,
    edge_type: str,
    from_rid: str,
    to_rid: str,
    properties: dict[str, Any] | None = None,
) -> str:
    """Create an ``edge_type`` edge from ``from_rid`` to ``to_rid``. Returns the edge's @rid.

    ``edge_type`` / property names / rids are interpolated (callers MUST pass values validated by
    ``CreateEdgeArgs``); property *values* bind as parameters.
    """
    set_clause = ""
    params: dict[str, Any] = {}
    if properties:
        clauses = []
        for key, value in properties.items():
            clauses.append(f"{key} = :p_{key}")
            params[f"p_{key}"] = value
        set_clause = " SET " + ", ".join(clauses)
    created = await db.command(
        f"CREATE EDGE {edge_type} FROM {from_rid} TO {to_rid}{set_clause}", params
    )
    rid = str(created[0]["@rid"]) if created and created[0].get("@rid") else None
    return rid or "created"


async def create_node(
    db: ArcadeClient,
    user_id: str,
    node_type: str,
    properties: dict[str, Any],
) -> str:
    """Create an instance (record) of ``node_type``, stamp it, and link it to the User.

    Returns the new record's @rid (or ``"created"`` if the backend returned none). ``node_type``
    and property *names* are interpolated (DDL/identifiers can't bind), so callers MUST pass values
    already validated by ``CreateNodeArgs``; property *values* bind as parameters. The node is
    linked to the user via a ``HAS_NODE`` edge so it is reachable by traversal from the User vertex.
    """
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    set_clauses = ["user_id = :uid", "created_at = :ts"]
    params: dict[str, Any] = {"uid": user_id, "ts": _now()}
    for key, value in properties.items():
        set_clauses.append(f"{key} = :p_{key}")
        params[f"p_{key}"] = value
    created = await db.command(
        f"CREATE VERTEX {node_type} SET " + ", ".join(set_clauses), params
    )

    rid = str(created[0]["@rid"]) if created and created[0].get("@rid") else None
    if rid and _RID_RE.match(rid):
        await db.command(
            f"CREATE EDGE HAS_NODE FROM (SELECT FROM User WHERE user_id = :uid) TO {rid}",
            {"uid": user_id},
        )
    return rid or "created"


def _sanitize_rid(rid: str) -> str:
    """Turn an ArcadeDB record id into a DOM/JS-safe id for the UI (``#38:0`` -> ``38_0``)."""
    return rid.lstrip("#").replace(":", "_")


# Row keys that are ArcadeDB internals or bookkeeping, not user-facing instance properties.
_NON_DISPLAY_KEYS = frozenset({"@rid", "@type", "@cat", "@in", "@out", "user_id"})


async def get_user_graph(
    db: ArcadeClient, user_id: str, limit: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """Serialize this user's agent-built knowledge graph (nodes + edges) for UI rendering.

    Returns ``{"nodes": [...], "edges": [...]}`` with DOM-safe ids (``#38:0`` -> ``38_0``):
    nodes are the instance vertices reachable from the User via ``HAS_NODE`` (depth-limited by
    ``limit``); edges are the agent-created relationships whose endpoints are *both* among those
    nodes. Internal edges are naturally excluded: instances only ever have an *incoming* ``HAS_NODE``
    from the User (and no outgoing internal edges), so the outgoing-edge traversal yields only
    agent-created relationships, and the both-endpoints-in-set filter drops any pointing past the
    limit. Scoped to this user (and the per-user database). Each node carries its type, memory
    ``kind`` (semantic/episodic, resolved once from ``schema:types``) and remaining scalar
    properties so the frontend can label, color, and inspect it.
    """
    # Resolve each type's memory kind once so every node can be tagged (semantic vs. episodic).
    kind_by_type = {t["name"]: t.get("kind") for t in await list_vertex_types(db) if t.get("name")}
    rows = await db.query(
        "SELECT *, @rid, @type FROM (SELECT expand(out('HAS_NODE')) FROM User WHERE user_id = :uid) "
        "LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )
    nodes: list[dict[str, Any]] = []
    rid_set: set[str] = set()
    for row in rows:
        rid = str(row.get("@rid") or "")
        if not rid:
            continue
        rid_set.add(rid)
        props = {
            k: v for k, v in row.items() if k not in _NON_DISPLAY_KEYS and not k.startswith("@")
        }
        # Prefer a human label: a 'name' property, else the first string prop, else the type.
        label = props.get("name")
        if not isinstance(label, str):
            label = next((v for v in props.values() if isinstance(v, str)), None)
        node_type = row.get("@type")
        nodes.append(
            {
                "id": _sanitize_rid(rid),
                "type": node_type,
                "label": label or node_type,
                "kind": kind_by_type.get(node_type),
                "properties": props,
            }
        )

    edges: list[dict[str, Any]] = []
    if rid_set:
        # Outgoing edges of every instance node. ArcadeDB exposes an edge's endpoints as @out/@in;
        # `FROM E` is not queryable, so we expand outE() from the instances instead.
        edge_rows = await db.query(
            "SELECT @rid AS rid, @type AS type, @out AS src, @in AS dst FROM ("
            "SELECT expand(outE()) FROM (SELECT expand(out('HAS_NODE')) FROM User WHERE user_id = :uid)"
            ")",
            {"uid": user_id},
        )
        for e in edge_rows:
            src, dst = str(e.get("src") or ""), str(e.get("dst") or "")
            # Keep only edges between two displayed instance nodes (drops any past the limit).
            if src not in rid_set or dst not in rid_set:
                continue
            edges.append(
                {
                    "id": _sanitize_rid(str(e.get("rid") or f"{src}-{dst}")),
                    "source": _sanitize_rid(src),
                    "target": _sanitize_rid(dst),
                    "label": e.get("type"),
                }
            )
    return {"nodes": nodes, "edges": edges}


async def get_recent_events(
    db: ArcadeClient,
    user_id: str,
    limit: int = 10,
    types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return this user's recent EPISODIC instances, most-recent event first.

    Episodic types are resolved from ``schema:types`` (their CUSTOM ``kind``); the agent-created
    instances reachable from the User via ``HAS_NODE`` are filtered to those types and ordered by
    ``occurred_at DESC`` (falling back to ``created_at`` when an instance has no ``occurred_at``).
    Pass ``types`` to restrict to specific episodic type names (each validated PascalCase). Returns
    ``[]`` when no episodic types exist. This mirrors research.md's "time-series for events" split
    (semantic recall stays on ``search_facts``/``search_memory``).
    """
    episodic = {t["name"] for t in await list_vertex_types(db) if t.get("kind") == "episodic" and t.get("name")}
    if types is not None:
        episodic &= set(types)
    if not episodic:
        return []
    # Type names are validated PascalCase identifiers, so interpolating the IN-list is safe.
    # `*` already carries occurred_at/created_at; rid/type are aliased for the caller.
    in_list = ", ".join(f"'{name}'" for name in sorted(episodic))
    return await db.query(
        "SELECT *, @rid AS rid, @type AS type "
        "FROM (SELECT expand(out('HAS_NODE')) FROM User WHERE user_id = :uid) "
        f"WHERE @type IN [{in_list}] "
        "ORDER BY occurred_at DESC, created_at DESC LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )


async def write_log(
    db: ArcadeClient,
    conversation_id: str,
    level: str,
    event: str,
    payload: Any = None,
) -> None:
    """Write a LogEntry (tool call, error, lifecycle event) and link it to the conversation."""
    log_id = _new_id()
    payload_str = json.dumps(payload, default=str) if payload is not None else None
    await db.command(
        "CREATE VERTEX LogEntry SET log_id = :lid, conversation_id = :cid, "
        "level = :level, event = :event, payload = :payload, created_at = :ts",
        {
            "lid": log_id,
            "cid": conversation_id,
            "level": level,
            "event": event,
            "payload": payload_str,
            "ts": _now(),
        },
    )
    await db.command(
        "CREATE EDGE LOGGED "
        "FROM (SELECT FROM Conversation WHERE conversation_id = :cid) "
        "TO (SELECT FROM LogEntry WHERE log_id = :lid)",
        {"cid": conversation_id, "lid": log_id},
    )

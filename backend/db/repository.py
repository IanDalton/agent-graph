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
) -> None:
    """Ensure the User and Conversation vertices (and their link) exist. Idempotent.

    ``mode`` selects the agent profile for this conversation ('regular'/'research'/'swarm' — see
    ``backend.main.MODES``). It is fixed at creation: the persistence hook's idempotent call (which
    always passes the default) cannot overwrite a mode the API set, because an existing
    conversation returns early.
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
    await db.command(
        "CREATE VERTEX Conversation SET conversation_id = :cid, user_id = :uid, "
        "title = :title, mode = :mode, started_at = :ts",
        {"cid": conversation_id, "uid": user_id, "title": title, "mode": mode, "ts": _now()},
    )
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
) -> None:
    """Persist a single message and link it to its conversation."""
    message_id = _new_id()
    await db.command(
        "CREATE VERTEX Message SET message_id = :mid, conversation_id = :cid, "
        "user_id = :uid, role = :role, content = :content, created_at = :ts",
        {
            "mid": message_id,
            "cid": conversation_id,
            "uid": user_id,
            "role": role,
            "content": content,
            "ts": _now(),
        },
    )
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
    """Return the most recent messages of a conversation in chronological order."""
    rows = await db.query(
        "SELECT role, content, created_at FROM Message "
        "WHERE conversation_id = :cid ORDER BY created_at DESC LIMIT :limit",
        {"cid": conversation_id, "limit": limit},
    )
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


async def list_conversations(
    db: ArcadeClient,
    user_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return this user's conversations, most recently started first.

    Backs the UI's left-pane conversation list. Scoped by ``user_id`` (defense-in-depth on
    top of the per-user database). ``mode`` is the stored agent profile; conversations created
    before modes existed have none and report ``"regular"``.
    """
    rows = await db.query(
        "SELECT conversation_id, title, mode, started_at FROM Conversation "
        "WHERE user_id = :uid ORDER BY started_at DESC LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )
    for row in rows:
        row["mode"] = row.get("mode") or "regular"
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


async def search_messages(
    db: ArcadeClient,
    user_id: str,
    text: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Substring search across this user's past messages."""
    return await db.query(
        "SELECT content, created_at FROM Message "
        "WHERE user_id = :uid AND content LIKE :pat ORDER BY created_at DESC LIMIT :limit",
        {"uid": user_id, "pat": f"%{text}%", "limit": limit},
    )


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
    db: ArcadeClient, user_id: str, text: str, embedding: list[float] | None = None
) -> None:
    """Store a durable fact about the user and link it to the User vertex.

    When ``embedding`` is given (semantic search is enabled), it is stored on the Fact's vector
    property so :func:`search_facts` can rank by similarity. Omitted ⇒ the fact is still searchable
    via substring matching.
    """
    fact_id = _new_id()
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    set_clause = "fact_id = :fid, user_id = :uid, text = :text, created_at = :ts"
    params: dict[str, Any] = {"fid": fact_id, "uid": user_id, "text": text, "ts": _now()}
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
) -> int:
    """Replace the text of an existing fact (so the agent can revise instead of duplicating).

    Returns the number of facts updated (0 if no such fact for this user). Scoped by user_id and
    matched on the indexed fact_id. When ``embedding`` is given, the stored vector is refreshed too
    so semantic search stays consistent with the revised text.
    """
    set_clause = "text = :text, updated_at = :ts"
    params: dict[str, Any] = {"text": text, "ts": _now(), "fid": fact_id, "uid": user_id}
    if embedding is not None:
        set_clause += ", embedding = :emb"
        params["emb"] = embedding
    result = await db.command(
        f"UPDATE Fact SET {set_clause} WHERE fact_id = :fid AND user_id = :uid",
        params,
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
    conversation_id: str,
    title: str,
    content: str,
    mime_type: str = "text/markdown",
    encoding: str = "text",
) -> str:
    """Create an agent-authored document, linked to its conversation. Returns the document_id.

    Documents are durable artifacts (reports, notes, code listings) the agent produces for the
    user. They surface in the web UI's Documents pane, where text-based ones can be edited by
    the user via :func:`update_document`. ``encoding`` is ``"text"`` for literal text content or
    ``"base64"`` for binary artifacts (PDFs, images) produced by the Python sandbox.
    """
    document_id = _new_id()
    now = _now()
    await db.command(
        "CREATE VERTEX Document SET document_id = :did, conversation_id = :cid, "
        "user_id = :uid, title = :title, content = :content, mime_type = :mime, "
        "encoding = :enc, created_at = :ts, updated_at = :ts",
        {
            "did": document_id,
            "cid": conversation_id,
            "uid": user_id,
            "title": title,
            "content": content,
            "mime": mime_type,
            "enc": encoding,
            "ts": now,
        },
    )
    await db.command(
        "CREATE EDGE HAS_DOCUMENT "
        "FROM (SELECT FROM Conversation WHERE conversation_id = :cid) "
        "TO (SELECT FROM Document WHERE document_id = :did)",
        {"cid": conversation_id, "did": document_id},
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
        "SELECT document_id, conversation_id, title, content, mime_type, encoding, "
        "created_at, updated_at "
        "FROM Document WHERE document_id = :did AND user_id = :uid",
        {"did": document_id, "uid": user_id},
    )
    return rows[0] if rows else None


async def list_documents(
    db: ArcadeClient,
    user_id: str,
    conversation_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return this user's documents (metadata only, no content), most recently updated first.

    Pass ``conversation_id`` to restrict to one conversation (the Documents pane does); omit it
    for all of the user's documents. Content is excluded so the list stays light — fetch one
    document's body with :func:`get_document`.
    """
    where = "user_id = :uid"
    params: dict[str, Any] = {"uid": user_id, "limit": limit}
    if conversation_id is not None:
        where += " AND conversation_id = :cid"
        params["cid"] = conversation_id
    return await db.query(
        "SELECT document_id, conversation_id, title, mime_type, encoding, created_at, updated_at "
        f"FROM Document WHERE {where} ORDER BY updated_at DESC LIMIT :limit",
        params,
    )


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
) -> str:
    """Create a swarm sub-agent definition, linked to the User. Returns the agent_id.

    AgentSpec vertices are the swarm's persistent roster: the orchestrator defines a specialist
    once (name, role, system prompt, granted tool groups) and re-dispatches it across turns and
    conversations. Name uniqueness per user is enforced by the capability layer (it checks
    :func:`get_agent_spec` first), not by the database.
    """
    agent_id = _new_id()
    now = _now()
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    await db.command(
        "CREATE VERTEX AgentSpec SET agent_id = :aid, user_id = :uid, name = :name, "
        "role = :role, instructions = :instructions, tools = :tools, "
        "created_at = :ts, updated_at = :ts",
        {
            "aid": agent_id,
            "uid": user_id,
            "name": name,
            "role": role,
            "instructions": instructions,
            "tools": tools,
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


async def get_agent_spec(db: ArcadeClient, user_id: str, ref: str) -> dict[str, Any] | None:
    """Return one sub-agent spec by its agent_id OR its name, or None. User-scoped."""
    rows = await db.query(
        "SELECT agent_id, name, role, instructions, tools, created_at, updated_at "
        "FROM AgentSpec WHERE user_id = :uid AND (agent_id = :ref OR name = :ref)",
        {"uid": user_id, "ref": ref},
    )
    return rows[0] if rows else None


async def list_agent_specs(db: ArcadeClient, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return this user's swarm roster (full specs, instructions included), oldest first."""
    return await db.query(
        "SELECT agent_id, name, role, instructions, tools, created_at, updated_at "
        "FROM AgentSpec WHERE user_id = :uid ORDER BY created_at ASC LIMIT :limit",
        {"uid": user_id, "limit": limit},
    )


async def update_agent_spec(
    db: ArcadeClient,
    user_id: str,
    agent_id: str,
    role: str | None = None,
    instructions: str | None = None,
    tools: list[str] | None = None,
) -> int:
    """Revise a sub-agent spec in place. Returns the number updated (0 if not this user's).

    Pass only the fields to change; ``name`` is immutable (it is how the orchestrator and the
    stored reports refer to the agent).
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

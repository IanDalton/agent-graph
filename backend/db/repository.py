"""Persistence layer for conversations, messages, facts and logs.

This is the single source of truth for all database access: both the agent
tools (in ``backend.skills.graph_capability``) and the automatic persistence
hooks call into these functions. All statements are parameterized ArcadeDB SQL
and scoped by ``user_id`` so each user's memory stays isolated.
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
) -> None:
    """Ensure the User and Conversation vertices (and their link) exist. Idempotent."""
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
        "title = :title, started_at = :ts",
        {"cid": conversation_id, "uid": user_id, "title": title, "ts": _now()},
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


async def store_fact(db: ArcadeClient, user_id: str, text: str) -> None:
    """Store a durable fact about the user and link it to the User vertex."""
    fact_id = _new_id()
    await db.command(
        "UPDATE User SET user_id = :uid UPSERT WHERE user_id = :uid",
        {"uid": user_id},
    )
    await db.command(
        "CREATE VERTEX Fact SET fact_id = :fid, user_id = :uid, text = :text, created_at = :ts",
        {"fid": fact_id, "uid": user_id, "text": text, "ts": _now()},
    )
    await db.command(
        "CREATE EDGE KNOWS "
        "FROM (SELECT FROM User WHERE user_id = :uid) "
        "TO (SELECT FROM Fact WHERE fact_id = :fid)",
        {"uid": user_id, "fid": fact_id},
    )


async def search_facts(
    db: ArcadeClient,
    user_id: str,
    text: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Substring search across this user's stored facts (includes fact_id for in-place updates)."""
    return await db.query(
        "SELECT fact_id, text, created_at FROM Fact "
        "WHERE user_id = :uid AND text LIKE :pat ORDER BY created_at DESC LIMIT :limit",
        {"uid": user_id, "pat": f"%{text}%", "limit": limit},
    )


async def update_fact(db: ArcadeClient, user_id: str, fact_id: str, text: str) -> int:
    """Replace the text of an existing fact (so the agent can revise instead of duplicating).

    Returns the number of facts updated (0 if no such fact for this user). Scoped by user_id and
    matched on the indexed fact_id.
    """
    result = await db.command(
        "UPDATE Fact SET text = :text, updated_at = :ts WHERE fact_id = :fid AND user_id = :uid",
        {"text": text, "ts": _now(), "fid": fact_id, "uid": user_id},
    )
    return _affected(result)


async def delete_fact(db: ArcadeClient, user_id: str, fact_id: str) -> int:
    """Delete a fact (and its KNOWS edge) by id, if it belongs to ``user_id``. Returns count deleted."""
    result = await db.command(
        "DELETE VERTEX FROM (SELECT FROM Fact WHERE fact_id = :fid AND user_id = :uid)",
        {"fid": fact_id, "uid": user_id},
    )
    return _affected(result)


async def vertex_type_exists(db: ArcadeClient, type_name: str) -> bool:
    """True if a type named ``type_name`` already exists in this database's schema."""
    rows = await db.query("SELECT FROM schema:types")
    return any(r.get("name") == type_name for r in rows)


async def list_vertex_types(db: ArcadeClient) -> list[dict[str, Any]]:
    """Return the current ontology: each type's name, usage note, and property names.

    The usage note is the type-level CUSTOM ``description`` set by :func:`create_vertex_type`,
    so callers can read "when to use this type" and reuse an existing generic type.
    """
    rows = await db.query("SELECT FROM schema:types")
    types: list[dict[str, Any]] = []
    for row in rows:
        custom = row.get("custom") or {}
        usage = custom.get("description") if isinstance(custom, dict) else None
        props = row.get("properties") or []
        prop_names = [p.get("name") for p in props if isinstance(p, dict) and p.get("name")]
        types.append({"name": row.get("name"), "usage": usage, "properties": prop_names})
    return types


async def create_vertex_type(
    db: ArcadeClient,
    type_name: str,
    usage: str,
    properties: dict[str, str] | None = None,
) -> bool:
    """Create a vertex type, attach its usage instruction, and add its properties (idempotent).

    Returns True if the type was newly created. ``type_name``/property names are interpolated
    (DDL cannot bind identifiers); callers MUST pass values already validated by the Pydantic
    models in ``backend.schemas.graph_schemas``. ``usage`` is stored as type-level CUSTOM
    metadata (key ``description``) so future runs can read it via ``schema:types``.
    """
    newly_created = not await vertex_type_exists(db, type_name)
    # `IF NOT EXISTS` is a SUFFIX for types/properties (ArcadeDB quirk; see ensure_schema).
    await db.command(f"CREATE VERTEX TYPE {type_name} IF NOT EXISTS")
    # Type-level documentation. CUSTOM *values* are string literals, so they bind as parameters.
    await db.command(f"ALTER TYPE {type_name} CUSTOM description = :usage", {"usage": usage})
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

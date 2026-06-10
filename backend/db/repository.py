"""Persistence layer for conversations, messages, facts and logs.

This is the single source of truth for all database access: both the agent
tools (in ``backend.skills.graph_capability``) and the automatic persistence
hooks call into these functions. All statements are parameterized ArcadeDB SQL
and scoped by ``user_id`` so each user's memory stays isolated.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.db.arcade_db import ArcadeClient


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


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
    """Substring search across this user's stored facts."""
    return await db.query(
        "SELECT text, created_at FROM Fact "
        "WHERE user_id = :uid AND text LIKE :pat ORDER BY created_at DESC LIMIT :limit",
        {"uid": user_id, "pat": f"%{text}%", "limit": limit},
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

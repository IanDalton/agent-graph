"""Agent dependencies injected via Pydantic AI's ``deps_type``."""

from __future__ import annotations

from dataclasses import dataclass

from backend.db.arcade_db import ArcadeClient


@dataclass
class GraphDependencies:
    """Per-run dependencies for the conversation-memory agent.

    ``user_id`` isolates each user's memory; ``conversation_id`` scopes the
    current thread of messages and logs.
    """

    db: ArcadeClient
    user_id: str
    conversation_id: str

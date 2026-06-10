"""Async HTTP client for ArcadeDB.

Wraps the ArcadeDB REST API (https://docs.arcadedb.com/#HTTP-API):

- writes / DDL  -> POST /api/v1/command/{database}
- idempotent reads -> POST /api/v1/query/{database}

Both endpoints accept ``{"language": ..., "command": ..., "params": {...}}`` and
authenticate with HTTP Basic auth. Connection settings come from the environment
with the docker-compose defaults baked in, so a fresh checkout works out of the box.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

# Defaults mirror docker-compose.yml (arcadedb service).
DEFAULT_URL = "http://localhost:2480"
DEFAULT_DATABASE = "AgentMemory"
# The server root user; required for schema (DDL) operations. The per-database
# `admin` user created by docker-compose cannot alter the schema.
DEFAULT_USER = "root"
DEFAULT_PASSWORD = "playwithdata"


class ArcadeClient:
    """Thin async wrapper around the ArcadeDB HTTP API.

    Reuses a single :class:`httpx.AsyncClient`. Call :meth:`aclose` on shutdown,
    or use the client as an async context manager.
    """

    def __init__(
        self,
        url: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.url = (url or os.getenv("ARCADE_URL", DEFAULT_URL)).rstrip("/")
        self.database = database or os.getenv("ARCADE_DATABASE", DEFAULT_DATABASE)
        user = user or os.getenv("ARCADE_USER", DEFAULT_USER)
        password = password or os.getenv("ARCADE_PASSWORD", DEFAULT_PASSWORD)
        self._client = httpx.AsyncClient(
            base_url=self.url,
            auth=(user, password),
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    async def __aenter__(self) -> "ArcadeClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, endpoint: str, sql: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"language": "sql", "command": sql}
        if params:
            body["params"] = params
        resp = await self._client.post(f"/api/v1/{endpoint}/{self.database}", json=body)
        resp.raise_for_status()
        # ArcadeDB returns {"result": [...]} for both command and query.
        return resp.json().get("result", [])

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a write/DDL statement against the configured database."""
        return await self._post("command", sql, params)

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run an idempotent (read-only) query. The endpoint itself rejects mutations."""
        return await self._post("query", sql, params)

    async def ensure_schema(self) -> None:
        """Create the vertex/edge types and indexes the agent relies on.

        Idempotent: every statement uses ``IF NOT EXISTS`` so it is safe to call
        on every startup.
        """
        statements = [
            # Vertex types (`IF NOT EXISTS` is a suffix in ArcadeDB SQL).
            "CREATE VERTEX TYPE User IF NOT EXISTS",
            "CREATE VERTEX TYPE Conversation IF NOT EXISTS",
            "CREATE VERTEX TYPE Message IF NOT EXISTS",
            "CREATE VERTEX TYPE Fact IF NOT EXISTS",
            "CREATE VERTEX TYPE LogEntry IF NOT EXISTS",
            # Edge types
            "CREATE EDGE TYPE HAS_CONVERSATION IF NOT EXISTS",
            "CREATE EDGE TYPE HAS_MESSAGE IF NOT EXISTS",
            "CREATE EDGE TYPE KNOWS IF NOT EXISTS",
            "CREATE EDGE TYPE LOGGED IF NOT EXISTS",
            # Key properties + unique indexes (enable lookups and uniqueness).
            "CREATE PROPERTY User.user_id IF NOT EXISTS STRING",
            "CREATE INDEX IF NOT EXISTS ON User (user_id) UNIQUE",
            "CREATE PROPERTY Conversation.conversation_id IF NOT EXISTS STRING",
            "CREATE INDEX IF NOT EXISTS ON Conversation (conversation_id) UNIQUE",
            "CREATE PROPERTY Message.message_id IF NOT EXISTS STRING",
            "CREATE INDEX IF NOT EXISTS ON Message (message_id) UNIQUE",
            "CREATE PROPERTY Fact.fact_id IF NOT EXISTS STRING",
            "CREATE INDEX IF NOT EXISTS ON Fact (fact_id) UNIQUE",
            "CREATE PROPERTY LogEntry.log_id IF NOT EXISTS STRING",
            "CREATE INDEX IF NOT EXISTS ON LogEntry (log_id) UNIQUE",
            # Non-unique lookup indexes used by the repository queries.
            "CREATE PROPERTY Message.conversation_id IF NOT EXISTS STRING",
            "CREATE INDEX IF NOT EXISTS ON Message (conversation_id) NOTUNIQUE",
            "CREATE PROPERTY Message.user_id IF NOT EXISTS STRING",
            "CREATE INDEX IF NOT EXISTS ON Message (user_id) NOTUNIQUE",
            "CREATE PROPERTY Fact.user_id IF NOT EXISTS STRING",
            "CREATE INDEX IF NOT EXISTS ON Fact (user_id) NOTUNIQUE",
        ]
        for stmt in statements:
            await self.command(stmt)

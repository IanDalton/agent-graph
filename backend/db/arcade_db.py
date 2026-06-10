"""Async HTTP client for ArcadeDB.

Wraps the ArcadeDB REST API (https://docs.arcadedb.com/#HTTP-API):

- writes / DDL  -> POST /api/v1/command/{database}
- idempotent reads -> POST /api/v1/query/{database}

Both endpoints accept ``{"language": ..., "command": ..., "params": {...}}`` and
authenticate with HTTP Basic auth. Connection settings come from the environment
with the docker-compose defaults baked in, so a fresh checkout works out of the box.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

# Defaults mirror docker-compose.yml (arcadedb service).
DEFAULT_URL = "http://localhost:2480"
# Treated as a *base/prefix*: each user gets their own database named
# ``{DEFAULT_DATABASE}_{sanitized_user_id}_{hash}`` (see ``database_name_for_user``).
DEFAULT_DATABASE = "AgentMemory"
# The server root user; required for schema (DDL) operations and for creating
# databases. The per-database `admin` user from docker-compose cannot do either.
DEFAULT_USER = "root"
DEFAULT_PASSWORD = "playwithdata"

# Characters ArcadeDB database names safely allow.
_SAFE_DB_CHARS = re.compile(r"[^A-Za-z0-9_]")


def database_name_for_user(user_id: str, base: str | None = None) -> str:
    """Return the per-user database name, e.g. ``AgentMemory_u1_3f2a1b9c``.

    Each user is isolated in their own ArcadeDB database so one user's data can
    never appear in another user's queries. The raw ``user_id`` is sanitized to
    the characters ArcadeDB allows, then a short hash of the *original* id is
    appended so two ids that differ only in stripped characters (e.g. ``a.b`` vs
    ``a-b``) still map to distinct databases.
    """
    base = base or os.getenv("ARCADE_DATABASE", DEFAULT_DATABASE)
    sanitized = _SAFE_DB_CHARS.sub("_", user_id).strip("_") or "user"
    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{sanitized}_{digest}"


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
        max_retries: int = 5,
        retry_base_delay: float = 0.1,
    ) -> None:
        self.url = (url or os.getenv("ARCADE_URL", DEFAULT_URL)).rstrip("/")
        self.database = database or os.getenv("ARCADE_DATABASE", DEFAULT_DATABASE)
        user = user or os.getenv("ARCADE_USER", DEFAULT_USER)
        password = password or os.getenv("ARCADE_PASSWORD", DEFAULT_PASSWORD)
        # ArcadeDB answers 503 when momentarily overloaded (e.g. a burst of fact
        # writes in one turn) or while a freshly created database is opening.
        # These are transient, so retry the request with exponential backoff.
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
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
        resp = await self._request_with_retry(f"/api/v1/{endpoint}/{self.database}", body)
        # ArcadeDB returns {"result": [...]} for both command and query.
        return resp.json().get("result", [])

    async def _request_with_retry(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """POST ``body`` to ``path``, retrying transient 503s with exponential backoff.

        503 is the only status ArcadeDB uses for "try again shortly" (overloaded,
        or a database still opening); every other error is raised immediately.
        """
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(self.max_retries + 1):
            resp = await self._client.post(path, json=body)
            if resp.status_code != 503:
                resp.raise_for_status()
                return resp
            last_exc = httpx.HTTPStatusError(
                f"503 Service Unavailable for {path}", request=resp.request, response=resp
            )
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
        assert last_exc is not None  # loop ran at least once
        raise last_exc

    async def _server_command(self, command: str) -> dict[str, Any]:
        """Run a server-level command (``create database``, ``drop database``, ...).

        Hits ``POST /api/v1/server`` which requires the server root user.
        """
        resp = await self._client.post("/api/v1/server", json={"command": command})
        resp.raise_for_status()
        return resp.json()

    async def database_exists(self) -> bool:
        """True if the configured database already exists on the server."""
        resp = await self._client.get(f"/api/v1/exists/{self.database}")
        resp.raise_for_status()
        return bool(resp.json().get("result", False))

    async def ensure_database(self) -> None:
        """Create the configured database if it does not yet exist. Idempotent.

        Per-user isolation: each user's :class:`ArcadeClient` points at its own
        database, created on first use. Requires the server root user.
        """
        if not await self.database_exists():
            await self._server_command(f"create database {self.database}")

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

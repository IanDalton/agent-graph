"""ConversationMemory capability + automatic-persistence hooks.

Two pieces, both dropped into ``Agent(capabilities=...)`` via :func:`build_memory`:

1. ``memory_capability`` — the agent-facing tools (structured retrieval/storage
   plus a read-only raw-query escape hatch).
2. ``persistence_hooks`` — lifecycle hooks that persist every turn, tool call and
   error to ArcadeDB automatically, independent of what the model decides to do.

All DB access goes through :mod:`backend.db.repository`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable
from typing import Any

import httpx
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.capabilities import Capability, Hooks
from pydantic_ai.messages import TextPart, ToolCallPart, UserPromptPart

from backend import summarization
from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.schemas.graph_schemas import (
    MemoryHit,
    MemorySearchResult,
    RawQuery,
    StoreFactArgs,
)

logger = logging.getLogger("agent_graph.persistence")

# Statements the read-only raw-query tool is allowed to start with.
_READ_ONLY_PREFIXES = ("SELECT", "MATCH", "TRAVERSE")


async def _best_effort(awaitable: Awaitable[Any], *, what: str) -> None:
    """Run a persistence write that must never crash the agent.

    Memory persistence (conversation/message/log writes) is a side effect of a run, not the run's
    purpose. If ArcadeDB is briefly unavailable (e.g. a 503 that outlasts the client's retries), we
    log the failure and continue rather than aborting the whole agent loop.
    """
    try:
        await awaitable
    except Exception:  # noqa: BLE001 — deliberately swallow; persistence is best-effort.
        logger.warning("persistence step %r failed; continuing without it", what, exc_info=True)

INSTRUCTIONS = (
    "You have access to the user's persistent memory in ArcadeDB. "
    "CRITICAL RULE: before storing new data or creating a node, first query the "
    "database (use search_memory, get_conversation_history, or a read-only run_query) "
    "to check whether the relevant information or schema already exists. "
    "Use store_fact only for durable facts worth remembering across conversations. "
    "AVOID DUPLICATES: if a fact already exists but is now wrong or incomplete, do NOT store a second "
    "one — call update_fact with that fact's fact_id (returned by search_memory) to revise it in "
    "place, or delete_fact to remove a redundant/obsolete one."
)


def is_read_only(query: str) -> bool:
    """True if ``query`` is a read-only statement (starts with SELECT/MATCH/TRAVERSE)."""
    stripped = query.strip()
    if not stripped:
        return False
    return stripped.split(None, 1)[0].upper() in _READ_ONLY_PREFIXES


def _arcade_error_detail(exc: httpx.HTTPStatusError) -> str:
    """Pull ArcadeDB's human-readable ``detail`` out of an error response, falling back to the text.

    ArcadeDB returns ``{"error": ..., "detail": ..., "exception": ...}`` on a failed statement;
    e.g. querying a type that doesn't exist yet gives detail "Type with name 'X' was not found".
    """
    try:
        body = exc.response.json()
        return body.get("detail") or body.get("error") or exc.response.text
    except (json.JSONDecodeError, ValueError):
        return exc.response.text or str(exc)


def _text(content: Any) -> str:
    """Coerce a message part's content (str or multimodal sequence) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = [c for c in content if isinstance(c, str)]
        return " ".join(parts) if parts else str(content)
    return str(content)


# --------------------------------------------------------------------------- #
# Agent-facing tools
# --------------------------------------------------------------------------- #
memory_capability = Capability(id="ConversationMemory", instructions=INSTRUCTIONS)


@memory_capability.tool
async def search_memory(ctx: RunContext[GraphDependencies], query: str) -> MemorySearchResult:
    """Search the user's past messages and stored facts for relevant context."""
    deps = ctx.deps
    messages = await repo.search_messages(deps.db, deps.user_id, query)
    facts = await repo.search_facts(deps.db, deps.user_id, query)
    hits = [
        MemoryHit(kind="message", content=m.get("content", ""), created_at=m.get("created_at"))
        for m in messages
    ] + [
        MemoryHit(
            kind="fact",
            content=f.get("text", ""),
            created_at=f.get("created_at"),
            fact_id=f.get("fact_id"),
        )
        for f in facts
    ]
    return MemorySearchResult(hits=hits)


@memory_capability.tool
async def get_conversation_history(ctx: RunContext[GraphDependencies], limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent messages of the current conversation in chronological order."""
    return await repo.get_recent_messages(ctx.deps.db, ctx.deps.conversation_id, limit)


@memory_capability.tool
async def store_fact(ctx: RunContext[GraphDependencies], args: StoreFactArgs) -> str:
    """Remember a durable fact about the user for use in future conversations.

    Search first: if a related fact already exists, prefer update_fact over storing a duplicate.
    """
    await repo.store_fact(ctx.deps.db, ctx.deps.user_id, args.text)
    return f"Stored fact for user {ctx.deps.user_id}."


@memory_capability.tool
async def update_fact(ctx: RunContext[GraphDependencies], fact_id: str, text: str) -> str:
    """Revise an existing fact in place (use its fact_id from search_memory) instead of duplicating it."""
    updated = await repo.update_fact(ctx.deps.db, ctx.deps.user_id, fact_id, text)
    if not updated:
        raise ModelRetry(
            f"No fact with id {fact_id!r} for this user. Use search_memory to find the correct fact_id."
        )
    return f"Updated fact {fact_id}."


@memory_capability.tool
async def delete_fact(ctx: RunContext[GraphDependencies], fact_id: str) -> str:
    """Delete a redundant or obsolete fact by its fact_id (from search_memory)."""
    deleted = await repo.delete_fact(ctx.deps.db, ctx.deps.user_id, fact_id)
    if not deleted:
        raise ModelRetry(
            f"No fact with id {fact_id!r} for this user. Use search_memory to find the correct fact_id."
        )
    return f"Deleted fact {fact_id}."


@memory_capability.tool
async def run_query(ctx: RunContext[GraphDependencies], query_data: RawQuery) -> list[dict[str, Any]]:
    """Run an ad-hoc READ-ONLY ArcadeDB SQL query (SELECT/MATCH/TRAVERSE only)."""
    if not is_read_only(query_data.query):
        raise ValueError(
            f"Only read-only queries are permitted (must start with one of {_READ_ONLY_PREFIXES})."
        )
    # The query endpoint is idempotent and rejects mutations at the server, too.
    try:
        return await ctx.deps.db.query(query_data.query)
    except httpx.HTTPStatusError as exc:
        # A 503 is a genuine transient outage (already retried by the client) — let it surface.
        if exc.response.status_code == 503:
            raise
        # Any other status is a query-level problem — most commonly a SELECT against a type that
        # doesn't exist yet, which ArcadeDB answers with 500 + a SchemaException rather than an empty
        # result. This is NOT a model mistake to retry: querying a not-yet-created type is the normal
        # "check before create" path, and a legitimate negative result is "there are no such records".
        # Return that as an ordinary result (NOT ModelRetry — run_query's max_retries is 1, so a second
        # such error would raise UnexpectedModelBehavior and crash the run). The escape hatch must never
        # be able to abort the run; the model reads the note and adapts (create the type, fix the query).
        detail = _arcade_error_detail(exc)
        logger.info("run_query non-fatal error (returning as result): %s", detail)
        return [
            {
                "result": "no_records",
                "detail": f"Query did not run: {detail}",
                "hint": (
                    "A type/class that does not exist yet has no records. Call list_vertex_types to "
                    "see what exists; create the type (propose_schema_change → create_vertex_type, or "
                    "propose_edge_type → create_edge_type for relationships) before creating instances."
                ),
            }
        ]


# --------------------------------------------------------------------------- #
# Automatic persistence hooks
# --------------------------------------------------------------------------- #
persistence_hooks = Hooks()


@persistence_hooks.on.before_run
async def _ensure_conversation(ctx: RunContext[GraphDependencies]) -> None:
    deps = ctx.deps
    await _best_effort(
        repo.create_conversation(deps.db, deps.user_id, deps.conversation_id),
        what="create_conversation",
    )


@persistence_hooks.on.after_run
async def _persist_turn(ctx: RunContext[GraphDependencies], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
    deps = ctx.deps
    # Faithful replay record: the run's full serialized messages (tool calls + returns included),
    # so the next turn sees what the agent actually *did*, not just its text claims. main.run()
    # reloads these via repo.get_run_history; without it the model re-doubts completed tool work.
    await _best_effort(
        repo.append_run_messages(deps.db, deps.conversation_id, result.new_messages_json().decode()),
        what="append_run_messages",
    )
    # Human-readable, searchable mirror: role/content text for search_memory / get_conversation_history.
    for message in result.new_messages():
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                await _best_effort(
                    repo.append_message(deps.db, deps.user_id, deps.conversation_id, "user", _text(part.content)),
                    what="append_user_message",
                )
            elif isinstance(part, TextPart):
                await _best_effort(
                    repo.append_message(deps.db, deps.user_id, deps.conversation_id, "assistant", _text(part.content)),
                    what="append_assistant_message",
                )
    # Refresh the cached conversation summary, but only once every N messages (handled inside).
    # The response has already streamed to the user; this runs after, and is best-effort.
    await _best_effort(
        summarization.maybe_refresh_summary(deps.db, deps.conversation_id),
        what="refresh_summary",
    )
    return result


@persistence_hooks.on.after_tool_execute
async def _log_tool_call(
    ctx: RunContext[GraphDependencies],
    *,
    call: ToolCallPart,
    tool_def: Any,
    args: Any,
    result: Any,
) -> Any:
    logger.debug("tool %s executed (args=%s)", call.tool_name, call.args)
    await _best_effort(
        repo.write_log(
            ctx.deps.db,
            ctx.deps.conversation_id,
            level="info",
            event="tool_execute",
            payload={"tool": call.tool_name, "args": call.args, "result": str(result)[:2000]},
        ),
        what="write_log:tool_execute",
    )
    return result


@persistence_hooks.on.run_error
async def _log_error(ctx: RunContext[GraphDependencies], *, error: BaseException) -> AgentRunResult[Any]:
    # Always surface the run error on the application logger (with traceback), independent of
    # whether the DB write below succeeds.
    logger.error("agent run error: %s: %s", type(error).__name__, error, exc_info=error)
    await _best_effort(
        repo.write_log(
            ctx.deps.db,
            ctx.deps.conversation_id,
            level="error",
            event="run_error",
            payload={"type": type(error).__name__, "message": str(error)},
        ),
        what="write_log:run_error",
    )
    raise error


def build_memory() -> list[Capability | Hooks]:
    """Return the capabilities to add to ``Agent(capabilities=...)``.

    The database connection is supplied per-run through ``GraphDependencies``
    (``ctx.deps.db``), so nothing needs to be wired in here.
    """
    return [memory_capability, persistence_hooks]

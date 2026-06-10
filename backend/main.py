"""Entry point: wire the ArcadeDB-backed conversation-memory agent together.

Run a single turn from the command line:

    python -m backend.main "remember I like Recoleta apartments" --user u1 --conversation c1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
)

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient, database_name_for_user
from backend.db.dependencies import GraphDependencies
from backend.model_selection import select_model
from backend.skills.graph_capability import build_memory
from backend.skills.ontology_capability import build_ontology
from backend.skills.search_capability import build_search
from backend.web.client import WebClient

logger = logging.getLogger("agent_graph.main")


def build_agent() -> Agent[GraphDependencies, str]:
    """Construct the agent.

    Model selection via env: set ``AGENT_MODEL`` to any Pydantic AI model string
    (e.g. ``openai:gpt-5.2``). If unset, a local Ollama model named by
    ``OLLAMA_MODEL`` is used (mirrors the original notebook prototype).
    """
    return Agent(
        select_model("AGENT_MODEL"),
        deps_type=GraphDependencies,
        capabilities=[
            Thinking(effort="minimal"),
            *build_memory(),
            *build_ontology(),
            *build_search(),
        ],
    )


def _to_message_history(rows: list[dict[str, Any]]) -> list[ModelMessage]:
    """Rebuild Pydantic AI message history from stored per-run serialized blobs.

    Each row's ``raw`` is one run's ``new_messages_json()`` (see
    :func:`repo.append_run_messages`); deserializing and concatenating them oldest-first
    reconstructs the conversation *faithfully* — tool calls and their returns included — so the
    model sees what it actually did, not just its own text. This is what stops the agent
    re-doubting/redoing tool work it already completed. ``instructions`` are re-applied each run
    regardless of history. A corrupt blob is skipped rather than failing the whole run.
    """
    history: list[ModelMessage] = []
    for row in rows:
        raw = row.get("raw")
        if not raw:
            continue
        try:
            history.extend(ModelMessagesTypeAdapter.validate_json(raw))
        except Exception:  # noqa: BLE001 — a single bad blob must not blind the agent to the rest.
            logger.warning("skipping unparseable run-message blob", exc_info=True)
    return history


def _jsonable(value: Any) -> Any:
    """Coerce arbitrary tool args/results into something JSON-serializable for an SSE frame.

    Tool args arrive as a dict or a raw JSON string; tool results can be any Python value.
    Anything that isn't a plain JSON scalar/container is rendered as its ``str``.
    """
    if value is None or isinstance(value, (str, int, float, bool, dict, list)):
        return value
    return str(value)


async def stream_run(
    prompt: str, user_id: str = "default", conversation_id: str = "default"
) -> AsyncIterator[dict[str, Any]]:
    """Run one turn and yield structured events as they stream from the model.

    This is the single streaming source of truth, consumed by both the CLI (``run``) and the
    HTTP/SSE API (``backend.api``). Pydantic AI's event objects are mapped to a small, stable
    vocabulary so callers never depend on the library's event classes:

    - ``{"type": "thinking", "delta": str}`` — a chunk of the model's reasoning
    - ``{"type": "text", "delta": str}`` — a chunk of the user-facing answer
    - ``{"type": "tool_call", "tool_name", "tool_call_id", "args"}`` — a tool invocation
    - ``{"type": "tool_result", "tool_name", "tool_call_id", "content"}`` — its return
    - ``{"type": "final", "text": str}`` — the fully assembled answer (always emitted last)
    """
    agent = build_agent()
    # Each user gets their own database, so no user's data can surface in another
    # user's queries. The database is created on first use.
    async with ArcadeClient(database=database_name_for_user(user_id)) as db, WebClient() as web:
        await db.ensure_database()
        await db.ensure_schema()
        deps = GraphDependencies(
            db=db, user_id=user_id, conversation_id=conversation_id, web=web
        )

        # Load the prior turns of this thread so the agent retains context. The
        # persistence hooks store each run but never reload it, so without this
        # the model starts every run blind to the conversation. We load the faithful
        # per-run blobs (tool calls + returns), not the lossy role/content text, so the
        # agent sees the tool work it already did. after_run persists only the current
        # run's new_messages(), so reloaded history isn't re-written.
        history = _to_message_history(await repo.get_run_history(db, conversation_id))

        final_text = ""
        async with agent.run_stream_events(prompt, deps=deps, message_history=history) as stream:
            async for event in stream:
                if isinstance(event, FunctionToolCallEvent):
                    yield {
                        "type": "tool_call",
                        "tool_name": event.part.tool_name,
                        "tool_call_id": event.part.tool_call_id,
                        "args": _jsonable(event.part.args),
                    }
                    continue
                if isinstance(event, FunctionToolResultEvent):
                    part = event.part
                    yield {
                        "type": "tool_result",
                        "tool_name": getattr(part, "tool_name", None),
                        "tool_call_id": getattr(part, "tool_call_id", None),
                        "content": _jsonable(getattr(part, "content", None)),
                    }
                    continue

                node = event
                if isinstance(event, PartStartEvent):
                    node = event.part
                elif isinstance(event, PartDeltaEvent):
                    node = event.delta

                if isinstance(node, ThinkingPartDelta):
                    if node.content_delta:
                        yield {"type": "thinking", "delta": node.content_delta}
                elif isinstance(node, TextPart):
                    final_text += node.content
                    yield {"type": "text", "delta": node.content}
                elif isinstance(node, TextPartDelta):
                    final_text += node.content_delta
                    yield {"type": "text", "delta": node.content_delta}

        yield {"type": "final", "text": final_text}


async def run(prompt: str, user_id: str = "default", conversation_id: str = "default") -> str:
    """Run one turn, streaming thinking/text to stdout, and return the final output."""
    final_text = ""
    async for event in stream_run(prompt, user_id=user_id, conversation_id=conversation_id):
        kind = event["type"]
        if kind == "thinking":
            print(f"\033[94m{event['delta']}\033[0m", end="", flush=True)
        elif kind == "text":
            print(event["delta"], end="", flush=True)
        elif kind == "final":
            final_text = event["text"]
    print()
    return final_text


def _configure_logging() -> None:
    """Send application logs to stderr. Level via ``LOG_LEVEL`` (default INFO).

    Persistence failures, DB retries and run errors are emitted under the ``agent_graph.*``
    loggers, so a database hiccup shows up as a clear log line instead of a raw traceback.
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Run one turn of the conversation-memory agent.")
    parser.add_argument("prompt")
    parser.add_argument("--user", default="default")
    parser.add_argument("--conversation", default="default")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.prompt, user_id=args.user, conversation_id=args.conversation))
    except Exception as exc:  # noqa: BLE001 — CLI boundary: log cleanly instead of dumping a traceback.
        logging.getLogger("agent_graph").error("Agent run failed: %s: %s", type(exc).__name__, exc)
        logging.getLogger("agent_graph").debug("Full traceback:", exc_info=exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

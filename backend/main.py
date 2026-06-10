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
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import (
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
from backend.skills.graph_capability import build_memory
from backend.skills.ontology_capability import build_ontology

logger = logging.getLogger("agent_graph.main")


def build_agent() -> Agent[GraphDependencies, str]:
    """Construct the agent.

    Model selection via env: set ``AGENT_MODEL`` to any Pydantic AI model string
    (e.g. ``openai:gpt-5.2``). If unset, a local Ollama model named by
    ``OLLAMA_MODEL`` is used (mirrors the original notebook prototype).
    """
    model_string = os.getenv("AGENT_MODEL")
    if model_string:
        model: Agent | str = model_string
    else:
        from pydantic_ai.models.ollama import OllamaModel

        model = OllamaModel(os.getenv("OLLAMA_MODEL", "qwen3"))

    return Agent(
        model,
        deps_type=GraphDependencies,
        capabilities=[Thinking(effort="minimal"), *build_memory(), *build_ontology()],
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


async def run(prompt: str, user_id: str = "default", conversation_id: str = "default") -> str:
    """Run one turn, streaming thinking/text to stdout, and return the final output."""
    agent = build_agent()
    # Each user gets their own database, so no user's data can surface in another
    # user's queries. The database is created on first use.
    async with ArcadeClient(database=database_name_for_user(user_id)) as db:
        await db.ensure_database()
        await db.ensure_schema()
        deps = GraphDependencies(db=db, user_id=user_id, conversation_id=conversation_id)

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
                if isinstance(event, PartStartEvent):
                    event = event.part
                elif isinstance(event, PartDeltaEvent):
                    event = event.delta
                if isinstance(event, ThinkingPartDelta):
                    print(f"\033[94m{event.content_delta}\033[0m", end="", flush=True)
                elif isinstance(event, TextPart):
                    final_text += event.content
                    print(event.content, end="", flush=True)
                elif isinstance(event, TextPartDelta):
                    final_text += event.content_delta
                    print(event.content_delta, end="", flush=True)
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

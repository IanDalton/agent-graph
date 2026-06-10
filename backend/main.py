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
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    UserPromptPart,
)

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient, database_name_for_user
from backend.db.dependencies import GraphDependencies
from backend.skills.graph_capability import build_memory
from backend.skills.ontology_capability import build_ontology


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
    """Rebuild Pydantic AI message history from stored ``role``/``content`` rows.

    Each prior user turn becomes a ``ModelRequest`` and each assistant turn a
    ``ModelResponse`` so the model sees the conversation it already had. Passing
    these as ``message_history`` is what makes the agent retain context across
    runs; ``instructions`` are always re-applied regardless of history.
    """
    history: list[ModelMessage] = []
    for row in rows:
        content = row.get("content", "")
        if row.get("role") == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif row.get("role") == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=content)]))
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
        # persistence hooks store each turn but never reload it, so without this
        # the model starts every run blind to the conversation. after_run persists
        # only the current run's new_messages(), so reloaded history isn't re-written.
        history = _to_message_history(await repo.get_recent_messages(db, conversation_id))

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

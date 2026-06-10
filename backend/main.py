"""Entry point: wire the ArcadeDB-backed conversation-memory agent together.

Run a single turn from the command line:

    python -m backend.main "remember I like Recoleta apartments" --user u1 --conversation c1
"""

from __future__ import annotations

import argparse
import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
)

from backend.db.arcade_db import ArcadeClient
from backend.db.dependencies import GraphDependencies
from backend.skills.graph_capability import build_memory


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
        capabilities=[Thinking(effort="minimal"), *build_memory()],
    )


async def run(prompt: str, user_id: str = "default", conversation_id: str = "default") -> str:
    """Run one turn, streaming thinking/text to stdout, and return the final output."""
    agent = build_agent()
    async with ArcadeClient() as db:
        await db.ensure_schema()
        deps = GraphDependencies(db=db, user_id=user_id, conversation_id=conversation_id)

        final_text = ""
        async with agent.run_stream_events(prompt, deps=deps) as stream:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one turn of the conversation-memory agent.")
    parser.add_argument("prompt")
    parser.add_argument("--user", default="default")
    parser.add_argument("--conversation", default="default")
    args = parser.parse_args()
    asyncio.run(run(args.prompt, user_id=args.user, conversation_id=args.conversation))


if __name__ == "__main__":
    main()

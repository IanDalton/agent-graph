"""Automatic conversation-title generation.

The title is generated lazily after the first completed turn when a Conversation has no title yet.
The model is env-driven so the local title model can be swapped without code changes:

- ``TITLE_MODEL`` - any explicit Pydantic AI model string (for example ``openai:gpt-5.2``).
- ``TITLE_OLLAMA_MODEL`` - local Ollama model name. Defaults to ``supra-title-350m-exp``.

When ``TITLE_MODEL`` is unset, the title generator uses Ollama with the local model name above.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient
from backend.model_selection import select_model

load_dotenv()

logger = logging.getLogger("agent_graph.titles")

_TITLE_INSTRUCTIONS = (
    "Write a short, specific chat title in 3-6 words. "
    "Use title case, avoid quotation marks, avoid a trailing period, and focus on the main topic. "
    "Return only the title text."
)

_TITLE_MODEL = "TITLE_MODEL"
_TITLE_OLLAMA_MODEL = "TITLE_OLLAMA_MODEL"
_TITLE_DEFAULT = "supra-title-350m-exp"


def title_model_label() -> str:
    """Return the configured title model as a UI-friendly label."""
    model = os.getenv(_TITLE_MODEL)
    return model or f"ollama/{os.getenv(_TITLE_OLLAMA_MODEL, _TITLE_DEFAULT)}"


def _normalize_title(raw: str) -> str:
    title = " ".join(raw.strip().split())
    title = title.strip("\"'`")
    if title.lower().startswith("title:"):
        title = title.split(":", 1)[1].strip()
    if len(title) > 72:
        title = title[:72].rstrip()
    return title


async def generate_title(db: ArcadeClient, conversation_id: str) -> str:
    """Generate and persist a title for ``conversation_id`` if it does not already have one."""
    existing = (await repo.get_conversation_title(db, conversation_id)).strip()
    if existing:
        return existing

    messages = await repo.get_recent_messages(db, conversation_id, limit=12)
    if not messages:
        return ""

    transcript = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    title_agent: Agent[None, str] = Agent(
        select_model(_TITLE_MODEL, _TITLE_OLLAMA_MODEL, _TITLE_DEFAULT),
        instructions=_TITLE_INSTRUCTIONS,
    )
    result = await title_agent.run(transcript)
    title = _normalize_title(result.output)
    if not title:
        return ""

    await repo.set_conversation_title(db, conversation_id, title)
    logger.debug("generated title for %s: %s", conversation_id, title)
    return title


async def maybe_refresh_title(db: ArcadeClient, conversation_id: str) -> None:
    """Best-effort title generation that only fills blank titles."""
    await generate_title(db, conversation_id)

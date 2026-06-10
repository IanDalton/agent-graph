"""Conversation summary generation, gated on a message-count threshold.

The right-pane "Summary" is a derived artifact: a short LLM digest of the conversation. Generating
it on every tab load (the old ``GET /summary`` behaviour) paid a 1-10s model round-trip for a
summary that rarely changed. Instead we regenerate it *at write time* — only once every
``SUMMARY_EVERY_N_MESSAGES`` messages — and store it on the ``Conversation`` vertex, so the endpoint
that serves it is a fast DB read.

This is a dependency-free leaf module (like ``backend.model_selection``): ``backend.main`` imports
``backend.skills.graph_capability``, whose ``after_run`` hook calls in here, so it must not import
``main`` (or it would create a cycle).
"""

from __future__ import annotations

import logging
import os

from pydantic_ai import Agent

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient
from backend.model_selection import select_model

logger = logging.getLogger("agent_graph.summarization")

# How many new messages to accumulate before regenerating the summary (~3 turns at the default 6).
SUMMARY_EVERY_N_MESSAGES = int(os.getenv("SUMMARY_EVERY_N_MESSAGES", "6"))

_SUMMARY_INSTRUCTIONS = (
    "Summarize the following conversation in 2-3 short bullet points. "
    "Capture the topic and any decisions or facts established. "
    "Reply with only the bullets, no preamble."
)


async def generate_summary(db: ArcadeClient, conversation_id: str) -> str:
    """Run the LLM summarizer over the conversation and store the result. Returns the summary.

    Always regenerates (no threshold) — used by the force-refresh endpoint. Returns ``""`` if the
    conversation has no messages yet.
    """
    count = await repo.count_messages(db, conversation_id)
    if count <= 0:
        return ""
    messages = await repo.get_recent_messages(db, conversation_id, limit=30)
    if not messages:
        return ""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    summarizer: Agent[None, str] = Agent(
        select_model("AGENT_MODEL"),
        instructions=_SUMMARY_INSTRUCTIONS,
    )
    result = await summarizer.run(transcript)
    summary = result.output.strip()
    await repo.set_conversation_summary(db, conversation_id, summary, count)
    logger.debug("generated summary for %s at %d messages", conversation_id, count)
    return summary


async def maybe_refresh_summary(db: ArcadeClient, conversation_id: str) -> None:
    """Regenerate and store the conversation summary iff enough new messages have accumulated.

    Compares the current message count against the count recorded when the summary was last
    generated; only runs the (slow) LLM call once the delta reaches ``SUMMARY_EVERY_N_MESSAGES``.
    Intended to be invoked from the after-run persistence hook (wrapped in best-effort there), so a
    failure here never crashes the agent loop.
    """
    count = await repo.count_messages(db, conversation_id)
    if count <= 0:
        return
    meta = await repo.get_conversation_summary(db, conversation_id)
    if count - meta["summary_message_count"] < SUMMARY_EVERY_N_MESSAGES:
        return
    await generate_summary(db, conversation_id)

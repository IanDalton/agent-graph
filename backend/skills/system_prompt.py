"""Best-practices base system prompt + per-run injection of the user's relevant facts.

Unlike the capability bundles (which attach static, tool-scoped ``instructions`` strings), this
module owns the *agent-level* system prompt:

1. :data:`BASE_SYSTEM_PROMPT` — a fixed, provider-agnostic identity + behaviour prompt, attached
   to the main agent via ``Agent(instructions=...)``.
2. A **dynamic** ``@agent.instructions`` callable (registered by :func:`register_system_prompt`)
   that, each run, embeds the user's current prompt and injects the most relevant facts already
   stored in their graph memory — so a turn starts grounded in what we know about the user without
   waiting for the model to call ``search_memory``.

Both pieces apply to the **main agent only**; delegated sub-agents keep their task-specific
prompts. Fact loading is **best-effort** (the same contract as the persistence hooks and web
tools): any DB/embedder failure logs and degrades to no fact block — it must never abort a turn.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Iterable

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, UserPromptPart

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies

logger = logging.getLogger("agent_graph.system_prompt")

# How many of the most relevant stored facts to inject into the system prompt each turn. A small
# cap keeps the prompt focused; the model can always call search_memory for more.
_MAX_FACTS = 8
# Cap on the always-included "important" facts the user (or agent) curated. Bounds the prompt even
# when a user marks many facts important.
_MAX_IMPORTANT = 20

BASE_SYSTEM_PROMPT = (
    "You are a helpful, capable assistant backed by a persistent graph memory of THIS user. "
    "You remember durable facts about them across conversations, and you can search the web, run "
    "Python in a sandbox, and author documents.\n"
    "\n"
    "HOW TO WORK:\n"
    "- Be agentic: keep working until the user's request is fully resolved before ending your "
    "turn. Prefer using your tools to find the answer over guessing or asking the user for "
    "information you can retrieve yourself.\n"
    "- Use memory: relevant facts the user has told you before may be listed below. Treat them as "
    "context, not as the user's current message. Store genuinely durable new facts about the user "
    "with your memory tools, and revise an existing fact rather than storing a duplicate.\n"
    "- Reach for tools instead of speculating: search the web for current or external information, "
    "run code to compute or verify, and save substantial deliverables as documents.\n"
    "\n"
    "HONESTY:\n"
    "- Never fabricate facts, sources, citations, or tool results. If you are unsure, say so. If a "
    "tool fails or returns nothing, say that plainly instead of inventing an answer.\n"
    "\n"
    "STYLE:\n"
    "- Reply in clear, concise Markdown. Match the user's language and level of detail."
)


def _today() -> str:
    """Today's date as an ISO string, for date-awareness in the system prompt."""
    return date.today().isoformat()


def _latest_user_prompt(messages: Iterable[ModelMessage]) -> str:
    """Return the text of the most recent user prompt in ``messages`` (newest wins), else "".

    Mirrors the ``UserPromptPart`` handling in
    :mod:`backend.skills.graph_capability`; kept pure so it is unit-testable without a run.
    """
    latest = ""
    for message in messages:
        for part in getattr(message, "parts", ()):  # ModelResponse parts have no UserPromptPart
            if isinstance(part, UserPromptPart):
                latest = _content_text(part.content)
    return latest


def _content_text(content: Any) -> str:
    """Coerce a message part's content (str or multimodal sequence) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = [c for c in content if isinstance(c, str)]
        return " ".join(parts) if parts else str(content)
    return str(content)


async def relevant_facts_block(deps: GraphDependencies, query: str) -> str:
    """Build the 'known facts about the user' block, or "" when there's nothing.

    Hybrid selection: always include the facts the user (or agent) marked *important*, then fill any
    remaining room with the facts most semantically relevant to ``query``. Important facts come from
    :func:`repo.list_facts`; relevance from :func:`repo.search_facts` (which embeds the query when an
    embedder is configured and falls back to substring matching on its own). Deduped by ``fact_id``.
    Best-effort: any failure logs and returns "" so a turn is never blocked.
    """
    try:
        embedder = deps.embedder
        important = await repo.list_facts(
            deps.db, deps.user_id, limit=_MAX_IMPORTANT, important_only=True
        )
        relevant: list[dict] = []
        if query:
            embedding = await embedder.embed(query) if embedder is not None else None
            relevant = await repo.search_facts(
                deps.db, deps.user_id, query, limit=_MAX_FACTS, embedding=embedding
            )
    except Exception:  # noqa: BLE001 — fact recall is best-effort; never abort the run.
        logger.warning("relevant-facts lookup failed; continuing without it", exc_info=True)
        return ""
    # Important facts first, then semantically-relevant ones not already included.
    lines: list[str] = []
    seen: set[str] = set()
    for f in important + relevant:
        text = f.get("text")
        fact_id = f.get("fact_id")
        if not text or (fact_id and fact_id in seen):
            continue
        if fact_id:
            seen.add(fact_id)
        lines.append(f"- {text}")
    if not lines:
        return ""
    return "Known facts about the user (from memory — context, not their current message):\n" + "\n".join(lines)


async def user_profile_block(deps: GraphDependencies) -> str:
    """Build the durable 'user profile' block, or "" when there is none.

    The profile is the rolling cross-conversation synopsis maintained by the background memory
    curator (:mod:`backend.memory_curator`) and stored on the ``User`` vertex. Injecting it each turn
    grounds the agent in who the user is without waiting for it to search memory. Best-effort: any
    DB failure logs and returns "" so a turn is never blocked.
    """
    try:
        profile = (await repo.get_user_profile(deps.db, deps.user_id))["profile"]
    except Exception:  # noqa: BLE001 — profile recall is best-effort; never abort the run.
        logger.warning("user-profile lookup failed; continuing without it", exc_info=True)
        return ""
    if not profile:
        return ""
    return "What we know about this user (durable profile, curated from past conversations):\n" + profile


def register_system_prompt(agent: Agent[GraphDependencies, Any]) -> None:
    """Attach the dynamic per-run instructions (date + relevant user facts) to ``agent``.

    The static :data:`BASE_SYSTEM_PROMPT` is passed to the ``Agent`` constructor separately; this
    adds the parts that depend on the run (``ctx.deps`` / the current prompt).
    """

    @agent.instructions
    def _date(_ctx: RunContext[GraphDependencies]) -> str:
        return f"Today's date is {_today()}."

    @agent.instructions
    async def _user_profile(ctx: RunContext[GraphDependencies]) -> str:
        # The durable profile frames the more granular facts below it.
        return await user_profile_block(ctx.deps)

    @agent.instructions
    async def _user_facts(ctx: RunContext[GraphDependencies]) -> str:
        return await relevant_facts_block(ctx.deps, _latest_user_prompt(ctx.messages))

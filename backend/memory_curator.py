"""Background memory curator — keeps the user's long-term memory accurate between turns.

A normal turn relies on the *main* agent remembering to call ``store_fact``/``update_fact`` mid-task,
which it often skips or duplicates. This module adds a small dedicated agent that runs
automatically, gated on a message-count threshold (like :mod:`backend.summarization`), to:

1. Extract and dedupe durable **facts** about the user from the recent conversation.
2. Rewrite a persistent, cross-conversation **user profile** (stored on the ``User`` vertex via
   :func:`backend.db.repository.set_user_profile`) that the main agent's system prompt injects every
   turn — see :func:`backend.skills.system_prompt.user_profile_block`.
3. Optionally model durable **entities/relationships** in the knowledge graph (the ontology tools).

It reuses the existing fact/ontology tool capabilities so all dedupe + DDL logic stays in one place,
and runs **silently** (``agent.run``, not ``run_stream_events``) so its tool calls never appear in
the live UI stream. Like the persistence hooks, the trigger is best-effort: a failure here never
affects the user's turn.

This is a dependency-light leaf module: it imports the tool capabilities (``memory_capability``,
``build_ontology``) but must **not** import :mod:`backend.main` (that would create a cycle —
``main`` → ``graph_capability`` → this module via the after-run hook's deferred import).
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.usage import UsageLimits

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.model_selection import resolve_model
from backend.skills.graph_capability import memory_capability
from backend.skills.ontology_capability import build_ontology

logger = logging.getLogger("agent_graph.memory_curator")

# How many new messages to accumulate before running a curation pass (~4 turns at the default 8).
MEMORY_CURATION_EVERY_N_MESSAGES = int(os.getenv("MEMORY_CURATION_EVERY_N_MESSAGES", "8"))
# Runaway backstop on the curator's own agent loop (it may call several memory/ontology tools).
MEMORY_CURATOR_REQUEST_LIMIT = int(os.getenv("MEMORY_CURATOR_REQUEST_LIMIT", "12"))
# Kill switch — set MEMORY_CURATION_ENABLED=0 to disable background curation entirely.
MEMORY_CURATION_ENABLED = os.getenv("MEMORY_CURATION_ENABLED", "1") not in ("0", "false", "False")

# How many recent messages of the conversation to feed the curator, and how many stored facts to
# show it (with their fact_ids, so it can revise/delete in place rather than duplicate).
_TRANSCRIPT_LIMIT = 30
_FACTS_LIMIT = 200

CURATOR_INSTRUCTIONS = (
    "You are the background memory curator for a personal assistant with persistent graph memory of "
    "ONE user. You run periodically, on your own, to keep that user's long-term memory accurate and "
    "useful for future conversations. You are given the recent conversation, the facts already "
    "stored about the user (each with its fact_id), and the current user profile.\n"
    "\n"
    "Do the following, then stop:\n"
    "1. FACTS: store genuinely durable, future-useful facts about the user (identity, preferences, "
    "goals, ongoing projects, constraints, relationships) with store_fact. If a stored fact is now "
    "wrong, outdated, or redundant, call update_fact with its fact_id (or delete_fact) — NEVER "
    "create a duplicate. Ignore ephemeral, conversation-specific details (one-off questions, "
    "transient task state).\n"
    "2. PROFILE: call update_user_profile exactly once with a concise, well-organized synopsis of "
    "who the user is and what matters for helping them well — a few short sections or bullet groups, "
    "about 200 words or fewer. Integrate any new information and REPLACE the old profile in full "
    "(your text becomes the entire profile). If nothing meaningful changed, you may skip this.\n"
    "3. GRAPH (optional): for durable entities and the relationships between them, you may grow the "
    "knowledge graph. Always list_vertex_types first; reuse existing types/instances; follow the "
    "guarded flow propose_schema_change/propose_edge_type -> create_vertex_type/create_edge_type -> "
    "create_node/create_edge. Never duplicate an entity that already exists.\n"
    "\n"
    "Be conservative: only record what is clearly durable and supported by the conversation or the "
    "existing facts. Never invent anything. Reply with a one-line note of what you changed."
)


# --------------------------------------------------------------------------- #
# The user-profile tool (the curator's one extra capability beyond memory/ontology)
# --------------------------------------------------------------------------- #
profile_capability = Capability(
    id="UserProfile",
    instructions=(
        "update_user_profile replaces the user's durable profile in FULL with the text you pass — "
        "always send the complete rewritten profile, not a delta."
    ),
)


@profile_capability.tool
async def update_user_profile(ctx: RunContext[GraphDependencies], profile: str) -> str:
    """Replace the user's persistent profile with ``profile`` (the complete, rewritten synopsis)."""
    await repo.set_user_profile(ctx.deps.db, ctx.deps.user_id, profile.strip())
    return f"Updated user profile for {ctx.deps.user_id}."


def _facts_block(facts: list[dict]) -> str:
    """Render the stored facts (with fact_ids) for the curator's input, or a 'none yet' note."""
    if not facts:
        return "(no facts stored yet)"
    return "\n".join(f"- [{f.get('fact_id')}] {f.get('text', '')}" for f in facts)


def _build_prompt(transcript: str, facts: list[dict], profile: str) -> str:
    """Assemble the curator's run input from the transcript, existing facts and current profile."""
    return (
        "RECENT CONVERSATION:\n"
        f"{transcript or '(no messages)'}\n\n"
        "FACTS ALREADY STORED (id in brackets — use it to update_fact / delete_fact):\n"
        f"{_facts_block(facts)}\n\n"
        "CURRENT USER PROFILE:\n"
        f"{profile or '(empty)'}\n\n"
        "Curate the memory now per your instructions."
    )


async def curate_memory(deps: GraphDependencies) -> None:
    """Run one curation pass: extract/dedupe facts, rewrite the profile, optionally grow the graph.

    Builds a fresh single-purpose agent granted the memory + ontology tools (plus update_user_profile)
    — but **no persistence hooks** (a curator run must not be recorded as a conversation turn, nor
    recurse into curation). Runs on an isolated deps copy so the ontology propose->create ordering
    guard starts clean. Always advances the curation watermark so the threshold gate moves forward.
    """
    count = await repo.count_messages(deps.db, deps.conversation_id)
    messages = await repo.get_recent_messages(deps.db, deps.conversation_id, limit=_TRANSCRIPT_LIMIT)
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    facts = await repo.list_facts(deps.db, deps.user_id, limit=_FACTS_LIMIT)
    profile = (await repo.get_user_profile(deps.db, deps.user_id))["profile"]

    # Fresh proposal state so the ontology guard never collides with the main run's proposals.
    sub = replace(deps, proposed_schemas={}, proposed_edges={})
    agent: Agent[GraphDependencies, str] = Agent(
        resolve_model(deps.model),
        deps_type=GraphDependencies,
        instructions=CURATOR_INSTRUCTIONS,
        capabilities=[memory_capability, *build_ontology(), profile_capability],
    )
    # Silent run: agent.run (not run_stream_events) so curator tool-chips never reach deps.event_sink.
    await agent.run(
        _build_prompt(transcript, facts, profile),
        deps=sub,
        usage_limits=UsageLimits(request_limit=MEMORY_CURATOR_REQUEST_LIMIT),
    )
    await repo.set_memory_curation_watermark(deps.db, deps.conversation_id, count)
    logger.debug("curated memory for %s at %d messages", deps.conversation_id, count)


async def maybe_curate_memory(deps: GraphDependencies) -> None:
    """Run a curation pass iff enabled and enough new messages have accumulated since the last one.

    Intended to be invoked (best-effort) from the after-run persistence hook, so any failure here is
    swallowed and never crashes the agent loop. The (slow) curator agent loop only runs once every
    ``MEMORY_CURATION_EVERY_N_MESSAGES`` messages.
    """
    if not MEMORY_CURATION_ENABLED:
        return
    count = await repo.count_messages(deps.db, deps.conversation_id)
    if count <= 0:
        return
    watermark = await repo.get_memory_curation_watermark(deps.db, deps.conversation_id)
    if count - watermark < MEMORY_CURATION_EVERY_N_MESSAGES:
        return
    await curate_memory(deps)

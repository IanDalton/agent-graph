"""DeepResearch: the research-mode methodology and the deep-research sub-agent prompt.

Two consumers:

- **Research mode** (a conversation created with ``mode="research"``): :func:`build_research`
  returns an instructions-only capability layered onto the regular agent, turning it into a
  methodical researcher — plan, fan out searches, read sources, cross-check, then deliver a
  cited report as a document. No new tools: the work runs on the existing ``web_search`` /
  ``fetch_url`` / ``create_document`` tools, so every step streams to the UI as ordinary
  tool chips. A best-effort ``after_run`` safeguard persists the report itself if the model
  forgot to call ``create_document`` (common on smaller/local models).
- **The swarm's built-in ``deep_research`` tool**: :data:`DEEP_RESEARCH_INSTRUCTIONS` is the
  system prompt of the delegated researcher sub-agent (run via
  :func:`backend.skills.subagent.run_subagent` with the ``web`` + ``documents`` tool groups).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.capabilities import Capability, Hooks
from pydantic_ai.messages import ToolCallPart

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies

logger = logging.getLogger("agent_graph.research")

# Ceiling on the deep-research sub-agent's model round-trips. Research is the deepest delegated
# loop (many search/fetch cycles), so it gets a higher default than ordinary sub-agents.
DEEP_RESEARCH_REQUEST_LIMIT = int(os.getenv("DEEP_RESEARCH_REQUEST_LIMIT", "40"))

# Minimum length of an answer the research-mode safeguard will save as a report document — short
# clarifying replies ("which region?") are not reports and shouldn't spawn an artifact.
_MIN_REPORT_CHARS = 200

# The shared research method. Phrased agent-agnostically so it reads correctly both as a mode
# overlay on the main agent and inside the delegated researcher's system prompt.
_METHOD = (
    "RESEARCH METHOD — follow these phases in order, and do NOT cut them short:\n"
    "1. PLAN: break the question into 3-5 concrete sub-questions. State them briefly before "
    "searching.\n"
    "2. SEARCH WIDE: run at least 2 focused `web_search` calls per sub-question, varying keywords "
    "and angles between them — don't settle for the first page of one query. Reformulate whenever "
    "the results are weak or one-sided.\n"
    "3. READ DEEP: `fetch_url` the most authoritative results (primary sources, official docs, "
    "reputable outlets) — snippets are leads, not evidence. Read at least 3-4 distinct sources "
    "across the question, and 2-3 per sub-question when coverage allows.\n"
    "4. CROSS-CHECK: for every important claim, look for a second independent source. Note "
    "disagreements honestly instead of picking a side silently; prefer recent sources for "
    "time-sensitive facts.\n"
    "5. SYNTHESIZE: you MUST finish by calling `create_document` exactly once with the complete, "
    "cited markdown report — an executive summary up top, a section per sub-question, and a "
    "Sources section listing every cited URL, with the URL cited next to each claim it supports. "
    "A research turn that ends without a saved report document is INCOMPLETE; never stop at a "
    "chat-only summary.\n"
    "RULES: never invent a source or a fact — if the web tools error or coverage is thin, say "
    "exactly what could not be verified. Keep intermediate notes out of the report; it should "
    "read as a finished brief."
)

RESEARCH_MODE_INSTRUCTIONS = (
    "This conversation is in DEEP RESEARCH mode: the user wants investigated, sourced answers, "
    "not from-memory replies. For any substantive question, run the full research method below "
    "before answering. The saved report document is THE deliverable — keep your chat reply to a "
    "short summary of the findings that points the user at the report document.\n" + _METHOD
)

DEEP_RESEARCH_INSTRUCTIONS = (
    "You are a deep-research specialist dispatched to investigate one question. You have web "
    "search/fetch and document tools. The saved report document is your primary deliverable; your "
    "final text reply goes to the agent that dispatched you — make it a concise digest of your "
    "findings (key facts, the strongest sources, open uncertainties) and name the report document "
    "you created.\n" + _METHOD
)

research_mode_capability = Capability(
    id="DeepResearchMode", instructions=RESEARCH_MODE_INSTRUCTIONS
)

research_safeguard_hooks = Hooks()


def _report_title(output: str) -> str:
    """Derive a short document title from the report body (first heading / non-empty line)."""
    for line in output.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:80]
    return "Research Report"


@research_safeguard_hooks.on.after_run
async def _ensure_report_document(
    ctx: RunContext[GraphDependencies], *, result: AgentRunResult[Any]
) -> AgentRunResult[Any]:
    """Save the run's answer as a report document when the agent skipped ``create_document``.

    Research mode's report is instruction-driven, so a model that forgets the final
    ``create_document`` call would leave the user with a chat-only answer and no saved brief. This
    backstops that: if no ``create_document`` tool call ran this turn and the agent produced a
    substantive answer, persist it as a markdown document. Best-effort — a DB failure is logged and
    swallowed, never crashing the run (same contract as the memory-persistence hooks).
    """
    made_document = any(
        isinstance(part, ToolCallPart) and part.tool_name == "create_document"
        for message in result.new_messages()
        for part in message.parts
    )
    output = (result.output or "").strip()
    if made_document or len(output) < _MIN_REPORT_CHARS:
        return result
    deps = ctx.deps
    try:
        await repo.create_document(
            deps.db,
            deps.user_id,
            deps.conversation_id,
            title=_report_title(output),
            content=output,
        )
    except Exception:  # noqa: BLE001 — the safeguard is best-effort; never abort the run.
        logger.warning("research report safeguard failed; continuing without it", exc_info=True)
    return result


def build_research() -> list[Capability]:
    """Return the research-mode overlay for ``Agent(capabilities=...)``.

    Added on top of the regular capability set when a conversation's mode is ``research``: the
    method instructions run on the existing web/document tools, and the ``after_run`` safeguard
    guarantees a report document even when the model forgets to create one itself.
    """
    return [research_mode_capability, research_safeguard_hooks]

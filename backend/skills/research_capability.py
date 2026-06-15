"""DeepResearch: the research-mode methodology and the deep-research sub-agent prompt.

Two consumers:

- **Research mode** (a conversation created with ``mode="research"``): :func:`build_research`
  returns an instructions-only capability layered onto the regular agent, turning it into a
  methodical researcher — plan, fan out searches, read sources, cross-check, then deliver a
  cited report as a document. No new tools: the work runs on the existing ``web_search`` /
  ``fetch_url`` / ``create_document`` tools, so every step streams to the UI as ordinary
  tool chips.
- **The swarm's built-in ``deep_research`` tool**: :data:`DEEP_RESEARCH_INSTRUCTIONS` is the
  system prompt of the delegated researcher sub-agent (run via
  :func:`backend.skills.subagent.run_subagent` with the ``web`` + ``documents`` tool groups).
"""

from __future__ import annotations

import os

from pydantic_ai.capabilities import Capability

# Ceiling on the deep-research sub-agent's model round-trips. Research is the deepest delegated
# loop (many search/fetch cycles), so it gets a higher default than ordinary sub-agents.
DEEP_RESEARCH_REQUEST_LIMIT = int(os.getenv("DEEP_RESEARCH_REQUEST_LIMIT", "40"))

# The shared research method. Phrased agent-agnostically so it reads correctly both as a mode
# overlay on the main agent and inside the delegated researcher's system prompt.
_METHOD = (
    "RESEARCH METHOD — follow these phases in order:\n"
    "1. PLAN: break the question into 2-5 concrete sub-questions. State them briefly before "
    "searching.\n"
    "2. SEARCH WIDE: run several focused `web_search` calls — one per sub-question, plus "
    "reformulations when results are weak. Vary keywords and angles; don't settle for the first "
    "page of one query.\n"
    "3. READ DEEP: `fetch_url` the most authoritative results (primary sources, official docs, "
    "reputable outlets) — snippets are leads, not evidence. Read at least 2-3 sources per "
    "sub-question when available.\n"
    "4. CROSS-CHECK: for every important claim, look for a second independent source. Note "
    "disagreements honestly instead of picking a side silently; prefer recent sources for "
    "time-sensitive facts.\n"
    "5. SYNTHESIZE: write the findings up as a structured markdown report and save it with "
    "`create_document` — sections per sub-question, an executive summary up top, and a Sources "
    "section listing every cited URL. Cite the URL next to each claim it supports.\n"
    "RULES: never invent a source or a fact — if the web tools error or coverage is thin, say "
    "exactly what could not be verified. Keep intermediate notes out of the report; it should "
    "read as a finished brief."
)

RESEARCH_MODE_INSTRUCTIONS = (
    "This conversation is in DEEP RESEARCH mode: the user wants investigated, sourced answers, "
    "not from-memory replies. For any substantive question, run the full research method below "
    "before answering; keep the chat reply to a short summary of findings that points at the "
    "report document.\n" + _METHOD
)

DEEP_RESEARCH_INSTRUCTIONS = (
    "You are a deep-research specialist dispatched to investigate one question. You have web "
    "search/fetch and document tools, and your final text reply goes to the agent that "
    "dispatched you — make it a concise digest of your findings (key facts, the strongest "
    "sources, open uncertainties) and name the report document you created.\n" + _METHOD
)

research_mode_capability = Capability(
    id="DeepResearchMode", instructions=RESEARCH_MODE_INSTRUCTIONS
)


def build_research() -> list[Capability]:
    """Return the research-mode overlay for ``Agent(capabilities=...)`` (instructions only).

    Added on top of the regular capability set when a conversation's mode is ``research``; the
    method runs on the existing web/document tools, so nothing else needs wiring.
    """
    return [research_mode_capability]

"""Pydantic models for the Swarm capability's tool inputs/outputs.

The swarm follows the "agency" communication model: the entry-point orchestrator defines specialist
sub-agents (persisted as ``AgentSpec`` vertices) AND the communication chart between them (each
agent's ``recipients`` — the teammates it may ``send_message``). Agents communicate with the single
``send_message`` primitive (one recipient) or ``send_messages`` (an independent batch run
concurrently); messages flow multi-hop along the chart. The identifier validators here are the
safety boundary for agent names (they are stored as data, never interpolated into DDL, so this is
hygiene rather than injection defense — unlike graph_schemas).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from backend.schemas.document_schemas import DocumentInfo

# kebab-case, 2-40 chars: 'pitch-deck-designer', 'market-researcher'.
_AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,39}$")

# The tool bundles a sub-agent may be granted. Keys are what CreateAgentArgs.tools accepts;
# the matching capability builders live in backend.skills.subagent.
TOOL_GROUPS: dict[str, str] = {
    "web": "web_search + fetch_url (live internet via SearXNG)",
    "documents": "create/update/read/list documents (durable artifacts the user sees)",
    "sandbox": "run_python (containerized Python; can produce PDFs and files)",
    "memory": "search_memory/store_fact/run_query (the user's graph memory)",
}


def _valid_tools(tools: list[str]) -> list[str]:
    cleaned = [t.strip().lower() for t in tools]
    unknown = [t for t in cleaned if t not in TOOL_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown tool group(s) {unknown}; choose from {sorted(TOOL_GROUPS)}."
        )
    # De-duplicate, preserving order.
    return list(dict.fromkeys(cleaned))


def _valid_recipients(recipients: list[str]) -> list[str]:
    """Normalize the agency-chart edges: kebab-case teammate names, de-duplicated.

    The names are validated for shape only; whether each names a real roster agent is checked
    tolerantly at dispatch (an agent may list a teammate created later in the same turn).
    """
    cleaned = [r.strip().lower() for r in recipients]
    bad = [r for r in cleaned if not _AGENT_NAME_RE.match(r)]
    if bad:
        raise ValueError(
            f"Invalid recipient name(s) {bad}; each must be a kebab-case agent name, "
            "e.g. 'market-researcher'."
        )
    return list(dict.fromkeys(cleaned))


class CreateAgentArgs(BaseModel):
    """A new specialist sub-agent for the swarm roster."""

    name: str = Field(
        ...,
        description="Unique kebab-case name, e.g. 'pitch-deck-designer' or 'market-researcher'.",
    )
    role: str = Field(
        ..., min_length=1, description="One-line description of what this agent is for."
    )
    instructions: str = Field(
        ...,
        min_length=1,
        description=(
            "The agent's system prompt: its expertise, working style, and quality bar. "
            "Write it like a job description for a focused specialist."
        ),
    )
    tools: list[str] = Field(
        default_factory=lambda: ["web", "documents"],
        description=f"Tool groups to grant, from {sorted(TOOL_GROUPS)}.",
    )
    recipients: list[str] = Field(
        default_factory=list,
        description=(
            "The agency communication chart for this agent: the kebab-case names of teammates it "
            "may `send_message` (delegate to). Leave empty for a leaf specialist that only reports "
            "back. Recipients may name agents you create later in the same turn."
        ),
    )

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        low = v.strip().lower()
        if not _AGENT_NAME_RE.match(low):
            raise ValueError(
                "name must be kebab-case (lowercase letters, digits, hyphens), 2-40 chars, "
                "e.g. 'pitch-deck-designer'."
            )
        return low

    @field_validator("tools")
    @classmethod
    def _check_tools(cls, v: list[str]) -> list[str]:
        return _valid_tools(v)

    @field_validator("recipients")
    @classmethod
    def _check_recipients(cls, v: list[str]) -> list[str]:
        return _valid_recipients(v)


class UpdateAgentArgs(BaseModel):
    """Revise an existing sub-agent in place (instead of creating a near-duplicate)."""

    agent: str = Field(..., description="The agent's id or name (from list_agents).")
    role: str | None = Field(None, description="New one-line role; omit to keep the current one.")
    instructions: str | None = Field(
        None, description="New full system prompt (replaces the old one); omit to keep it."
    )
    tools: list[str] | None = Field(
        None, description=f"New tool-group list (replaces the old one), from {sorted(TOOL_GROUPS)}."
    )
    recipients: list[str] | None = Field(
        None,
        description=(
            "New communication-chart edges (replaces the old list): kebab-case teammate names this "
            "agent may `send_message`. Pass [] to make it a leaf; omit to keep the current edges."
        ),
    )

    @field_validator("tools")
    @classmethod
    def _check_tools(cls, v: list[str] | None) -> list[str] | None:
        return _valid_tools(v) if v is not None else None

    @field_validator("recipients")
    @classmethod
    def _check_recipients(cls, v: list[str] | None) -> list[str] | None:
        return _valid_recipients(v) if v is not None else None


class AgentSpecInfo(BaseModel):
    """One sub-agent of the swarm roster, as returned by list_agents/create_agent."""

    agent_id: str
    name: str
    role: str = ""
    instructions: str = ""
    tools: list[str] = Field(default_factory=list)
    recipients: list[str] = Field(
        default_factory=list,
        description="Teammates this agent may send_message (its outgoing communication-chart edges).",
    )
    created_at: str | None = None
    updated_at: str | None = None


class SendMessageArgs(BaseModel):
    """One message to send to a teammate agent (the agency `send_message` primitive)."""

    recipient: str = Field(
        ..., description="The teammate agent's id or name (from list_agents), per the chart."
    )
    message: str = Field(
        ..., min_length=1, description="The full, self-contained assignment/question for the agent."
    )
    context: str | None = Field(
        None,
        description=(
            "Optional context the recipient needs (findings so far, constraints, source material). "
            "Recipients do NOT see this conversation — pass everything they need here."
        ),
    )


class SendMessagesArgs(BaseModel):
    """A batch of INDEPENDENT messages to send concurrently (parallel agency fan-out)."""

    messages: list[SendMessageArgs] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "Independent messages (max 8); they are delivered in parallel and must not depend on "
            "each other's results. Recipients may differ or repeat."
        ),
    )


class AgentRunReport(BaseModel):
    """The report from one delivered message (tolerant: failures land in `error`, never raise)."""

    agent_id: str
    name: str = ""
    # The assignment the agent was sent (kept as ``task`` for stable tool-result/UI contracts).
    task: str = ""
    output: str = ""
    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="Documents the agent created (already persisted; reference, don't recreate).",
    )
    error: str | None = None


class SwarmRunResult(BaseModel):
    """All reports of a send_messages batch, in the same order as the submitted messages."""

    reports: list[AgentRunReport] = Field(default_factory=list)


class DeepResearchArgs(BaseModel):
    """A question for the built-in deep-research sub-agent."""

    question: str = Field(..., min_length=1, description="The research question to investigate.")
    focus: str | None = Field(
        None,
        description="Optional scope/angle to emphasize (time window, region, audience, depth).",
    )


class DeepResearchResult(BaseModel):
    """The deep researcher's findings (tolerant: failures land in `error`, never raise)."""

    question: str
    report: str = ""
    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="The cited report document(s) the researcher persisted.",
    )
    error: str | None = None

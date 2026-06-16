"""SwarmOrchestrator capability: define specialist sub-agents and dispatch work to them.

Exposed via :func:`build_swarm`, added to ``Agent(capabilities=...)`` when a conversation's mode
is ``swarm``. The main agent becomes an orchestrator that:

- manages a persistent roster of specialist sub-agents (``AgentSpec`` vertices, user-scoped):
  ``list_agents`` / ``create_agent`` / ``update_agent`` / ``delete_agent``;
- dispatches work: ``run_agent`` (one task) and ``run_swarm`` (several independent tasks run
  CONCURRENTLY, capped by ``SWARM_MAX_PARALLEL``);
- carries a built-in ``deep_research`` sub-agent (web + documents, the deep-research method from
  :mod:`backend.skills.research_capability`) so heavy research can be delegated without first
  defining an agent for it.

Sub-agents run via :func:`backend.skills.subagent.run_subagent` on the parent run's dependencies,
so their documents/facts persist into the same conversation; documents they create ride back on
the tool results (``AgentRunReport.documents``) and become UI artifact cards via
``main._document_events``. Dispatch is tolerant (a failed delegate is an ``error`` in its report,
never an exception); roster mistakes (unknown/duplicate names) raise ``ModelRetry`` like the
document tools.
"""

from __future__ import annotations

import asyncio
import logging
import os

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.schemas.document_schemas import DocumentInfo
from backend.schemas.swarm_schemas import (
    TOOL_GROUPS,
    AgentRunReport,
    AgentSpecInfo,
    AgentTask,
    CreateAgentArgs,
    DeepResearchArgs,
    DeepResearchResult,
    RunSwarmArgs,
    SwarmRunResult,
    UpdateAgentArgs,
)
from backend.skills.research_capability import (
    DEEP_RESEARCH_INSTRUCTIONS,
    DEEP_RESEARCH_REQUEST_LIMIT,
)
from backend.skills.subagent import run_subagent

logger = logging.getLogger("agent_graph.swarm")

# How many sub-agents may run at the same moment inside one run_swarm call. Each one is a full
# agent loop (model calls + tools), so this caps model-provider and DB pressure, not correctness:
# excess tasks simply queue on the semaphore.
DEFAULT_MAX_PARALLEL = int(os.getenv("SWARM_MAX_PARALLEL", "4"))

_TOOL_GROUP_LINES = "\n".join(f"  - {name}: {desc}" for name, desc in sorted(TOOL_GROUPS.items()))

INSTRUCTIONS = (
    "This conversation is in SWARM mode: you are the ORCHESTRATOR of a team of specialist "
    "sub-agents that you design yourself. Your job is to decompose the user's goal, staff it, "
    "dispatch the work, and synthesize the results — not to grind through every subtask "
    "personally.\n"
    "THE ROSTER: sub-agents are durable — they persist across turns and conversations. ALWAYS "
    "call `list_agents` before creating one: reuse a fitting specialist, or `update_agent` to "
    "sharpen its instructions/tools, instead of creating a near-duplicate. Create new agents "
    "with `create_agent` (kebab-case name, one-line role, a focused system prompt written like "
    "a job description, and only the tool groups the job needs):\n"
    f"{_TOOL_GROUP_LINES}\n"
    "Retire obsolete specialists with `delete_agent`.\n"
    "DISPATCHING: send one task with `run_agent`. When subtasks are INDEPENDENT of each other "
    "(e.g. researching the market, drafting slide copy, and building a financial model for a "
    "pitch deck), batch them into ONE `run_swarm` call — its tasks run in PARALLEL, which is "
    "much faster than sequential run_agent calls. Sub-agents do NOT see this conversation: every "
    "task must be self-contained, with the goal, constraints, audience and any needed findings "
    "passed in `task`/`context`. For work that needs serious multi-source investigation, use the "
    "built-in `deep_research` instead of a hand-rolled researcher.\n"
    "SYNTHESIS: sub-agent reports come back as tool results, and documents they produced are "
    "listed on each report — reference those documents, don't recreate their content. After a "
    "dispatch, weave the reports into one coherent answer (or a final document) yourself; "
    "resolve conflicts between reports, and re-dispatch with sharper instructions when a result "
    "is weak. If a report carries an `error`, tell the user honestly and decide whether to "
    "retry, reassign, or proceed without it."
)

swarm_capability = Capability(id="SwarmOrchestrator", instructions=INSTRUCTIONS)


def _spec_info(row: dict) -> AgentSpecInfo:
    return AgentSpecInfo(
        agent_id=row.get("agent_id", ""),
        name=row.get("name") or "",
        role=row.get("role") or "",
        instructions=row.get("instructions") or "",
        tools=list(row.get("tools") or []),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _subagent_system_prompt(row: dict) -> str:
    """The delegated agent's system prompt: its spec, framed as a swarm member."""
    return (
        f"You are '{row.get('name')}', a specialist sub-agent in a swarm.\n"
        f"ROLE: {row.get('role') or 'specialist'}\n"
        f"{row.get('instructions') or ''}\n"
        "You were dispatched by an orchestrator with ONE task. You cannot see the wider "
        "conversation — work only from the assignment you were given. Do the task completely, "
        "then reply with a focused report of what you did and found (your reply goes to the "
        "orchestrator, not the end user). If you created documents, name them in the report "
        "instead of pasting their full content."
    )


async def _dispatch(deps: GraphDependencies, task: AgentTask) -> AgentRunReport:
    """Run one task on its sub-agent. Tolerant: every failure becomes an error report."""
    try:
        row = await repo.get_agent_spec(deps.db, deps.user_id, task.agent)
    except Exception as exc:  # noqa: BLE001 — a DB hiccup must not abort a parallel batch.
        logger.warning("agent lookup failed for %r: %s", task.agent, exc, exc_info=True)
        return AgentRunReport(
            agent_id=task.agent, task=task.task, error=f"Agent lookup failed: {exc}"
        )
    if row is None:
        return AgentRunReport(
            agent_id=task.agent,
            task=task.task,
            error=(
                f"No agent named or with id {task.agent!r}. Call list_agents to see the roster, "
                "or create_agent to define this specialist first."
            ),
        )
    prompt = task.task
    if task.context:
        prompt += f"\n\nCONTEXT FROM THE ORCHESTRATOR:\n{task.context}"
    outcome = await run_subagent(
        deps,
        instructions=_subagent_system_prompt(row),
        tool_groups=list(row.get("tools") or []),
        prompt=prompt,
    )
    return AgentRunReport(
        agent_id=row.get("agent_id", task.agent),
        name=row.get("name") or "",
        task=task.task,
        output=outcome.output,
        documents=outcome.documents,
        error=outcome.error,
    )


@swarm_capability.tool
async def list_agents(ctx: RunContext[GraphDependencies]) -> list[AgentSpecInfo]:
    """List the swarm roster: every sub-agent's id, name, role, instructions and tool groups."""
    rows = await repo.list_agent_specs(ctx.deps.db, ctx.deps.user_id)
    return [_spec_info(r) for r in rows if r.get("agent_id")]


@swarm_capability.tool
async def create_agent(
    ctx: RunContext[GraphDependencies], args: CreateAgentArgs
) -> AgentSpecInfo:
    """Define a new specialist sub-agent for the swarm. Returns its agent_id.

    Check list_agents first — update an existing specialist with update_agent rather than
    creating a near-duplicate.
    """
    deps = ctx.deps
    existing = await repo.get_agent_spec(deps.db, deps.user_id, args.name)
    if existing is not None:
        raise ModelRetry(
            f"An agent named {args.name!r} already exists (id {existing.get('agent_id')!r}). "
            "Use update_agent to revise it, or pick a different name."
        )
    agent_id = await repo.create_agent_spec(
        deps.db,
        deps.user_id,
        name=args.name,
        role=args.role,
        instructions=args.instructions,
        tools=args.tools,
    )
    return AgentSpecInfo(
        agent_id=agent_id,
        name=args.name,
        role=args.role,
        instructions=args.instructions,
        tools=args.tools,
    )


@swarm_capability.tool
async def update_agent(ctx: RunContext[GraphDependencies], args: UpdateAgentArgs) -> str:
    """Revise an existing sub-agent's role, instructions and/or tools (its name is immutable).

    `instructions` and `tools` REPLACE the old values — send the complete new prompt/list.
    """
    if args.role is None and args.instructions is None and args.tools is None:
        raise ModelRetry("Nothing to update: pass a new role, instructions and/or tools.")
    deps = ctx.deps
    row = await repo.get_agent_spec(deps.db, deps.user_id, args.agent)
    if row is None:
        raise ModelRetry(
            f"No agent named or with id {args.agent!r}. Use list_agents to find it."
        )
    await repo.update_agent_spec(
        deps.db,
        deps.user_id,
        row["agent_id"],
        role=args.role,
        instructions=args.instructions,
        tools=args.tools,
    )
    return f"Updated agent {row.get('name') or row['agent_id']}."


@swarm_capability.tool
async def delete_agent(ctx: RunContext[GraphDependencies], agent: str) -> str:
    """Retire a sub-agent from the roster, by id or name (from list_agents)."""
    deps = ctx.deps
    row = await repo.get_agent_spec(deps.db, deps.user_id, agent)
    if row is None:
        raise ModelRetry(f"No agent named or with id {agent!r}. Use list_agents to find it.")
    await repo.delete_agent_spec(deps.db, deps.user_id, row["agent_id"])
    return f"Deleted agent {row.get('name') or row['agent_id']}."


@swarm_capability.tool
async def run_agent(ctx: RunContext[GraphDependencies], args: AgentTask) -> AgentRunReport:
    """Dispatch ONE self-contained task to a sub-agent and return its report.

    The sub-agent cannot see this conversation — put everything it needs in task/context. A
    failure inside the delegate comes back as the report's `error`, with whatever documents it
    managed to produce.
    """
    return await _dispatch(ctx.deps, args)


@swarm_capability.tool
async def run_swarm(ctx: RunContext[GraphDependencies], args: RunSwarmArgs) -> SwarmRunResult:
    """Dispatch several INDEPENDENT tasks concurrently; reports return in task order.

    Much faster than sequential run_agent calls — use it whenever subtasks don't depend on each
    other's results. Tasks may target the same agent or different ones. Each report is
    independent: one failing task never affects the others.
    """
    deps = ctx.deps
    semaphore = asyncio.Semaphore(DEFAULT_MAX_PARALLEL)

    async def _guarded(task: AgentTask) -> AgentRunReport:
        async with semaphore:
            return await _dispatch(deps, task)

    reports = await asyncio.gather(*(_guarded(t) for t in args.tasks))
    return SwarmRunResult(reports=list(reports))


@swarm_capability.tool
async def deep_research(
    ctx: RunContext[GraphDependencies], args: DeepResearchArgs
) -> DeepResearchResult:
    """Delegate a question to the built-in deep-research sub-agent (web + documents).

    It plans sub-questions, fans out searches, reads and cross-checks sources, and saves a cited
    markdown report document; the returned digest summarizes the findings. Use it for anything
    needing real multi-source investigation — no need to create_agent a researcher first.
    """
    prompt = f"Research question: {args.question}"
    if args.focus:
        prompt += f"\nFocus / scope: {args.focus}"
    outcome = await run_subagent(
        ctx.deps,
        instructions=DEEP_RESEARCH_INSTRUCTIONS,
        tool_groups=["web", "documents"],
        prompt=prompt,
        request_limit=DEEP_RESEARCH_REQUEST_LIMIT,
    )
    documents = outcome.documents
    # Guarantee a saved report even when the delegate forgot to call create_document: persist its
    # digest as a markdown document. Best-effort — a DB hiccup just leaves documents empty.
    if not documents and outcome.output.strip():
        title = f"Research: {args.question}"[:80]
        try:
            document_id = await repo.create_document(
                ctx.deps.db,
                ctx.deps.user_id,
                ctx.deps.conversation_id,
                title=title,
                content=outcome.output,
            )
            documents = [
                DocumentInfo(
                    document_id=document_id,
                    conversation_id=ctx.deps.conversation_id,
                    title=title,
                    mime_type="text/markdown",
                    encoding="text",
                )
            ]
        except Exception:  # noqa: BLE001 — fallback persistence is best-effort; never raise.
            logger.warning("deep_research report fallback failed; continuing", exc_info=True)
    return DeepResearchResult(
        question=args.question,
        report=outcome.output,
        documents=documents,
        error=outcome.error,
    )


def build_swarm() -> list[Capability]:
    """Return the swarm-orchestrator capability to add to ``Agent(capabilities=...)``.

    Dependencies (db/web/sandbox/model) are supplied per-run through ``GraphDependencies``, so
    nothing needs to be wired in here.
    """
    return [swarm_capability]

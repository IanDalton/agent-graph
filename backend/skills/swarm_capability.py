"""SwarmOrchestrator capability: design an agency of specialists and route work through it.

Exposed via :func:`build_swarm`, added to ``Agent(capabilities=...)`` when a conversation's mode
is ``swarm``. The main agent becomes the **entry point of an agency** (the agent the user talks to)
that:

- manages a persistent roster of specialist agents (``AgentSpec`` vertices, user-scoped) AND the
  communication chart between them — each agent's ``recipients`` are the teammates it may
  ``send_message``: ``list_agents`` / ``create_agent`` / ``update_agent`` / ``delete_agent``;
- communicates with the single agency primitive ``send_message`` (one recipient) or
  ``send_messages`` (a batch of INDEPENDENT messages delivered CONCURRENTLY, capped by
  ``SWARM_MAX_PARALLEL``). Messages flow MULTI-HOP along the chart: a dispatched specialist that
  has its own ``recipients`` is granted its own ``send_message`` tool (within ``SWARM_MAX_DEPTH``);
- carries a built-in ``deep_research`` specialist (web + documents, the deep-research method from
  :mod:`backend.skills.research_capability`) so heavy research can be delegated without first
  defining an agent for it.

Specialists run via :func:`backend.skills.subagent.dispatch_message` →
:func:`backend.skills.subagent.run_subagent` on the parent run's dependencies, so their
documents/facts persist into the same conversation; documents they create ride back on the tool
results (``AgentRunReport.documents``) and become UI artifact cards via ``main._document_events``.
Communication is tolerant (a bad/out-of-chart recipient or a failed delegate is an ``error`` in its
report, never an exception); roster mistakes (unknown/duplicate names) raise ``ModelRetry`` like the
document tools.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability, Hooks

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.schemas.document_schemas import DocumentInfo
from backend.schemas.swarm_schemas import (
    TOOL_GROUPS,
    AgentRunReport,
    AgentSpecInfo,
    CreateAgentArgs,
    DeepResearchArgs,
    DeepResearchResult,
    SendMessageArgs,
    SendMessagesArgs,
    SwarmRunResult,
    UpdateAgentArgs,
)
from backend.skills.research_capability import (
    DEEP_RESEARCH_INSTRUCTIONS,
    DEEP_RESEARCH_REQUEST_LIMIT,
)
from backend.skills.subagent import dispatch_message, dispatch_messages, run_subagent

logger = logging.getLogger("agent_graph.swarm")

_TOOL_GROUP_LINES = "\n".join(f"  - {name}: {desc}" for name, desc in sorted(TOOL_GROUPS.items()))

INSTRUCTIONS = (
    "This conversation is in SWARM mode: you are the ENTRY POINT and ORCHESTRATOR of an AGENCY of "
    "specialist sub-agents. CRITICAL: you have NO tools to browse the web, run code, or write "
    "documents yourself — you cannot do the work directly. Your ONLY way to get anything done is to "
    "delegate to specialists and synthesize their reports. The user only talks to you.\n"
    "THE ROSTER: a standard team already exists — ALWAYS call `list_agents` first to see it "
    "(typically web-researcher, report-writer, website-builder, pdf-author, presentation-designer, "
    "and a `team-lead` sub-orchestrator). Reuse a fitting specialist; `update_agent` to sharpen "
    "one; only `create_agent` for a genuine gap (kebab-case name, one-line role, a focused "
    "job-description prompt, the tool groups the job needs, and `recipients` if it should delegate "
    "onward). The tool groups specialists can be granted:\n"
    f"{_TOOL_GROUP_LINES}\n"
    "`recipients` are the communication chart: a specialist may only `send_message` the teammates "
    "in its own `recipients`, and those messages flow multi-hop along the chart. Leave recipients "
    "empty for a leaf worker that only reports back. Retire obsolete specialists with "
    "`delete_agent`.\n"
    "SKILLS: the user has a library of installed skills (focused procedures, some with runnable "
    "scripts) — they are listed below under 'Skills in the user's library'. When a specialist's job "
    "matches one, GRANT it by putting the skill name in that agent's `skills` (on create_agent / "
    "update_agent); the agent then gets `load_skill` and uses it. A skill that ships scripts ALSO "
    "needs the `sandbox` tool group to run them. Assign only the skills relevant to each agent's job "
    "— don't grant the whole library.\n"
    "DELEGATING — PREFER PARALLEL: when subtasks are INDEPENDENT (e.g. research the market AND draft "
    "the copy AND build a model), you MUST batch them into ONE `send_messages` call so they run in "
    "PARALLEL — do NOT issue them as separate sequential `send_message` calls. Use a single "
    "`send_message` only for a one-off task or a step that depends on a previous result. For "
    "serious multi-source investigation, use the built-in `deep_research`. Specialists do NOT see "
    "this conversation: every message must be self-contained — put the goal, constraints, audience "
    "and any findings to build on in `message`/`context`.\n"
    "SUB-ORCHESTRATORS: for a big, multi-part sub-goal, don't micro-manage every worker yourself — "
    "hand the whole sub-goal to a sub-orchestrator that has its own team. The seeded `team-lead` "
    "(its `recipients` are the worker specialists) will itself fan out to them in PARALLEL and "
    "synthesize the results; you can also `create_agent` a new sub-orchestrator by giving it the "
    "right `recipients`. Delegating one rich brief to `team-lead` is often better than issuing many "
    "low-level messages yourself.\n"
    "DELIVERABLES & SYNTHESIS: the specialists produce the artifacts (documents, PDFs, sites), "
    "which are listed on each report. You cannot create documents yourself, so REFERENCE those "
    "documents in your answer — never paste or recreate their content. Weave the reports into one "
    "coherent reply, resolve conflicts between them, and re-dispatch with sharper instructions when "
    "a result is weak. If a report carries an `error`, tell the user honestly and decide whether to "
    "retry, reassign, or proceed without it."
)

swarm_capability = Capability(id="SwarmOrchestrator", instructions=INSTRUCTIONS)


# The standard specialist roster seeded on a user's first swarm turn (see ``swarm_seed_hooks``).
# The workers are leaves (no ``recipients``) the orchestrator messages directly; together they cover
# the common deliverables (research, markdown reports, interactive sites, PDFs, slide decks). The
# ``team-lead`` is a SUB-ORCHESTRATOR whose ``recipients`` are those workers: hand it a complex
# multi-part sub-goal and it fans out to the workers in parallel and synthesizes their reports — so
# hierarchical delegation works out of the box without the user hand-wiring a chart.
_WORKER_NAMES = [
    "web-researcher",
    "report-writer",
    "website-builder",
    "pdf-author",
    "presentation-designer",
]
DEFAULT_SWARM_AGENTS: list[dict] = [
    {
        "name": "web-researcher",
        "role": "Runs targeted web searches and reads the sources",
        "instructions": (
            "You are a web researcher. Use web_search to find authoritative sources and fetch_url "
            "to read them, then answer the assignment with the concrete facts you found, each with "
            "its source URL. If asked to save your findings, use create_document (markdown). Never "
            "invent a source or a fact; if something can't be confirmed, say so."
        ),
        "tools": ["web", "documents"],
    },
    {
        "name": "report-writer",
        "role": "Writes polished markdown reports and notes as documents",
        "instructions": (
            "You write clear, well-structured markdown documents from the material you are given. "
            "Call create_document exactly once with the full report (headings, bullet lists, "
            "tables where useful) and reply with the document title. Do NOT browse or invent "
            "facts — work only from the content in your assignment."
        ),
        "tools": ["documents"],
    },
    {
        "name": "website-builder",
        "role": "Builds a self-contained interactive HTML website",
        "instructions": (
            "You build a single self-contained web page. Call create_document with "
            "mime_type='text/html' and content that is ONE complete HTML document with all CSS and "
            "JavaScript INLINE — no external files, CDNs, or network calls (it runs in a sandboxed "
            "iframe). Make it clean and responsive, and reply with the document title. Work only "
            "from the content in your assignment."
        ),
        "tools": ["documents"],
    },
    {
        "name": "pdf-author",
        "role": "Produces polished PDF documents with fpdf2",
        "instructions": (
            "You produce PDF documents. Write a complete, self-contained Python program (stdlib + "
            "fpdf2 only, no network) that builds the PDF and saves it to /out/<name>.pdf, then call "
            "run_python with it. Use Helvetica and multi_cell for wrapping, and keep text "
            "latin-1-safe. The saved /out PDF is your deliverable — reply with its filename. Work "
            "only from the content in your assignment."
        ),
        "tools": ["sandbox"],
    },
    {
        "name": "presentation-designer",
        "role": "Builds slide-deck PDFs, one slide per page",
        "instructions": (
            "You build slide-deck presentations as a PDF. Write a self-contained Python program "
            "(stdlib + fpdf2 only) that creates a LANDSCAPE PDF with ONE slide per page — a large "
            "title, a few concise bullet points, generous spacing — saved to /out/<name>.pdf, then "
            "call run_python. Keep text latin-1-safe and reply with the saved filename."
        ),
        "tools": ["sandbox"],
    },
    {
        "name": "team-lead",
        "role": "Sub-orchestrator: decomposes a complex goal and runs the workers in parallel",
        "instructions": (
            "You are a SUB-ORCHESTRATOR leading a team of worker specialists. You were handed ONE "
            "complex, multi-part goal. Break it into the independent pieces each worker is best at, "
            "then dispatch them with a SINGLE `send_messages` call so they run IN PARALLEL (use one "
            "`send_message` only for a lone or dependent step). Your teammates: web-researcher "
            "(web search + reading sources), report-writer (markdown reports), website-builder "
            "(interactive HTML), pdf-author (PDF via fpdf2), presentation-designer (slide-deck PDF). "
            "Each teammate is blind to your context, so give every message a complete, "
            "self-contained brief. When their reports come back, synthesize them into one coherent "
            "answer (or a combined document) and reply with that summary, naming the documents they "
            "produced rather than pasting their contents."
        ),
        "tools": ["documents"],
        "recipients": _WORKER_NAMES,
    },
]

swarm_seed_hooks = Hooks()


@swarm_seed_hooks.on.before_run
async def _seed_default_agents(ctx: RunContext[GraphDependencies]) -> None:
    """Seed the standard specialist roster on a user's first swarm turn. Best-effort.

    Seed-when-empty: if the user already has any ``AgentSpec``, the roster is left untouched (so
    later renames/edits/deletes are never fought). A DB hiccup is logged and swallowed — seeding
    must never block the turn (same tolerance contract as the other hooks).
    """
    deps = ctx.deps
    try:
        existing = await repo.list_agent_specs(deps.db, deps.user_id)
        if existing:
            return
        for spec in DEFAULT_SWARM_AGENTS:
            await repo.create_agent_spec(
                deps.db,
                deps.user_id,
                name=spec["name"],
                role=spec["role"],
                instructions=spec["instructions"],
                tools=spec["tools"],
                recipients=spec.get("recipients"),
            )
    except Exception:  # noqa: BLE001 — seeding is best-effort; never block the turn.
        logger.warning("default-agent seeding failed; continuing", exc_info=True)


def _spec_info(row: dict) -> AgentSpecInfo:
    return AgentSpecInfo(
        agent_id=row.get("agent_id", ""),
        name=row.get("name") or "",
        role=row.get("role") or "",
        instructions=row.get("instructions") or "",
        tools=list(row.get("tools") or []),
        recipients=list(row.get("recipients") or []),
        skills=list(row.get("skills") or []),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@swarm_capability.tool
async def list_agents(ctx: RunContext[GraphDependencies]) -> list[AgentSpecInfo]:
    """List the agency roster: every agent's id, name, role, instructions, tools and recipients.

    The `recipients` of each agent are the communication chart — who that agent may send_message.
    """
    rows = await repo.list_agent_specs(ctx.deps.db, ctx.deps.user_id)
    return [_spec_info(r) for r in rows if r.get("agent_id")]


@swarm_capability.tool
async def create_agent(
    ctx: RunContext[GraphDependencies], args: CreateAgentArgs
) -> AgentSpecInfo:
    """Define a new specialist agent for the agency. Returns its agent_id.

    Check list_agents first — update an existing specialist with update_agent rather than
    creating a near-duplicate. `recipients` are the teammates this agent may message (its
    outgoing edges in the communication chart); they may name agents you create later this turn.
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
        recipients=args.recipients,
        skills=args.skills,
    )
    return AgentSpecInfo(
        agent_id=agent_id,
        name=args.name,
        role=args.role,
        instructions=args.instructions,
        tools=args.tools,
        recipients=args.recipients,
        skills=args.skills,
    )


@swarm_capability.tool
async def update_agent(ctx: RunContext[GraphDependencies], args: UpdateAgentArgs) -> str:
    """Revise an existing agent's role, instructions, tools, recipients and/or skills (name is immutable).

    `instructions`, `tools`, `recipients` and `skills` REPLACE the old values — send the complete new
    prompt/lists. Pass `recipients=[]` to cut an agent's outgoing edges (make it a leaf), or
    `skills=[]` to revoke all its skills.
    """
    if (
        args.role is None
        and args.instructions is None
        and args.tools is None
        and args.recipients is None
        and args.skills is None
    ):
        raise ModelRetry(
            "Nothing to update: pass a new role, instructions, tools, recipients and/or skills."
        )
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
        recipients=args.recipients,
        skills=args.skills,
    )
    return f"Updated agent {row.get('name') or row['agent_id']}."


@swarm_capability.tool
async def delete_agent(ctx: RunContext[GraphDependencies], agent: str) -> str:
    """Retire an agent from the roster, by id or name (from list_agents)."""
    deps = ctx.deps
    row = await repo.get_agent_spec(deps.db, deps.user_id, agent)
    if row is None:
        raise ModelRetry(f"No agent named or with id {agent!r}. Use list_agents to find it.")
    await repo.delete_agent_spec(deps.db, deps.user_id, row["agent_id"])
    return f"Deleted agent {row.get('name') or row['agent_id']}."


@swarm_capability.tool
async def send_message(
    ctx: RunContext[GraphDependencies], args: SendMessageArgs
) -> AgentRunReport:
    """Send ONE self-contained assignment to a teammate agent and return its report.

    The recipient cannot see this conversation — put everything it needs in message/context. A
    failure inside the delegate (or a bad recipient) comes back as the report's `error`, with
    whatever documents it managed to produce.
    """
    return await dispatch_message(ctx.deps, args.recipient, args.message, args.context)


@swarm_capability.tool
async def send_messages(
    ctx: RunContext[GraphDependencies], args: SendMessagesArgs
) -> SwarmRunResult:
    """Send several INDEPENDENT messages concurrently; reports return in message order.

    Much faster than sequential send_message calls — use it whenever the assignments don't depend
    on each other's results. Messages may target the same agent or different ones. Each report is
    independent: one failing message never affects the others.
    """
    return SwarmRunResult(reports=await dispatch_messages(ctx.deps, args.messages))


@swarm_capability.tool
async def deep_research(
    ctx: RunContext[GraphDependencies], args: DeepResearchArgs
) -> DeepResearchResult:
    """Delegate a question to the built-in deep-research specialist (web + documents).

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
        agent_id="deep_research",
        agent_name="deep-research",
        instance_id=uuid4().hex[:8],
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
    """Return the swarm-orchestrator capability + seeding hook for ``Agent(capabilities=...)``.

    The entry-point orchestrator gets the roster tools plus ``send_message``/``send_messages``
    here; dispatched specialists are granted their own ``send_message`` by ``run_subagent`` when
    their chart and depth allow. ``swarm_seed_hooks`` seeds the standard roster on the user's
    first swarm turn. These are added to the orchestrator only (sub-agents never receive
    ``build_swarm``), so there is no double-seeding and no hooks on delegates. Dependencies
    (db/web/sandbox/model) are supplied per-run through ``GraphDependencies``.
    """
    return [swarm_capability, swarm_seed_hooks]

"""OntologyManager capability: a strict two-tool pipeline for growing the memory ontology.

The agent can extend its own graph schema by creating new GENERIC vertex types, but only
through a guarded pipeline that prevents ontology fragmentation and DDL injection:

1. ``propose_schema_change`` — the cognitive layer. Never touches the database. It validates a
   proposed node type structurally (PascalCase name, typed properties), requires a brief usage
   instruction, then asks an LLM **evaluator sub-agent** to judge it *semantically* against the
   existing ontology (is it generic? a duplicate? should an existing type be reused?). Only an
   approved proposal is recorded in run-scoped state; a rejected one is returned with the
   evaluator's reasoning and a steer toward the right existing type, and is NOT recorded — so the
   executor stays blocked. The evaluator fails open: if it errors, the proposal is approved as
   before, so a flaky evaluator never crashes or stalls the run.
2. ``create_vertex_type`` — the executor. Only runs a node type that was approved earlier in the
   same run; a ``before_tool_execute`` guard rejects any attempt to skip step 1.

A read-only ``list_vertex_types`` tool lets the agent inspect the current ontology (each type's
name, usage note and properties) before proposing, so it reuses an existing generic type instead
of creating a duplicate.

All DB access goes through :mod:`backend.db.repository`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import Capability, Hooks
from pydantic_ai.messages import ToolCallPart

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.model_selection import select_model
from backend.schemas.graph_schemas import (
    CreateEdgeArgs,
    CreateNodeArgs,
    DropEdgeTypeArgs,
    DropVertexTypeArgs,
    EdgeProposal,
    EvaluatorVerdict,
    ProposeEdgeArgs,
    ProposeSchemaArgs,
    SchemaProposal,
    UpdateNodeArgs,
    VertexProperty,
    VertexTypeInfo,
)

logger = logging.getLogger("agent_graph.ontology")

# Internal vertex types the agent must not edit/delete via the generic node/type tools; facts have
# their own update_fact/delete_fact tools, and conversation/message/log/replay records are managed by
# hooks. Dropping any of these would break the memory system.
_PROTECTED_VERTEX_TYPES = frozenset(
    {"User", "Conversation", "Message", "Fact", "LogEntry", "RunMessages"}
)
# Internal edge types that wire the memory graph together — never droppable here.
_PROTECTED_EDGE_TYPES = frozenset(
    {"HAS_CONVERSATION", "HAS_MESSAGE", "HAS_RUN_MESSAGES", "KNOWS", "LOGGED", "HAS_NODE"}
)

ONTOLOGY_INSTRUCTIONS = (
    "You can extend your own memory ontology by creating new vertex (node) TYPES. "
    "You MUST follow this STRICT TWO-STEP PIPELINE, in order:\n"
    "  1. Call `propose_schema_change` FIRST. This does NOT touch the database — it validates your "
    "idea and enforces the Rule of Generality. It returns an approved proposal.\n"
    "  2. ONLY after a proposal for that exact node_name is approved may you call "
    "`create_vertex_type` with the same node_name to actually create it. Calling create_vertex_type "
    "without a prior approved proposal WILL be rejected.\n"
    "RULE OF GENERALITY — propose GENERIC, reusable CATEGORIES, never specific instances:\n"
    "  GOOD: 'SoftwareFramework', 'Person', 'City', 'ProgrammingLanguage', 'Company'.\n"
    "  BAD:  'React', 'JohnDoe', 'BuenosAires', 'Python', 'Anthropic' — those are *instances* and "
    "must be stored as DATA inside a generic type, not as their own types.\n"
    "Every proposal MUST include a brief `usage` instruction — one or two sentences on WHEN to use "
    "the type and what instances belong in it. It is stored ON the type so you (in future runs) "
    "can read it and reuse the right type.\n"
    "Before proposing, ALWAYS call `list_vertex_types` to see what types already exist and read "
    "their usage notes. Never create a duplicate when an existing type's usage note already covers "
    "your data.\n"
    "STORING DATA: `create_vertex_type` only creates the TYPE (the category). To save an actual "
    "INSTANCE — e.g. the framework 'Django' — call `create_node` with that existing type and the "
    "instance's property values. The type must exist first. So the full flow is: "
    "list_vertex_types -> (if needed) propose_schema_change -> create_vertex_type -> create_node. "
    "`create_node` returns the new node's record id (e.g. '#29:0').\n"
    "RELATIONSHIPS (edges): to connect two instances (e.g. a Person USES a SoftwareFramework), use "
    "the parallel edge pipeline — propose_edge_type FIRST (cognitive, enforces a GENERIC "
    "UPPER_SNAKE_CASE name like 'USES'), then create_edge_type to create it, then `create_edge` "
    "passing the source and target record ids. Edge types follow the SAME Rule of Generality: "
    "propose 'USES', never 'usesDjango'. Calling create_edge_type without a prior approved "
    "propose_edge_type WILL be rejected.\n"
    "AVOID DUPLICATES: before creating a node, search for an existing one (run_query, e.g. "
    "`SELECT @rid, name FROM Person WHERE name = 'Alice'`). If it already exists, DON'T create "
    "another — call `update_node` with its @rid to revise its properties, or `delete_node` to remove "
    "a redundant one.\n"
    "RETIRING TYPES (destructive): you have full control of your own schema. To remove a whole "
    "category you created and no longer need, call `delete_vertex_type` (drops the vertex type AND "
    "all its instances + their edges) or `delete_edge_type` (drops the relationship type AND all its "
    "edges, keeping the connected nodes). These are irreversible and operate on the TYPE — to delete "
    "a single record use delete_node instead. Internal types (User, Conversation, Message, Fact, "
    "LogEntry, RunMessages, and the HAS_*/KNOWS/LOGGED relationships) are protected and cannot be dropped."
)

ontology_capability = Capability(id="OntologyManager", instructions=ONTOLOGY_INSTRUCTIONS)


# --------------------------------------------------------------------------- #
# Semantic evaluator sub-agent
#
# A small delegated agent that judges whether a proposed vertex/edge type is a good addition to the
# ontology, given the types that already exist. The propose_* tools call it after Pydantic's
# structural validation; only an approved verdict records the proposal (and so unblocks the matching
# create_*_type). It runs on EVALUATOR_MODEL if set, else the same model the main agent uses.
# --------------------------------------------------------------------------- #
_EVALUATOR_INSTRUCTIONS = (
    "You are an ontology reviewer for a knowledge graph. You judge ONE proposed type at a time and "
    "decide whether it should be added to the graph's schema.\n"
    "Structural rules (PascalCase vertex names, UPPER_SNAKE_CASE edge names, valid property types) "
    "are ALREADY enforced before you see the proposal — do NOT re-check them. Judge SEMANTICS only.\n"
    "APPROVE only when the proposal is a GENERIC, reusable CATEGORY or RELATION that is NOT already "
    "covered by one of the existing types' name or usage note.\n"
    "REJECT when:\n"
    "  - It is a specific INSTANCE, not a category (e.g. 'React', 'Python', 'JohnDoe', 'WORKS_AT_GOOGLE'). "
    "Those belong as DATA (a node/edge) inside a generic type, not as a type.\n"
    "  - An existing type already covers it (a duplicate or near-synonym, e.g. 'UTILIZES' when 'USES' "
    "exists, or 'WebFramework' when 'SoftwareFramework' exists). Set suggested_existing_type to that "
    "type's EXACT name.\n"
    "  - It is too narrow/over-specific to be reused.\n"
    "When you reject, give a brief reason and, when applicable, a revision_hint (how to generalize) "
    "and/or suggested_existing_type (the existing type to use instead)."
)

_evaluator_agent: Agent[None, EvaluatorVerdict] | None = None


def _get_evaluator() -> Agent[None, EvaluatorVerdict]:
    """Lazily build the module-level evaluator agent (so importing this module needs no model env).

    Tests override its model with ``_get_evaluator().override(model=...)`` — the standard Pydantic AI
    test seam — so they never hit a real LLM.
    """
    global _evaluator_agent
    if _evaluator_agent is None:
        model = os.getenv("EVALUATOR_MODEL") or select_model("AGENT_MODEL")
        _evaluator_agent = Agent(
            model, output_type=EvaluatorVerdict, instructions=_EVALUATOR_INSTRUCTIONS
        )
    return _evaluator_agent


def _build_evaluator_prompt(
    kind: str,
    name: str,
    usage: str,
    rationale: str,
    properties: list[VertexProperty],
    existing: list[dict[str, Any]],
) -> str:
    """Render the proposal + current ontology into a prompt for the evaluator."""
    if existing:
        lines = [
            f"- {row.get('name')}: {row.get('usage') or '(no usage note)'}"
            for row in existing
            if row.get("name")
        ]
        ontology = "\n".join(lines)
    else:
        ontology = "(the ontology is empty — no types exist yet)"
    props = ", ".join(f"{p.name}:{p.type}" for p in properties) or "(none)"
    return (
        f"Existing types in the ontology:\n{ontology}\n\n"
        f"Proposed new {kind}:\n"
        f"  name: {name}\n"
        f"  usage: {usage}\n"
        f"  properties: {props}\n"
        f"  rationale: {rationale}\n\n"
        "Should this type be added? Judge it semantically and return your verdict."
    )


def _rejection_guidance(create_tool: str, verdict: EvaluatorVerdict) -> str:
    """Build the steer text returned on a rejected proposal."""
    parts = [f"Rejected by the ontology evaluator: {verdict.reason}"]
    if verdict.suggested_existing_type:
        parts.append(
            f"Use the existing type '{verdict.suggested_existing_type}' instead "
            "(store your data there via create_node/create_edge)."
        )
    if verdict.revision_hint:
        parts.append(f"To revise: {verdict.revision_hint}")
    parts.append(
        f"This was NOT approved, so {create_tool} stays blocked until you submit an approved proposal."
    )
    return " ".join(parts)


async def _evaluate_proposal(
    ctx: RunContext[GraphDependencies],
    *,
    kind: str,
    name: str,
    usage: str,
    rationale: str,
    properties: list[VertexProperty],
) -> EvaluatorVerdict:
    """Ask the evaluator to judge a proposal. Fails open (approves) on any error."""
    try:
        existing = await repo.list_vertex_types(ctx.deps.db)
    except Exception:  # noqa: BLE001 — best-effort; a DB hiccup must not block proposing.
        logger.warning(
            "evaluator: ontology unavailable; approving %s %r by default", kind, name, exc_info=True
        )
        return EvaluatorVerdict(approved=True, reason="evaluator skipped (ontology unavailable)")
    prompt = _build_evaluator_prompt(kind, name, usage, rationale, properties, existing)
    try:
        result = await _get_evaluator().run(prompt, usage=ctx.usage)
        return result.output
    except Exception:  # noqa: BLE001 — FAIL-OPEN: never crash/block the run on a flaky evaluator.
        logger.warning(
            "evaluator failed for %s %r; approving by default", kind, name, exc_info=True
        )
        return EvaluatorVerdict(approved=True, reason="evaluator unavailable; approved by default")


@ontology_capability.tool
async def list_vertex_types(ctx: RunContext[GraphDependencies]) -> list[VertexTypeInfo]:
    """List the current ontology — every existing type with its usage note and property names.

    Call this BEFORE proposing a new type, so you can reuse an existing generic type instead of
    creating a duplicate.
    """
    rows = await repo.list_vertex_types(ctx.deps.db)
    return [
        VertexTypeInfo(name=r["name"], usage=r.get("usage"), properties=r.get("properties", []))
        for r in rows
        if r.get("name")
    ]


@ontology_capability.tool
async def propose_schema_change(
    ctx: RunContext[GraphDependencies], args: ProposeSchemaArgs
) -> SchemaProposal:
    """Propose a new GENERIC vertex type. Cognitive layer only — does NOT modify the database.

    Pydantic has already enforced PascalCase + valid property types. An evaluator sub-agent then
    judges the proposal semantically against the existing ontology. If approved, the proposal is
    recorded in run-scoped state so create_vertex_type can later execute it. If rejected, it is NOT
    recorded (so create_vertex_type stays blocked) and the returned guidance steers you toward the
    right existing type or a revision.
    """
    verdict = await _evaluate_proposal(
        ctx,
        kind="vertex type",
        name=args.node_name,
        usage=args.usage,
        rationale=args.rationale,
        properties=args.properties,
    )
    if not verdict.approved:
        ctx.deps.proposed_schemas.pop(args.node_name, None)  # clear any stale prior approval
        return SchemaProposal(
            approved=False,
            node_name=args.node_name,
            usage=args.usage,
            properties=args.properties,
            suggested_existing_type=verdict.suggested_existing_type,
            guidance=_rejection_guidance("create_vertex_type", verdict),
        )
    proposal = SchemaProposal(
        approved=True,
        node_name=args.node_name,
        usage=args.usage,
        properties=args.properties,
        guidance=(
            f"Approved. To create it, call create_vertex_type(node_name='{args.node_name}'). "
            "If a suitable generic type already exists, store your data there instead."
        ),
    )
    ctx.deps.proposed_schemas[args.node_name] = proposal
    return proposal


@ontology_capability.tool
async def create_vertex_type(ctx: RunContext[GraphDependencies], node_name: str) -> str:
    """Execute a previously-approved proposal: create the vertex type in THIS user's database.

    Reads the canonical approved proposal from run-scoped state (so the agent cannot smuggle in
    unapproved properties), checks existence, then creates the type + properties if missing.
    """
    proposal = ctx.deps.proposed_schemas.get(node_name)
    if proposal is None or not proposal.approved:
        # Defense-in-depth; the guard hook normally catches this first.
        raise ModelRetry(
            f"No approved proposal for '{node_name}'. Call propose_schema_change first."
        )
    props = {p.name: p.type for p in proposal.properties}
    newly_created = await repo.create_vertex_type(
        ctx.deps.db, node_name, usage=proposal.usage, properties=props
    )
    verb = "Created" if newly_created else "Confirmed existing"
    return f"{verb} vertex type '{node_name}' with {len(props)} propert(y/ies) for this user."


@ontology_capability.tool
async def create_node(ctx: RunContext[GraphDependencies], args: CreateNodeArgs) -> str:
    """Create an INSTANCE (a node/record) of an existing generic vertex type, linked to the user.

    Use this to actually store data (e.g. the framework 'Django' in the 'SoftwareFramework' type).
    The type must already exist — create it first via propose_schema_change + create_vertex_type,
    or pick one from list_vertex_types.
    """
    if not await repo.vertex_type_exists(ctx.deps.db, args.node_type):
        raise ModelRetry(
            f"Vertex type '{args.node_type}' does not exist yet. Create it first via "
            "propose_schema_change then create_vertex_type, or choose an existing type "
            "from list_vertex_types."
        )
    rid = await repo.create_node(ctx.deps.db, ctx.deps.user_id, args.node_type, args.properties)
    return f"Created {args.node_type} node ({rid}) with {len(args.properties)} propert(y/ies) for this user."


@ontology_capability.tool
async def propose_edge_type(
    ctx: RunContext[GraphDependencies], args: ProposeEdgeArgs
) -> EdgeProposal:
    """Propose a new GENERIC relationship (edge) type. Cognitive layer only — does NOT modify the DB.

    Pydantic has enforced UPPER_SNAKE_CASE + valid property types. An evaluator sub-agent then judges
    the proposal semantically against the existing ontology. If approved, it is recorded in run-scoped
    state so create_edge_type can execute it. If rejected, it is NOT recorded (so create_edge_type
    stays blocked) and the guidance steers you toward the right existing relationship or a revision.
    """
    verdict = await _evaluate_proposal(
        ctx,
        kind="edge (relationship) type",
        name=args.edge_name,
        usage=args.usage,
        rationale=args.rationale,
        properties=args.properties,
    )
    if not verdict.approved:
        ctx.deps.proposed_edges.pop(args.edge_name, None)  # clear any stale prior approval
        return EdgeProposal(
            approved=False,
            edge_name=args.edge_name,
            usage=args.usage,
            properties=args.properties,
            suggested_existing_type=verdict.suggested_existing_type,
            guidance=_rejection_guidance("create_edge_type", verdict),
        )
    proposal = EdgeProposal(
        approved=True,
        edge_name=args.edge_name,
        usage=args.usage,
        properties=args.properties,
        guidance=(
            f"Approved. To create it, call create_edge_type(edge_name='{args.edge_name}'), then "
            "connect two instances with create_edge(from_rid, to_rid)."
        ),
    )
    ctx.deps.proposed_edges[args.edge_name] = proposal
    return proposal


@ontology_capability.tool
async def create_edge_type(ctx: RunContext[GraphDependencies], edge_name: str) -> str:
    """Execute a previously-approved edge proposal: create the relationship type in this database."""
    proposal = ctx.deps.proposed_edges.get(edge_name)
    if proposal is None or not proposal.approved:
        raise ModelRetry(
            f"No approved proposal for edge '{edge_name}'. Call propose_edge_type first."
        )
    props = {p.name: p.type for p in proposal.properties}
    newly_created = await repo.create_edge_type(
        ctx.deps.db, edge_name, usage=proposal.usage, properties=props
    )
    verb = "Created" if newly_created else "Confirmed existing"
    return f"{verb} edge type '{edge_name}' with {len(props)} propert(y/ies)."


@ontology_capability.tool
async def create_edge(ctx: RunContext[GraphDependencies], args: CreateEdgeArgs) -> str:
    """Create a relationship between two existing instance nodes, identified by record id.

    The edge type must already exist (propose_edge_type + create_edge_type), and both endpoints
    must exist. Get record ids from create_node's return value or a run_query that selects @rid.
    """
    db = ctx.deps.db
    if not await repo.vertex_type_exists(db, args.edge_type):
        raise ModelRetry(
            f"Edge type '{args.edge_type}' does not exist yet. Create it first via "
            "propose_edge_type then create_edge_type."
        )
    for label, rid in (("from_rid", args.from_rid), ("to_rid", args.to_rid)):
        if not await repo.node_exists(db, rid):
            raise ModelRetry(
                f"{label} {rid!r} does not match an existing node. Create the node first "
                "(create_node) or look up a valid record id with run_query."
            )
    edge_rid = await repo.create_edge(db, args.edge_type, args.from_rid, args.to_rid, args.properties)
    return f"Created {args.edge_type} edge ({edge_rid}) from {args.from_rid} to {args.to_rid}."


async def _resolve_editable_node(ctx: RunContext[GraphDependencies], rid: str) -> str:
    """Confirm ``rid`` is an existing, agent-editable instance node; raise ModelRetry otherwise."""
    node_type = await repo.node_type(ctx.deps.db, rid)
    if node_type is None:
        raise ModelRetry(
            f"No node found at {rid!r}. Look up a valid record id with run_query (SELECT @rid ...)."
        )
    if node_type in _PROTECTED_VERTEX_TYPES:
        raise ModelRetry(
            f"{rid!r} is an internal '{node_type}' record and can't be changed here. "
            "Use update_fact/delete_fact for facts; conversation history is managed automatically."
        )
    return node_type


@ontology_capability.tool
async def update_node(ctx: RunContext[GraphDependencies], args: UpdateNodeArgs) -> str:
    """Revise an existing instance node's properties in place (use its @rid) instead of duplicating it."""
    node_type = await _resolve_editable_node(ctx, args.rid)
    updated = await repo.update_node(ctx.deps.db, ctx.deps.user_id, args.rid, args.properties)
    if not updated:
        raise ModelRetry(f"Node {args.rid!r} was not updated (not owned by this user?).")
    return f"Updated {node_type} node {args.rid} ({len(args.properties)} propert(y/ies))."


@ontology_capability.tool
async def delete_node(ctx: RunContext[GraphDependencies], rid: str) -> str:
    """Delete a redundant/obsolete instance node (and its edges) by its @rid."""
    node_type = await _resolve_editable_node(ctx, rid)
    deleted = await repo.delete_node(ctx.deps.db, ctx.deps.user_id, rid)
    if not deleted:
        raise ModelRetry(f"Node {rid!r} was not deleted (not owned by this user?).")
    return f"Deleted {node_type} node {rid}."


@ontology_capability.tool
async def delete_vertex_type(ctx: RunContext[GraphDependencies], args: DropVertexTypeArgs) -> str:
    """DESTRUCTIVE: drop a vertex TYPE and ALL its instances (and their edges) from this user's DB.

    Use to retire a type you created that is no longer needed (e.g. a wrong/duplicate category). To
    remove a single record instead, use delete_node. Internal types (User, Conversation, Message,
    Fact, LogEntry, RunMessages) cannot be dropped.
    """
    name = args.node_name
    if name in _PROTECTED_VERTEX_TYPES:
        raise ModelRetry(f"'{name}' is an internal vertex type and cannot be dropped.")
    category = await repo.type_category(ctx.deps.db, name)
    if category is None:
        raise ModelRetry(
            f"No type named '{name}' exists. Call list_vertex_types to see what exists."
        )
    if category != "vertex":
        raise ModelRetry(
            f"'{name}' is a {category} type, not a vertex type. Use delete_edge_type for relationships."
        )
    removed = await repo.drop_vertex_type(ctx.deps.db, name)
    return f"Dropped vertex type '{name}' and its {removed} instance(s)."


@ontology_capability.tool
async def delete_edge_type(ctx: RunContext[GraphDependencies], args: DropEdgeTypeArgs) -> str:
    """DESTRUCTIVE: drop an edge (relationship) TYPE and ALL its edges from this user's DB.

    The connected nodes are kept — only the relationships of this type are removed. Use to retire a
    relationship type you created that is no longer needed. Internal relationship types
    (HAS_CONVERSATION, HAS_MESSAGE, HAS_RUN_MESSAGES, KNOWS, LOGGED, HAS_NODE) cannot be dropped.
    """
    name = args.edge_name
    if name in _PROTECTED_EDGE_TYPES:
        raise ModelRetry(f"'{name}' is an internal relationship type and cannot be dropped.")
    category = await repo.type_category(ctx.deps.db, name)
    if category is None:
        raise ModelRetry(
            f"No type named '{name}' exists. Call list_vertex_types to see what exists."
        )
    if category != "edge":
        raise ModelRetry(
            f"'{name}' is a {category} type, not an edge type. Use delete_vertex_type for node types."
        )
    removed = await repo.drop_edge_type(ctx.deps.db, name)
    return f"Dropped edge type '{name}' and its {removed} edge(s)."


# --------------------------------------------------------------------------- #
# Technical ordering guard
# --------------------------------------------------------------------------- #
ontology_guard = Hooks()


@ontology_guard.on.before_tool_execute(tools=["create_vertex_type", "create_edge_type"])
async def _require_prior_proposal(ctx: RunContext[GraphDependencies], *, call: ToolCallPart, tool_def, args):
    """Reject create_*_type unless the matching propose_* ran for that name earlier this run."""
    data = call.args_as_dict() or {}
    if call.tool_name == "create_vertex_type":
        name, store, proposer = data.get("node_name"), ctx.deps.proposed_schemas, "propose_schema_change"
    else:
        name, store, proposer = data.get("edge_name"), ctx.deps.proposed_edges, "propose_edge_type"
    if name not in store:
        raise ModelRetry(
            f"{call.tool_name}({name!r}) blocked: no approved proposal this run. "
            f"You must call {proposer} first."
        )
    return args


def build_ontology() -> list[Capability | Hooks]:
    """Capabilities to add to ``Agent(capabilities=...)``: the tools + the ordering guard."""
    return [ontology_capability, ontology_guard]

"""Shared sub-agent runner for delegated work (swarm dispatches and deep research).

A sub-agent is a fresh, single-purpose Pydantic AI agent built per dispatch: its own system
prompt, a granted subset of the tool bundles (see :data:`TOOL_GROUP_BUILDERS`), and the parent
run's dependencies (same per-user database, conversation, web client, embedder, sandbox) — so
everything a sub-agent persists (facts, documents, sandbox artifacts) lands in the same place the
main agent's work does. Sub-agents do NOT get the persistence hooks: the parent run's
``after_run`` already records the orchestrating turn (the sub-agent's report rides back on the
tool result), so adding hooks here would double-write messages.

Tolerance contract (same as run_query/web_search/run_python): :func:`run_subagent` never raises
for expected failures — a model error, tool blow-up, or exhausted usage limit comes back as an
``error`` on the outcome, so one broken delegate can never abort the orchestrator's run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, Hooks
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from backend.db.dependencies import GraphDependencies
from backend.model_selection import resolve_model
from backend.schemas.document_schemas import DocumentInfo

logger = logging.getLogger("agent_graph.subagent")

# Ceiling on a sub-agent's model round-trips (a runaway-loop backstop, not a quality knob).
DEFAULT_REQUEST_LIMIT = int(os.getenv("SUBAGENT_REQUEST_LIMIT", "25"))


def _tool_group_builders() -> dict[str, Callable[[], list[Capability]]]:
    """The capability bundle granted by each tool-group name (see swarm_schemas.TOOL_GROUPS).

    Imported lazily so this module stays a leaf: the capability modules import schemas/repo, and
    the swarm capability imports this module. ``memory`` grants only the tool capability — NOT
    ``persistence_hooks`` — because sub-agent turns must not be persisted as conversation turns.
    """
    from backend.skills.document_capability import documents_capability
    from backend.skills.graph_capability import memory_capability
    from backend.skills.sandbox_capability import sandbox_capability
    from backend.skills.search_capability import web_capability

    return {
        "web": lambda: [web_capability],
        "documents": lambda: [documents_capability],
        "sandbox": lambda: [sandbox_capability],
        "memory": lambda: [memory_capability],
    }


def capabilities_for(tool_groups: list[str]) -> list[Capability]:
    """Resolve granted tool-group names to capability objects; unknown names are skipped.

    Unknown entries can only come from hand-edited DB rows (the schemas validate tool lists), so
    they are logged and ignored rather than failing the dispatch.
    """
    builders = _tool_group_builders()
    capabilities: list[Capability] = []
    for group in tool_groups:
        builder = builders.get(group)
        if builder is None:
            logger.warning("unknown tool group %r on a sub-agent spec; skipping", group)
            continue
        capabilities.extend(builder())
    return capabilities


def _document_collector(bucket: list[DocumentInfo]) -> Hooks:
    """Hooks that record every document a sub-agent persists, for the outcome's ``documents``.

    The main stream's ``_document_events`` only sees the orchestrator's own tool results, so
    documents created *inside* a delegated run would be invisible to the UI without this — the
    swarm tools surface them on their results, and ``stream_run`` turns those into the same
    artifact-card frames a direct ``create_document`` gets.
    """
    hooks = Hooks()

    @hooks.on.after_tool_execute
    async def _collect(
        ctx: Any, *, call: ToolCallPart, tool_def: Any, args: Any, result: Any
    ) -> Any:
        if call.tool_name == "create_document" and isinstance(result, DocumentInfo):
            bucket.append(result)
        elif call.tool_name == "run_python":
            docs = getattr(result, "documents", None) or []
            bucket.extend(d for d in docs if isinstance(d, DocumentInfo))
        return result

    return hooks


@dataclass
class SubagentOutcome:
    """What a delegated run produced: the report text, persisted documents, or an error."""

    output: str = ""
    documents: list[DocumentInfo] = field(default_factory=list)
    error: str | None = None


async def run_subagent(
    deps: GraphDependencies,
    *,
    instructions: str,
    tool_groups: list[str],
    prompt: str,
    request_limit: int | None = None,
    model: Model | str | None = None,
) -> SubagentOutcome:
    """Run one delegated task to completion and return its outcome. Never raises.

    The sub-agent shares the parent's dependencies (fresh run-scoped dicts, no ontology state
    leaks) and runs on the conversation's model (``deps.model``) unless ``model`` overrides it
    (the test seam). Documents it creates are collected onto the outcome so the caller can
    reference them instead of recreating them.
    """
    documents: list[DocumentInfo] = []
    sub_deps = replace(deps, proposed_schemas={}, proposed_edges={})
    agent: Agent[GraphDependencies, str] = Agent(
        model if model is not None else resolve_model(deps.model),
        deps_type=GraphDependencies,
        instructions=instructions,
        capabilities=[*capabilities_for(tool_groups), _document_collector(documents)],
    )
    try:
        result = await agent.run(
            prompt,
            deps=sub_deps,
            usage_limits=UsageLimits(request_limit=request_limit or DEFAULT_REQUEST_LIMIT),
        )
    except Exception as exc:  # noqa: BLE001 — a broken delegate must never abort the orchestrator.
        logger.warning("sub-agent run failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        return SubagentOutcome(
            documents=documents,  # whatever it persisted before failing is still real
            error=f"{type(exc).__name__}: {exc}",
        )
    return SubagentOutcome(output=result.output, documents=documents)

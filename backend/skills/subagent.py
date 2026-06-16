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
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from backend.db.dependencies import GraphDependencies
from backend.model_selection import resolve_model
from backend.schemas.document_schemas import DocumentInfo
from backend.serialization import _jsonable

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
    agent_id: str = "",
    agent_name: str = "",
    instance_id: str = "",
) -> SubagentOutcome:
    """Run one delegated task to completion and return its outcome. Never raises.

    The sub-agent shares the parent's dependencies (fresh run-scoped dicts, no ontology state
    leaks) and runs on the conversation's model (``deps.model``) unless ``model`` overrides it
    (the test seam). Documents it creates are collected onto the outcome so the caller can
    reference them instead of recreating them.

    When ``deps.event_sink`` is set, the delegate's work is *streamed* and each event is pushed
    onto the sink as a frame tagged with ``agent_id``/``name``/``instance_id`` (plus
    ``agent_start``/``agent_end`` lifecycle frames), so ``stream_run`` can render the sub-agent's
    thinking/tool calls live and coloured. Otherwise the legacy blocking ``agent.run`` path is
    used (CLI / tests). Either way this never raises — a broken delegate becomes
    ``SubagentOutcome.error``.
    """
    documents: list[DocumentInfo] = []
    sub_deps = replace(deps, proposed_schemas={}, proposed_edges={})
    agent: Agent[GraphDependencies, str] = Agent(
        model if model is not None else resolve_model(deps.model),
        deps_type=GraphDependencies,
        instructions=instructions,
        capabilities=[*capabilities_for(tool_groups), _document_collector(documents)],
    )
    limits = UsageLimits(request_limit=request_limit or DEFAULT_REQUEST_LIMIT)

    if deps.event_sink is None:
        # Legacy blocking path: no live trace channel (CLI / non-swarm / tests).
        try:
            result = await agent.run(prompt, deps=sub_deps, usage_limits=limits)
        except Exception as exc:  # noqa: BLE001 — a broken delegate must never abort the orchestrator.
            logger.warning(
                "sub-agent run failed: %s: %s", type(exc).__name__, exc, exc_info=True
            )
            return SubagentOutcome(
                documents=documents,  # whatever it persisted before failing is still real
                error=f"{type(exc).__name__}: {exc}",
            )
        return SubagentOutcome(output=result.output, documents=documents)

    # Streaming path: relay this delegate's events onto the shared sink, tagged with its identity.
    sink = deps.event_sink

    def emit(frame: dict[str, Any]) -> None:
        # Pushing to the sink must never abort the delegate (tolerance contract). An unbounded
        # asyncio.Queue.put_nowait won't block; we still guard against any unexpected failure.
        try:
            sink.put_nowait({**frame, "agent_id": agent_id, "name": agent_name,
                             "instance_id": instance_id})
        except Exception:  # noqa: BLE001 — a sink hiccup can't be allowed to break the run.
            logger.warning("sub-agent event_sink push failed; dropping frame", exc_info=True)

    output = ""
    emit({"type": "agent_start"})
    try:
        async with agent.run_stream_events(
            prompt, deps=sub_deps, usage_limits=limits
        ) as stream:
            async for event in stream:
                if isinstance(event, FunctionToolCallEvent):
                    emit({
                        "type": "tool_call",
                        "tool_name": event.part.tool_name,
                        # Namespace by instance so the UI's tool_result->tool_call match never
                        # collides with the orchestrator or a sibling running the same spec.
                        "tool_call_id": f"{instance_id}:{event.part.tool_call_id}",
                        "args": _jsonable(event.part.args),
                    })
                    continue
                if isinstance(event, FunctionToolResultEvent):
                    part = event.part
                    emit({
                        "type": "tool_result",
                        "tool_name": getattr(part, "tool_name", None),
                        "tool_call_id": f"{instance_id}:{getattr(part, 'tool_call_id', '')}",
                        "content": _jsonable(getattr(part, "content", None)),
                    })
                    # Documents are NOT emitted here — they ride back on the report and the
                    # orchestrator's _document_events path turns them into artifact cards
                    # (emitting them here too would double the cards).
                    continue

                node = event
                if isinstance(event, PartStartEvent):
                    node = event.part
                elif isinstance(event, PartDeltaEvent):
                    node = event.delta

                if isinstance(node, ThinkingPartDelta):
                    if node.content_delta:
                        emit({"type": "thinking", "delta": node.content_delta})
                elif isinstance(node, TextPart):
                    output += node.content
                    emit({"type": "text", "delta": node.content})
                elif isinstance(node, TextPartDelta):
                    output += node.content_delta
                    emit({"type": "text", "delta": node.content_delta})
    except Exception as exc:  # noqa: BLE001 — a broken delegate must never abort the orchestrator.
        logger.warning("sub-agent run failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        return SubagentOutcome(
            documents=documents,  # whatever it persisted before failing is still real
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        emit({"type": "agent_end"})
    return SubagentOutcome(output=output, documents=documents)

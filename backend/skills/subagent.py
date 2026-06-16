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
from uuid import uuid4

from pydantic_ai import Agent, RunContext
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

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.model_selection import resolve_model
from backend.reasoning_split import ReasoningSplitter
from backend.schemas.document_schemas import DocumentInfo
from backend.schemas.swarm_schemas import AgentRunReport, SendMessageArgs
from backend.serialization import _jsonable

logger = logging.getLogger("agent_graph.subagent")

# Ceiling on a sub-agent's model round-trips (a runaway-loop backstop, not a quality knob).
DEFAULT_REQUEST_LIMIT = int(os.getenv("SUBAGENT_REQUEST_LIMIT", "25"))

# How many hops deep the agency communication chart may recurse. The entry-point orchestrator is
# at depth 0; each send_message delegation is one hop. A specialist is granted its own
# send_message tool only while its depth stays under this ceiling, so multi-hop flows
# (orchestrator -> A -> B -> ...) can't recurse without bound. Caps depth, not breadth.
SWARM_MAX_DEPTH = int(os.getenv("SWARM_MAX_DEPTH", "3"))


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


def _can_delegate(depth: int, recipients: list[str] | None) -> bool:
    """Whether an agent at ``depth`` with these chart edges may be granted its own send_message.

    The child of this dispatch runs at ``depth + 1``; it may delegate further only while that
    stays under :data:`SWARM_MAX_DEPTH` and it actually has teammates to message.
    """
    return bool(recipients) and (depth + 1) < SWARM_MAX_DEPTH


def _subagent_system_prompt(row: dict, can_delegate: bool) -> str:
    """The dispatched agent's system prompt: its spec, framed as a member of the agency.

    When ``can_delegate`` the prompt lists the teammates it may ``send_message`` (its chart edges),
    matching the send_message tool it is actually granted.
    """
    recipients = list(row.get("recipients") or [])
    prompt = (
        f"You are '{row.get('name')}', a specialist agent in an agency.\n"
        f"ROLE: {row.get('role') or 'specialist'}\n"
        f"{row.get('instructions') or ''}\n"
        "A teammate dispatched you with ONE assignment. You cannot see the wider conversation — "
        "work only from the assignment you were given. Do it completely, then reply with a focused "
        "report of what you did and found (your reply goes to the agent that messaged you, not the "
        "end user). If you created documents, name them in the report instead of pasting their "
        "full content."
    )
    if can_delegate and recipients:
        prompt += (
            "\nYou may delegate parts of this work to your teammates via `send_message`: "
            f"{', '.join(recipients)}. Give each a self-contained assignment (they cannot see your "
            "context) and fold their reports into your own."
        )
    return prompt


async def dispatch_message(
    deps: GraphDependencies,
    recipient: str,
    message: str,
    context: str | None = None,
) -> AgentRunReport:
    """Deliver one message to a teammate and return its report. Tolerant: never raises.

    This is the shared agency ``send_message`` mechanism, used by the orchestrator's
    ``send_message``/``send_messages`` tools and by every specialist that is granted the
    communication capability. It enforces the caller's communication chart
    (``deps.agency_recipients``: ``None`` = the entry point, which may message anyone; otherwise
    the recipient must be on the list), looks up the recipient spec (a bad name becomes an error
    report, not an exception), and runs it via :func:`run_subagent` one hop deeper.
    """
    try:
        row = await repo.get_agent_spec(deps.db, deps.user_id, recipient)
    except Exception as exc:  # noqa: BLE001 — a DB hiccup must not abort a parallel batch.
        logger.warning("agent lookup failed for %r: %s", recipient, exc, exc_info=True)
        return AgentRunReport(
            agent_id=recipient, task=message, error=f"Agent lookup failed: {exc}"
        )
    if row is None:
        return AgentRunReport(
            agent_id=recipient,
            task=message,
            error=(
                f"No agent named or with id {recipient!r}. Call list_agents to see the roster, "
                "or create_agent to define this specialist first."
            ),
        )
    # Chart enforcement: a dispatched specialist (agency_recipients set) may only message the
    # teammates its spec declares. The entry-point orchestrator (None) may message anyone.
    if deps.agency_recipients is not None and row.get("name") not in deps.agency_recipients:
        allowed = ", ".join(deps.agency_recipients) or "(none)"
        return AgentRunReport(
            agent_id=row.get("agent_id", recipient),
            name=row.get("name") or "",
            task=message,
            error=(
                f"{row.get('name') or recipient!r} is not in your communication chart. You may "
                f"send_message only to: {allowed}."
            ),
        )
    recipients = list(row.get("recipients") or [])
    can_delegate = _can_delegate(deps.agency_depth, recipients)
    prompt = message
    if context:
        prompt += f"\n\nCONTEXT FROM THE SENDER:\n{context}"
    outcome = await run_subagent(
        deps,
        instructions=_subagent_system_prompt(row, can_delegate),
        tool_groups=list(row.get("tools") or []),
        prompt=prompt,
        recipients=recipients,
        agent_id=row.get("agent_id", recipient),
        agent_name=row.get("name") or recipient,
        instance_id=uuid4().hex[:8],
    )
    return AgentRunReport(
        agent_id=row.get("agent_id", recipient),
        name=row.get("name") or "",
        task=message,
        output=outcome.output,
        documents=outcome.documents,
        error=outcome.error,
    )


_SEND_MESSAGE_INSTRUCTIONS = (
    "You are part of an agency and may consult teammates with `send_message`: give a teammate a "
    "self-contained assignment (they cannot see your context — include everything they need) and "
    "use their report. You can only message the teammates named in your instructions; an "
    "out-of-chart or failed message comes back as a report with an `error` you should handle."
)


def build_communication_capability() -> Capability:
    """A capability granting a dispatched specialist its own ``send_message`` tool (multi-hop).

    Added by :func:`run_subagent` to a delegate that has chart edges and is within the depth
    ceiling, so flows can recurse along the agency chart. The orchestrator gets its richer
    send_message/send_messages tools from the swarm capability instead.
    """
    capability = Capability(id="AgencyComms", instructions=_SEND_MESSAGE_INSTRUCTIONS)

    @capability.tool
    async def send_message(
        ctx: RunContext[GraphDependencies], args: SendMessageArgs
    ) -> AgentRunReport:
        """Send one self-contained assignment to a teammate agent and get its report back."""
        return await dispatch_message(ctx.deps, args.recipient, args.message, args.context)

    return capability


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
    recipients: list[str] | None = None,
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

    ``recipients`` are this delegate's agency communication-chart edges. When it has any and the
    depth ceiling (:data:`SWARM_MAX_DEPTH`) is not yet reached, it is granted its own
    ``send_message`` tool so flows can recurse one hop further; its dependencies are stamped with
    the chart position (``agency_recipients``/``agency_depth``) so its own dispatches are enforced
    and bounded.

    When ``deps.event_sink`` is set, the delegate's work is *streamed* and each event is pushed
    onto the sink as a frame tagged with ``agent_id``/``name``/``instance_id`` (plus
    ``agent_start``/``agent_end`` lifecycle frames), so ``stream_run`` can render the sub-agent's
    thinking/tool calls live and coloured. Otherwise the legacy blocking ``agent.run`` path is
    used (CLI / tests). Either way this never raises — a broken delegate becomes
    ``SubagentOutcome.error``.
    """
    documents: list[DocumentInfo] = []
    sub_deps = replace(
        deps,
        proposed_schemas={},
        proposed_edges={},
        agency_recipients=recipients,
        agency_depth=deps.agency_depth + 1,
    )
    comms = [build_communication_capability()] if _can_delegate(deps.agency_depth, recipients) else []
    agent: Agent[GraphDependencies, str] = Agent(
        model if model is not None else resolve_model(deps.model),
        deps_type=GraphDependencies,
        instructions=instructions,
        capabilities=[*capabilities_for(tool_groups), *comms, _document_collector(documents)],
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
    # Repair reasoning leaked across channels via literal <think>/</think> tags (e.g. Ollama
    # qwen3), so a delegate's report text is the answer only, not its trapped chain-of-thought.
    splitter = ReasoningSplitter()

    def route(channel: str, text: str) -> None:
        nonlocal output
        if not text:
            return
        if channel == "text":
            output += text
        emit({"type": channel, "delta": text})

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
                        for channel, text in splitter.feed_thinking(node.content_delta):
                            route(channel, text)
                elif isinstance(node, TextPart):
                    for channel, text in splitter.feed_text(node.content):
                        route(channel, text)
                elif isinstance(node, TextPartDelta):
                    for channel, text in splitter.feed_text(node.content_delta):
                        route(channel, text)
        # Release any partial-tag tail the splitter held back at the very end.
        for channel, text in splitter.flush():
            route(channel, text)
    except Exception as exc:  # noqa: BLE001 — a broken delegate must never abort the orchestrator.
        logger.warning("sub-agent run failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        return SubagentOutcome(
            documents=documents,  # whatever it persisted before failing is still real
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        emit({"type": "agent_end"})
    return SubagentOutcome(output=output, documents=documents)

"""Entry point: wire the ArcadeDB-backed conversation-memory agent together.

Run a single turn from the command line:

    python -m backend.main "remember I like Recoleta apartments" --user u1 --conversation c1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
)

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient, database_name_for_user
from backend.db.dependencies import GraphDependencies
from backend.embeddings import Embedder
from backend.model_selection import resolve_model
from backend.serialization import _jsonable
from backend.skills.document_capability import build_documents
from backend.skills.graph_capability import build_memory
from backend.skills.ontology_capability import build_ontology
from backend.skills.research_capability import build_research
from backend.skills.sandbox_capability import build_sandbox
from backend.skills.search_capability import build_search
from backend.skills.swarm_capability import build_swarm
from backend.skills.system_prompt import BASE_SYSTEM_PROMPT, register_system_prompt
from backend.web.client import WebClient

logger = logging.getLogger("agent_graph.main")

# Selectable thinking-effort levels for the UI (the named subset of Pydantic AI's ThinkingLevel).
# DEFAULT_EFFORT matches the original hardcoded value, so an unset/invalid request is unchanged.
THINKING_EFFORTS = ["minimal", "low", "medium", "high", "xhigh"]
DEFAULT_EFFORT = "minimal"

# Conversation modes (agent profiles), fixed per conversation at creation. "regular" is the base
# memory agent; "research" overlays the deep-research method; "swarm" adds the orchestrator tools.
MODES = ["regular", "research", "swarm"]
DEFAULT_MODE = "regular"

# Drain-loop terminator for stream_run: the background parent task pushes this last so the
# consumer knows no more frames are coming. Identity-compared, never serialized.
_STREAM_SENTINEL: Any = object()


def compose_instructions(system_prompt: str | None = None) -> str:
    """The agent's static system prompt: :data:`BASE_SYSTEM_PROMPT` plus the user's custom prompt.

    A conversation can carry its own extra instructions (set from the web UI). When present they are
    *appended* under a clear header so the agent keeps its base memory/tool/honesty rules and treats
    the user's text as additional guidance. An empty/missing prompt returns the base unchanged.
    """
    custom = (system_prompt or "").strip()
    if not custom:
        return BASE_SYSTEM_PROMPT
    return f"{BASE_SYSTEM_PROMPT}\n\nADDITIONAL INSTRUCTIONS (from the user):\n{custom}"


def build_agent(
    model: str | None = None,
    effort: str | None = None,
    mode: str | None = None,
    system_prompt: str | None = None,
) -> Agent[GraphDependencies, str]:
    """Construct the agent.

    ``model`` is an optional explicit model label (from the UI dropdown) — see
    :func:`backend.model_selection.resolve_model`. When omitted, model selection falls back to
    env: ``AGENT_MODEL`` (any Pydantic AI model string, e.g. ``openai:gpt-5.2``), else a local
    Ollama model named by ``OLLAMA_MODEL`` (mirrors the original notebook prototype).

    ``effort`` is an optional thinking-effort level (one of :data:`THINKING_EFFORTS`); an unknown
    or missing value falls back to :data:`DEFAULT_EFFORT`.

    ``mode`` is the conversation's agent profile (one of :data:`MODES`). ``regular`` and
    ``research`` keep the full base capability set (``research`` adds the deep-research
    methodology). ``swarm`` is different: the orchestrator is a **pure router** — it gets ONLY
    memory (incl. the persistence hooks) and the swarm roster/communication tools, NOT the
    "doing" tools (web/sandbox/ontology/documents). It can't browse or run code itself; it must
    delegate work to specialist sub-agents (which get their own tools via ``run_subagent``).
    Unknown/missing values fall back to :data:`DEFAULT_MODE`.

    ``system_prompt`` is the conversation's optional custom prompt, appended to
    :data:`BASE_SYSTEM_PROMPT` (see :func:`compose_instructions`). Main agent only — delegated
    sub-agents keep their own task-specific prompts.
    """
    effort = effort if effort in THINKING_EFFORTS else DEFAULT_EFFORT
    mode = mode if mode in MODES else DEFAULT_MODE
    if mode == "swarm":
        # Pure orchestrator: no web/sandbox/ontology/document tools, so it cannot do the work
        # itself — it can only manage the roster, delegate via send_message/send_messages, run
        # deep_research (its own sub-agent), and use memory. build_memory() also carries the
        # persistence hooks that save the turn, so it must stay.
        capabilities = [Thinking(effort=effort), *build_memory(), *build_swarm()]
    else:
        capabilities = [
            Thinking(effort=effort),
            *build_memory(),
            *build_ontology(),
            *build_search(),
            *build_documents(),
            *build_sandbox(),
        ]
        if mode == "research":
            capabilities += build_research()
    agent = Agent(
        resolve_model(model),
        deps_type=GraphDependencies,
        instructions=compose_instructions(system_prompt),
        capabilities=capabilities,
    )
    # Auto-load the user's relevant stored facts into the system prompt each run (and the date).
    register_system_prompt(agent)
    return agent


def _to_message_history(rows: list[dict[str, Any]]) -> list[ModelMessage]:
    """Rebuild Pydantic AI message history from stored per-run serialized blobs.

    Each row's ``raw`` is one run's ``new_messages_json()`` (see
    :func:`repo.append_run_messages`); deserializing and concatenating them oldest-first
    reconstructs the conversation *faithfully* — tool calls and their returns included — so the
    model sees what it actually did, not just its own text. This is what stops the agent
    re-doubting/redoing tool work it already completed. ``instructions`` are re-applied each run
    regardless of history. A corrupt blob is skipped rather than failing the whole run.
    """
    history: list[ModelMessage] = []
    for row in rows:
        raw = row.get("raw")
        if not raw:
            continue
        try:
            history.extend(ModelMessagesTypeAdapter.validate_json(raw))
        except Exception:  # noqa: BLE001 — a single bad blob must not blind the agent to the rest.
            logger.warning("skipping unparseable run-message blob", exc_info=True)
    return history


def _created_frame(info: Any) -> dict[str, Any] | None:
    """A ``created`` document frame from a DocumentInfo-shaped value (model or dict), or None."""
    get = info.get if isinstance(info, dict) else lambda k: getattr(info, k, None)
    document_id = get("document_id")
    if not document_id:
        return None
    return {
        "type": "document",
        "action": "created",
        "document_id": str(document_id),
        "title": str(get("title") or ""),
        "mime_type": str(get("mime_type") or "text/markdown"),
    }


def _documents_of(value: Any) -> list[Any]:
    """The ``documents`` list of a result (model or dict), or ``[]``."""
    docs = value.get("documents") if isinstance(value, dict) else getattr(value, "documents", None)
    return list(docs or [])


# Tools whose result carries a ``documents`` list of artifacts persisted during the call:
# run_python's /out files, and the documents a delegated specialist created (send_message /
# deep_research outcomes; send_messages nests one such report per message).
_DOCUMENT_BEARING_TOOLS = frozenset({"run_python", "send_message", "deep_research"})


def _document_events(
    tool_name: str | None, content: Any, args: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Map a finished tool call to its ``{"type": "document"}`` frames (usually zero or one).

    The UI uses these frames to drop an artifact card into the chat and flip the side panel to
    the document. ``create_document`` returns a ``DocumentInfo`` (id/title/mime ride on the tool
    result); ``run_python``/``send_message``/``deep_research`` results carry a ``documents`` list
    of everything persisted during the call (one frame each), and ``send_messages`` nests one such
    report per delivered message; ``update_document`` returns a plain string, so the id comes from
    the call's args (which Pydantic AI nests under the ``args`` parameter of the tool signature).
    """
    if tool_name == "create_document":
        frame = _created_frame(content)
        return [frame] if frame else []
    if tool_name in _DOCUMENT_BEARING_TOOLS:
        return [frame for d in _documents_of(content) if (frame := _created_frame(d))]
    if tool_name == "send_messages":
        reports = (
            content.get("reports") if isinstance(content, dict)
            else getattr(content, "reports", None)
        )
        return [
            frame
            for report in reports or []
            for d in _documents_of(report)
            if (frame := _created_frame(d))
        ]
    if tool_name == "update_document":
        inner = (args or {}).get("args")
        data = inner if isinstance(inner, dict) else (args or {})
        document_id = data.get("document_id")
        if not document_id:
            return []
        return [
            {
                "type": "document",
                "action": "updated",
                "document_id": str(document_id),
                "title": str(data.get("title") or ""),
                "mime_type": "",
            }
        ]
    return []


async def stream_run(
    prompt: str,
    user_id: str = "default",
    conversation_id: str = "default",
    model: str | None = None,
    effort: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one turn and yield structured events as they stream from the model.

    This is the single streaming source of truth, consumed by both the CLI (``run``) and the
    HTTP/SSE API (``backend.api``). Pydantic AI's event objects are mapped to a small, stable
    vocabulary so callers never depend on the library's event classes:

    - ``{"type": "thinking", "delta": str}`` — a chunk of the model's reasoning
    - ``{"type": "text", "delta": str}`` — a chunk of the user-facing answer
    - ``{"type": "tool_call", "tool_name", "tool_call_id", "args"}`` — a tool invocation
    - ``{"type": "tool_result", "tool_name", "tool_call_id", "content"}`` — its return
    - ``{"type": "document", "action", "document_id", "title", "mime_type"}`` — emitted right
      after the tool_result of create_document/update_document, so the UI can feature the
      document (artifact card in the chat + side-panel focus)
    - ``{"type": "final", "text": str}`` — the fully assembled answer (always emitted last)
    """
    # Each user gets their own database, so no user's data can surface in another
    # user's queries. The database is created on first use.
    async with (
        ArcadeClient(database=database_name_for_user(user_id)) as db,
        WebClient() as web,
        Embedder.from_env() as embedder,
    ):
        await db.ensure_database()
        await db.ensure_schema()
        # The conversation's mode picks the agent profile, and its custom system prompt (if any)
        # is appended to the base prompt. Both are re-read each turn, so a UI change takes effect
        # next turn. Tolerant: a lookup failure (or a CLI conversation that doesn't exist yet)
        # falls back to the regular agent / no custom prompt rather than blocking the turn.
        try:
            mode = await repo.get_conversation_mode(db, conversation_id)
        except Exception:  # noqa: BLE001 — mode resolution must never block a turn.
            logger.warning("mode lookup failed; using %r", DEFAULT_MODE, exc_info=True)
            mode = DEFAULT_MODE
        try:
            system_prompt = await repo.get_conversation_system_prompt(db, conversation_id)
        except Exception:  # noqa: BLE001 — a missing custom prompt must never block a turn.
            logger.warning("system-prompt lookup failed; using base prompt", exc_info=True)
            system_prompt = ""
        agent = build_agent(model, effort, mode=mode, system_prompt=system_prompt)
        # Live trace channel: the parent run AND every sub-agent (run_subagent) push frames onto
        # this one queue, so swarm sub-agents' thinking/tool calls interleave with the
        # orchestrator's in true arrival order (see the drain loop below).
        sink: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue()
        deps = GraphDependencies(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            web=web,
            embedder=embedder,
            model=model,
            event_sink=sink,
        )

        # Load the prior turns of this thread so the agent retains context. The
        # persistence hooks store each run but never reload it, so without this
        # the model starts every run blind to the conversation. We load the faithful
        # per-run blobs (tool calls + returns), not the lossy role/content text, so the
        # agent sees the tool work it already did. after_run persists only the current
        # run's new_messages(), so reloaded history isn't re-written.
        history = _to_message_history(await repo.get_run_history(db, conversation_id))

        final_text = ""
        # Args of in-flight tool calls, keyed by tool_call_id — _document_event needs the
        # document_id from update_document's *arguments* (its return is just a string).
        call_args: dict[str, dict[str, Any]] = {}

        def _emit_parent(event: Any) -> None:
            """Map one orchestrator event to frames and push them onto the sink (untagged).

            Untagged frames (no ``agent_id``) are the orchestrator's own work; sub-agents push
            their own tagged frames onto the same sink from run_subagent.
            """
            nonlocal final_text
            if isinstance(event, FunctionToolCallEvent):
                call_args[event.part.tool_call_id] = event.part.args_as_dict() or {}
                sink.put_nowait({
                    "type": "tool_call",
                    "tool_name": event.part.tool_name,
                    "tool_call_id": event.part.tool_call_id,
                    "args": _jsonable(event.part.args),
                })
                return
            if isinstance(event, FunctionToolResultEvent):
                part = event.part
                tool_name = getattr(part, "tool_name", None)
                sink.put_nowait({
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "tool_call_id": getattr(part, "tool_call_id", None),
                    "content": _jsonable(getattr(part, "content", None)),
                })
                for doc in _document_events(
                    tool_name,
                    getattr(part, "content", None),
                    call_args.get(getattr(part, "tool_call_id", None) or ""),
                ):
                    sink.put_nowait(doc)
                return

            node = event
            if isinstance(event, PartStartEvent):
                node = event.part
            elif isinstance(event, PartDeltaEvent):
                node = event.delta

            if isinstance(node, ThinkingPartDelta):
                if node.content_delta:
                    sink.put_nowait({"type": "thinking", "delta": node.content_delta})
            elif isinstance(node, TextPart):
                final_text += node.content
                sink.put_nowait({"type": "text", "delta": node.content})
            elif isinstance(node, TextPartDelta):
                final_text += node.content_delta
                sink.put_nowait({"type": "text", "delta": node.content_delta})

        async def _drive() -> None:
            """Run the parent agent, feeding frames onto the sink; always end with the sentinel."""
            try:
                async with agent.run_stream_events(
                    prompt, deps=deps, message_history=history
                ) as stream:
                    async for event in stream:
                        _emit_parent(event)
                sink.put_nowait({"type": "final", "text": final_text})
            finally:
                sink.put_nowait(_STREAM_SENTINEL)

        task = asyncio.create_task(_drive())
        try:
            while True:
                item = await sink.get()
                if item is _STREAM_SENTINEL:
                    break
                yield item
        finally:
            # On a clean finish the task is already done; on early generator close (client
            # disconnect / abort) cancel it so the background run doesn't leak and hold the
            # db/web/embedder contexts (and sandbox containers) open. await re-raises any parent
            # exception so the API layer can emit an {"type": "error"} frame.
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def run(prompt: str, user_id: str = "default", conversation_id: str = "default") -> str:
    """Run one turn, streaming thinking/text to stdout, and return the final output."""
    final_text = ""
    async for event in stream_run(prompt, user_id=user_id, conversation_id=conversation_id):
        kind = event["type"]
        if kind == "thinking":
            print(f"\033[94m{event['delta']}\033[0m", end="", flush=True)
        elif kind == "text":
            print(event["delta"], end="", flush=True)
        elif kind == "final":
            final_text = event["text"]
    print()
    return final_text


def _configure_logging() -> None:
    """Send application logs to stderr. Level via ``LOG_LEVEL`` (default INFO).

    Persistence failures, DB retries and run errors are emitted under the ``agent_graph.*``
    loggers, so a database hiccup shows up as a clear log line instead of a raw traceback.
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Run one turn of the conversation-memory agent.")
    parser.add_argument("prompt")
    parser.add_argument("--user", default="default")
    parser.add_argument("--conversation", default="default")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.prompt, user_id=args.user, conversation_id=args.conversation))
    except Exception as exc:  # noqa: BLE001 — CLI boundary: log cleanly instead of dumping a traceback.
        logging.getLogger("agent_graph").error("Agent run failed: %s: %s", type(exc).__name__, exc)
        logging.getLogger("agent_graph").debug("Full traceback:", exc_info=exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

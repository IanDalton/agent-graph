"""Agent dependencies injected via Pydantic AI's ``deps_type``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.db.arcade_db import ArcadeClient

if TYPE_CHECKING:
    from backend.embeddings import Embedder
    from backend.sandbox.runner import PythonSandbox
    from backend.schemas.graph_schemas import EdgeProposal, SchemaProposal
    from backend.web.client import WebClient


@dataclass
class GraphDependencies:
    """Per-run dependencies for the conversation-memory agent.

    ``user_id`` isolates each user's memory; ``conversation_id`` scopes the
    current thread of messages and logs.
    """

    db: ArcadeClient
    user_id: str
    conversation_id: str
    # Set by propose_schema_change, read by the create_vertex_type guard + tool.
    # Keyed by node_name. Run-scoped: a fresh instance is built per run() call.
    proposed_schemas: dict[str, "SchemaProposal"] = field(default_factory=dict)
    # Same pattern for the edge pipeline: set by propose_edge_type, read by the
    # create_edge_type guard + tool. Keyed by edge_name.
    proposed_edges: dict[str, "EdgeProposal"] = field(default_factory=dict)
    # Web access (SearXNG search + page fetch) for the WebSearch capability. Optional so
    # existing constructions keep working; main.run() supplies one. When None, the web tools
    # build a short-lived client from env per call.
    web: "WebClient | None" = None
    # Text->vector embedder for semantic memory search. Optional and inert when no embedding model
    # is configured (EMBED_MODEL unset): in that case fact search uses substring (LIKE) matching.
    embedder: "Embedder | None" = None
    # Containerized Python executor for the PythonSandbox capability. Optional (mainly a test
    # seam); when None, run_python builds one from env per call — the sandbox is stateless, so
    # there is nothing to share between calls.
    sandbox: "PythonSandbox | None" = None
    # The UI-selected model label for this run (see backend.model_selection.resolve_model).
    # Swarm/deep-research sub-agents read it so delegated work runs on the same model the user
    # chose for the conversation; None means the env-configured default.
    model: str | None = None
    # Live sub-agent trace side-channel. When set (swarm/streamed runs), run_subagent pushes
    # tagged event frames (thinking/tool_call/tool_result/agent_start/agent_end) onto it so
    # stream_run can multiplex each delegate's work into the UI in real time. None means no live
    # tracing (CLI / non-streamed / tests) — sub-agents run via the plain blocking path.
    # Run-scoped, and copied through dataclasses.replace() so nested dispatches share one sink.
    event_sink: "asyncio.Queue[dict[str, Any]] | None" = None
    # --- Swarm "agency" communication chart (swarm mode only) -------------------------------
    # The agent names the CURRENTLY-running agent may `send_message` to. ``None`` marks the
    # entry-point orchestrator (the agent the user talks to), which may message any roster agent;
    # a non-None list scopes a dispatched specialist to the recipients its AgentSpec declares, so
    # multi-hop flows (orchestrator -> A -> B) follow the explicit chart. Run-scoped; set per hop
    # by run_subagent via dataclasses.replace().
    agency_recipients: list[str] | None = None
    # Current hop depth in the agency: 0 at the entry point, +1 per delegation. run_subagent only
    # grants a dispatched specialist its own send_message tool while depth stays under
    # SWARM_MAX_DEPTH, bounding recursion alongside the per-run request limit.
    agency_depth: int = 0
    # Per-conversation overrides for the two swarm bounds (set from the UI's Configuration card;
    # see repo.get_conversation_swarm_settings). ``None`` means "use the env default"
    # (subagent.SWARM_MAX_PARALLEL / SWARM_MAX_DEPTH). Copied through dataclasses.replace() so a
    # conversation's values apply at every hop of the agency.
    swarm_max_parallel: int | None = None
    swarm_max_depth: int | None = None
    # Files the user uploaded this turn, already persisted as Documents by stream_run, as
    # ``[{document_id, filename, mime_type}]``. The after_run persistence hook stamps these onto
    # the human-readable user Message so a reloaded bubble can re-open them. Run-scoped; empty when
    # the turn carried no uploads.
    uploaded_attachments: list[dict[str, Any]] = field(default_factory=list)

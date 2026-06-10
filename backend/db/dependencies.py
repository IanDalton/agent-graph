"""Agent dependencies injected via Pydantic AI's ``deps_type``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.db.arcade_db import ArcadeClient

if TYPE_CHECKING:
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

"""Tests for graph topology serialization (repo.get_user_graph) used by the /api/graph endpoint.

Unit tests use a fake client that returns canned node/edge rows shaped like ArcadeDB's, so they
need no database. The live-DB behavior is exercised separately during manual verification.
"""

from __future__ import annotations

import asyncio
from typing import Any

from backend.db import repository as repo


class GraphClient:
    """Fake client returning canned node rows (HAS_NODE expand) and edge rows (outE expand).

    ``type_rows`` answers the ``schema:types`` lookup get_user_graph uses to tag each node's kind.
    """

    def __init__(
        self,
        node_rows: list[dict[str, Any]],
        edge_rows: list[dict[str, Any]],
        type_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._node_rows = node_rows
        self._edge_rows = edge_rows
        self._type_rows = type_rows or []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if "schema:types" in sql:
            return self._type_rows
        # The edge query is the only one that expands outE(); the node query just expands HAS_NODE.
        return self._edge_rows if "outE()" in sql else self._node_rows


def test_get_user_graph_serializes_nodes_and_edges() -> None:
    node_rows = [
        {"@rid": "#35:0", "@type": "Person", "@cat": "v", "user_id": "u", "name": "Alice", "age": 30},
        {"@rid": "#38:0", "@type": "Company", "@cat": "v", "user_id": "u", "name": "Acme"},
    ]
    edge_rows = [{"rid": "#41:0", "type": "WORKS_AT", "src": "#35:0", "dst": "#38:0"}]
    g = asyncio.run(repo.get_user_graph(GraphClient(node_rows, edge_rows), "u"))

    assert {n["id"] for n in g["nodes"]} == {"35_0", "38_0"}  # ids sanitized
    alice = next(n for n in g["nodes"] if n["id"] == "35_0")
    assert alice["type"] == "Person" and alice["label"] == "Alice"
    # Internal/bookkeeping keys are stripped; user-facing props kept.
    assert "user_id" not in alice["properties"] and "@rid" not in alice["properties"]
    assert alice["properties"]["age"] == 30

    assert len(g["edges"]) == 1
    e = g["edges"][0]
    assert e == {"id": "41_0", "source": "35_0", "target": "38_0", "label": "WORKS_AT"}


def test_get_user_graph_drops_edges_with_endpoint_past_limit() -> None:
    # Only one node is in the result set; an edge to a node outside the set must be dropped.
    node_rows = [{"@rid": "#35:0", "@type": "Person", "name": "Alice"}]
    edge_rows = [{"rid": "#41:0", "type": "KNOWS_X", "src": "#35:0", "dst": "#99:9"}]
    g = asyncio.run(repo.get_user_graph(GraphClient(node_rows, edge_rows), "u"))
    assert len(g["nodes"]) == 1
    assert g["edges"] == []  # dst not among displayed nodes


def test_get_user_graph_empty_when_no_nodes() -> None:
    g = asyncio.run(repo.get_user_graph(GraphClient([], []), "u"))
    assert g == {"nodes": [], "edges": []}


def test_get_user_graph_labels_fall_back_to_type() -> None:
    # No string property -> label falls back to the type name.
    node_rows = [{"@rid": "#5:0", "@type": "Measurement", "value": 42}]
    g = asyncio.run(repo.get_user_graph(GraphClient(node_rows, []), "u"))
    assert g["nodes"][0]["label"] == "Measurement"


def test_get_user_graph_attaches_kind_from_schema() -> None:
    node_rows = [
        {"@rid": "#35:0", "@type": "Person", "name": "Alice"},
        {"@rid": "#40:0", "@type": "Meeting", "name": "Standup"},
    ]
    type_rows = [
        {"name": "Person", "custom": {"kind": "semantic"}, "properties": []},
        {"name": "Meeting", "custom": {"kind": "episodic"}, "properties": []},
    ]
    g = asyncio.run(repo.get_user_graph(GraphClient(node_rows, [], type_rows), "u"))
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["35_0"]["kind"] == "semantic"
    assert by_id["40_0"]["kind"] == "episodic"


def test_get_user_graph_kind_is_none_for_unmarked_type() -> None:
    # A type with no kind marker (legacy/internal) reads back as None (treated as semantic by the UI).
    node_rows = [{"@rid": "#35:0", "@type": "Person", "name": "Alice"}]
    g = asyncio.run(repo.get_user_graph(GraphClient(node_rows, [], type_rows=[]), "u"))
    assert g["nodes"][0]["kind"] is None


def test_sanitize_rid() -> None:
    assert repo._sanitize_rid("#38:0") == "38_0"
    assert repo._sanitize_rid("#412:17") == "412_17"

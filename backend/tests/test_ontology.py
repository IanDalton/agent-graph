"""Tests for the OntologyManager capability (the two-tool node-creator pipeline).

All unit tests use a recording fake client and need no database. They exercise the
Pydantic validation boundary, the propose -> create pipeline, the ordering guard, and the
list_vertex_types discovery tool directly (the @capability.tool decorator returns the original
coroutine, so the tools are callable with a hand-built RunContext).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db.dependencies import GraphDependencies
from backend.schemas.graph_schemas import (
    CreateEdgeArgs,
    CreateNodeArgs,
    ProposeEdgeArgs,
    ProposeSchemaArgs,
    VertexProperty,
)
from backend.skills.graph_capability import build_memory
from backend.schemas.graph_schemas import UpdateNodeArgs
from backend.schemas.graph_schemas import DropEdgeTypeArgs, DropVertexTypeArgs
from backend.skills.ontology_capability import (
    _require_prior_proposal,
    build_ontology,
    create_edge,
    create_edge_type,
    create_node,
    create_vertex_type,
    delete_edge_type,
    delete_node,
    delete_vertex_type,
    list_vertex_types,
    propose_edge_type,
    propose_schema_change,
    update_node,
)

ONTOLOGY_TOOLS = {
    "list_vertex_types",
    "propose_schema_change",
    "create_vertex_type",
    "create_node",
    "propose_edge_type",
    "create_edge_type",
    "create_edge",
    "update_node",
    "delete_node",
    "delete_vertex_type",
    "delete_edge_type",
}


class RecordingClient:
    """Duck-typed stand-in for ArcadeClient that records commands instead of executing them."""

    def __init__(
        self,
        existing_types: list[dict[str, Any]] | None = None,
        rid_types: dict[str, str] | None = None,
    ) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._existing = existing_types or []
        self._rid_types = rid_types or {}  # rid -> type name, for node_type/node_exists lookups

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        # UPDATE/DELETE report affected rows as [{"count": N}].
        if sql.strip().upper().startswith(("UPDATE", "DELETE")):
            return [{"count": 1}]
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        if "schema:types" in sql:
            return self._existing
        # node_type: SELECT @type AS t FROM #<bucket>:<pos>
        if sql.strip().upper().startswith("SELECT @TYPE"):
            rid = sql.split("FROM", 1)[1].strip()
            t = self._rid_types.get(rid)
            return [{"t": t}] if t else []
        # create_conversation (persistence hook) checks an existence count; return 0 so it proceeds.
        if "count(" in sql.lower():
            return [{"n": 0}]
        return []


def _ctx(deps: GraphDependencies) -> RunContext[GraphDependencies]:
    """Minimal RunContext for invoking tool/guard coroutines directly in unit tests."""
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def _deps(db: RecordingClient | None = None) -> GraphDependencies:
    return GraphDependencies(db=db or RecordingClient(), user_id="u", conversation_id="c")


# --------------------------------------------------------------------------- #
# Validation boundary (no agent / DB)
# --------------------------------------------------------------------------- #
def test_node_name_must_be_pascal_case() -> None:
    # A valid generic, PascalCase name is accepted.
    ProposeSchemaArgs(node_name="SoftwareFramework", usage="frameworks", rationale="r")
    for bad in ["bad name", "lowercase", "User;DROP TYPE User", "9Thing", "", "Has-Dash"]:
        with pytest.raises(ValidationError):
            ProposeSchemaArgs(node_name=bad, usage="frameworks", rationale="r")


def test_usage_instruction_is_required() -> None:
    with pytest.raises(ValidationError):
        ProposeSchemaArgs(node_name="Person", usage="", rationale="r")


def test_vertex_property_validates_and_uppercases_type() -> None:
    assert VertexProperty(name="releasedYear", type="string").type == "STRING"
    with pytest.raises(ValidationError):
        VertexProperty(name="releasedYear", type="BLOB")  # not an allowed type
    with pytest.raises(ValidationError):
        VertexProperty(name="Bad Name", type="STRING")  # invalid property name


# --------------------------------------------------------------------------- #
# propose_schema_change (cognitive layer; no DB write)
# --------------------------------------------------------------------------- #
def test_propose_records_approved_proposal_and_does_not_touch_db() -> None:
    db = RecordingClient()
    deps = _deps(db)
    args = ProposeSchemaArgs(
        node_name="SoftwareFramework",
        usage="Use for software libraries/frameworks; store the specific framework as a record.",
        properties=[VertexProperty(name="name", type="STRING")],
        rationale="Need a generic home for frameworks like Django.",
    )
    proposal = asyncio.run(propose_schema_change(_ctx(deps), args))

    assert proposal.approved is True
    assert proposal.node_name == "SoftwareFramework"
    assert deps.proposed_schemas["SoftwareFramework"].usage == args.usage
    # Purely cognitive: no commands were issued.
    assert db.commands == []


# --------------------------------------------------------------------------- #
# create_vertex_type (executor) + ordering guard
# --------------------------------------------------------------------------- #
def test_create_without_prior_proposal_is_rejected() -> None:
    deps = _deps()
    with pytest.raises(ModelRetry):
        asyncio.run(create_vertex_type(_ctx(deps), "SoftwareFramework"))


def test_create_executes_approved_proposal() -> None:
    db = RecordingClient()
    deps = _deps(db)
    args = ProposeSchemaArgs(
        node_name="SoftwareFramework",
        usage="frameworks live here",
        properties=[
            VertexProperty(name="name", type="STRING"),
            VertexProperty(name="releasedYear", type="INTEGER"),
        ],
        rationale="r",
    )
    asyncio.run(propose_schema_change(_ctx(deps), args))
    message = asyncio.run(create_vertex_type(_ctx(deps), "SoftwareFramework"))

    sqls = [sql for sql, _ in db.commands]
    assert "CREATE VERTEX TYPE SoftwareFramework IF NOT EXISTS" in sqls
    assert "CREATE PROPERTY SoftwareFramework.name IF NOT EXISTS STRING" in sqls
    assert "CREATE PROPERTY SoftwareFramework.releasedYear IF NOT EXISTS INTEGER" in sqls

    # The usage instruction is persisted as type-level CUSTOM metadata (bound, not interpolated).
    alters = [
        params
        for sql, params in db.commands
        if sql.startswith("ALTER TYPE SoftwareFramework CUSTOM description")
    ]
    assert alters and alters[0]["usage"] == "frameworks live here"
    assert "Created" in message


def test_guard_blocks_create_without_matching_proposal() -> None:
    deps = _deps()
    call = ToolCallPart(tool_name="create_vertex_type", args={"node_name": "Person"})
    with pytest.raises(ModelRetry):
        asyncio.run(
            _require_prior_proposal(_ctx(deps), call=call, tool_def=None, args={"node_name": "Person"})
        )


def test_guard_passes_args_through_after_proposal() -> None:
    deps = _deps()
    asyncio.run(
        propose_schema_change(
            _ctx(deps), ProposeSchemaArgs(node_name="Person", usage="people", rationale="r")
        )
    )
    call = ToolCallPart(tool_name="create_vertex_type", args={"node_name": "Person"})
    sentinel = object()
    out = asyncio.run(_require_prior_proposal(_ctx(deps), call=call, tool_def=None, args=sentinel))
    assert out is sentinel  # guard returns the validated args unchanged when allowed


# --------------------------------------------------------------------------- #
# list_vertex_types (discovery)
# --------------------------------------------------------------------------- #
def test_list_vertex_types_extracts_usage_and_property_names() -> None:
    existing = [
        {
            "name": "SoftwareFramework",
            "custom": {"description": "frameworks"},
            "properties": [{"name": "name", "type": "STRING"}],
        },
        {"name": "Person", "custom": {}, "properties": []},
    ]
    deps = _deps(RecordingClient(existing_types=existing))
    infos = asyncio.run(list_vertex_types(_ctx(deps)))

    by_name = {i.name: i for i in infos}
    assert by_name["SoftwareFramework"].usage == "frameworks"
    assert by_name["SoftwareFramework"].properties == ["name"]
    assert by_name["Person"].usage is None
    assert by_name["Person"].properties == []


# --------------------------------------------------------------------------- #
# create_node (instance write)
# --------------------------------------------------------------------------- #
def test_create_node_args_validation() -> None:
    # Valid instance: PascalCase type, scalar property values.
    CreateNodeArgs(node_type="SoftwareFramework", properties={"name": "Django", "releasedYear": 2005})
    with pytest.raises(ValidationError):
        CreateNodeArgs(node_type="not pascal", properties={})
    with pytest.raises(ValidationError):
        CreateNodeArgs(node_type="SoftwareFramework", properties={"Bad Key": "x"})
    with pytest.raises(ValidationError):
        CreateNodeArgs(node_type="SoftwareFramework", properties={"meta": {"nested": "obj"}})


def test_create_node_requires_existing_type() -> None:
    # No such type in schema:types -> the tool refuses and tells the agent to create it first.
    deps = _deps(RecordingClient(existing_types=[]))
    with pytest.raises(ModelRetry):
        asyncio.run(create_node(_ctx(deps), CreateNodeArgs(node_type="SoftwareFramework", properties={"name": "Django"})))


def test_create_node_writes_vertex_with_bound_values() -> None:
    db = RecordingClient(existing_types=[{"name": "SoftwareFramework", "custom": {}, "properties": []}])
    deps = _deps(db)
    msg = asyncio.run(
        create_node(_ctx(deps), CreateNodeArgs(node_type="SoftwareFramework", properties={"name": "Django", "releasedYear": 2005}))
    )
    creates = [(sql, params) for sql, params in db.commands if sql.startswith("CREATE VERTEX SoftwareFramework SET")]
    assert creates, "expected a CREATE VERTEX for the instance"
    sql, params = creates[0]
    # Property names are interpolated (validated); values bind as parameters.
    assert "name = :p_name" in sql and "releasedYear = :p_releasedYear" in sql
    assert params["p_name"] == "Django" and params["p_releasedYear"] == 2005
    assert params["uid"] == "u"
    assert "Created SoftwareFramework node" in msg


# --------------------------------------------------------------------------- #
# Edge pipeline: propose_edge_type -> create_edge_type -> create_edge
# --------------------------------------------------------------------------- #
def test_edge_name_must_be_upper_snake_case() -> None:
    ProposeEdgeArgs(edge_name="WORKS_AT", usage="employment", rationale="r")
    for bad in ["usesDjango", "lower", "Has-Dash", "", "1USES"]:
        with pytest.raises(ValidationError):
            ProposeEdgeArgs(edge_name=bad, usage="u", rationale="r")


def test_create_edge_args_validation() -> None:
    CreateEdgeArgs(edge_type="USES", from_rid="#29:0", to_rid="#31:0", properties={"since": 2020})
    with pytest.raises(ValidationError):
        CreateEdgeArgs(edge_type="uses", from_rid="#29:0", to_rid="#31:0")  # bad edge name
    with pytest.raises(ValidationError):
        CreateEdgeArgs(edge_type="USES", from_rid="29:0", to_rid="#31:0")  # bad rid
    with pytest.raises(ValidationError):
        CreateEdgeArgs(edge_type="USES", from_rid="#29:0", to_rid="#31:0", properties={"x": [1, 2]})


def test_propose_edge_records_and_does_not_touch_db() -> None:
    db = RecordingClient()
    deps = _deps(db)
    proposal = asyncio.run(
        propose_edge_type(_ctx(deps), ProposeEdgeArgs(edge_name="USES", usage="person uses framework", rationale="r"))
    )
    assert proposal.approved is True
    assert deps.proposed_edges["USES"].usage == "person uses framework"
    assert db.commands == []


def test_create_edge_type_requires_prior_proposal_then_executes() -> None:
    deps = _deps()
    with pytest.raises(ModelRetry):
        asyncio.run(create_edge_type(_ctx(deps), "USES"))

    db = RecordingClient()
    deps = _deps(db)
    asyncio.run(
        propose_edge_type(
            _ctx(deps),
            ProposeEdgeArgs(edge_name="USES", usage="links a Person to a SoftwareFramework", rationale="r"),
        )
    )
    asyncio.run(create_edge_type(_ctx(deps), "USES"))
    sqls = [sql for sql, _ in db.commands]
    assert "CREATE EDGE TYPE USES IF NOT EXISTS" in sqls
    alters = [p for sql, p in db.commands if sql.startswith("ALTER TYPE USES CUSTOM description")]
    assert alters and alters[0]["usage"] == "links a Person to a SoftwareFramework"


def test_edge_type_guard_blocks_without_proposal() -> None:
    deps = _deps()
    call = ToolCallPart(tool_name="create_edge_type", args={"edge_name": "USES"})
    with pytest.raises(ModelRetry):
        asyncio.run(_require_prior_proposal(_ctx(deps), call=call, tool_def=None, args={"edge_name": "USES"}))


def test_create_edge_requires_existing_type_and_endpoints() -> None:
    # Edge type missing.
    deps = _deps(RecordingClient(existing_types=[]))
    with pytest.raises(ModelRetry):
        asyncio.run(create_edge(_ctx(deps), CreateEdgeArgs(edge_type="USES", from_rid="#29:0", to_rid="#31:0")))

    # Edge type exists, but the source node does not.
    deps = _deps(RecordingClient(existing_types=[{"name": "USES"}], rid_types={"#31:0": "SoftwareFramework"}))
    with pytest.raises(ModelRetry):
        asyncio.run(create_edge(_ctx(deps), CreateEdgeArgs(edge_type="USES", from_rid="#29:0", to_rid="#31:0")))


def test_create_edge_writes_edge_with_bound_values() -> None:
    db = RecordingClient(
        existing_types=[{"name": "USES"}],
        rid_types={"#29:0": "Person", "#31:0": "SoftwareFramework"},
    )
    deps = _deps(db)
    msg = asyncio.run(
        create_edge(
            _ctx(deps),
            CreateEdgeArgs(edge_type="USES", from_rid="#29:0", to_rid="#31:0", properties={"since": 2020}),
        )
    )
    creates = [(sql, params) for sql, params in db.commands if sql.startswith("CREATE EDGE USES FROM #29:0 TO #31:0")]
    assert creates, "expected a CREATE EDGE statement"
    sql, params = creates[0]
    assert "since = :p_since" in sql
    assert params["p_since"] == 2020
    assert "Created USES edge" in msg


# --------------------------------------------------------------------------- #
# update_node / delete_node (dedup management)
# --------------------------------------------------------------------------- #
def test_update_node_rejects_missing_and_protected() -> None:
    # No such node.
    deps = _deps(RecordingClient(rid_types={}))
    with pytest.raises(ModelRetry):
        asyncio.run(update_node(_ctx(deps), UpdateNodeArgs(rid="#29:0", properties={"age": 31})))

    # Protected internal type must not be editable via update_node.
    deps = _deps(RecordingClient(rid_types={"#3:0": "Message"}))
    with pytest.raises(ModelRetry):
        asyncio.run(update_node(_ctx(deps), UpdateNodeArgs(rid="#3:0", properties={"content": "x"})))


def test_update_node_writes_bound_values() -> None:
    db = RecordingClient(rid_types={"#29:0": "Person"})
    deps = _deps(db)
    msg = asyncio.run(update_node(_ctx(deps), UpdateNodeArgs(rid="#29:0", properties={"age": 31})))
    updates = [(sql, p) for sql, p in db.commands if sql.startswith("UPDATE #29:0 SET")]
    assert updates
    sql, params = updates[0]
    assert "age = :p_age" in sql and "WHERE user_id = :uid" in sql
    assert params["p_age"] == 31 and params["uid"] == "u"
    assert "Updated Person node #29:0" in msg


def test_delete_node_removes_existing_instance() -> None:
    db = RecordingClient(rid_types={"#29:0": "Person"})
    deps = _deps(db)
    msg = asyncio.run(delete_node(_ctx(deps), "#29:0"))
    assert any(sql.startswith("DELETE VERTEX FROM (SELECT FROM #29:0") for sql, _ in db.commands)
    assert "Deleted Person node #29:0" in msg


# --------------------------------------------------------------------------- #
# delete_vertex_type / delete_edge_type (drop a whole type)
# --------------------------------------------------------------------------- #
def test_drop_type_args_validation() -> None:
    DropVertexTypeArgs(node_name="Project")
    DropEdgeTypeArgs(edge_name="WORKS_ON")
    for bad in ["lower", "has space", "User;DROP TYPE User", ""]:
        with pytest.raises(ValidationError):
            DropVertexTypeArgs(node_name=bad)
    for bad in ["lower", "Has-Dash", "WORKS ON", ""]:
        with pytest.raises(ValidationError):
            DropEdgeTypeArgs(edge_name=bad)


def test_delete_vertex_type_drops_instances_then_type() -> None:
    db = RecordingClient(existing_types=[{"name": "Project", "type": "vertex"}])
    deps = _deps(db)
    msg = asyncio.run(delete_vertex_type(_ctx(deps), DropVertexTypeArgs(node_name="Project")))
    sqls = [sql for sql, _ in db.commands]
    assert "DELETE VERTEX FROM Project" in sqls
    assert "DROP TYPE Project IF EXISTS" in sqls
    # The delete must run before the drop (a non-empty type can't be dropped).
    assert sqls.index("DELETE VERTEX FROM Project") < sqls.index("DROP TYPE Project IF EXISTS")
    assert "Dropped vertex type 'Project'" in msg


def test_delete_edge_type_drops_edges_then_type() -> None:
    db = RecordingClient(existing_types=[{"name": "WORKS_ON", "type": "edge"}])
    deps = _deps(db)
    msg = asyncio.run(delete_edge_type(_ctx(deps), DropEdgeTypeArgs(edge_name="WORKS_ON")))
    sqls = [sql for sql, _ in db.commands]
    assert "DELETE FROM WORKS_ON UNSAFE" in sqls
    assert "DROP TYPE WORKS_ON IF EXISTS" in sqls
    assert sqls.index("DELETE FROM WORKS_ON UNSAFE") < sqls.index("DROP TYPE WORKS_ON IF EXISTS")
    assert "Dropped edge type 'WORKS_ON'" in msg


def test_delete_vertex_type_refuses_protected() -> None:
    # An internal type must never be droppable, even though it exists in the schema.
    db = RecordingClient(existing_types=[{"name": "Message", "type": "vertex"}])
    deps = _deps(db)
    with pytest.raises(ModelRetry):
        asyncio.run(delete_vertex_type(_ctx(deps), DropVertexTypeArgs(node_name="Message")))
    # RunMessages (replay store) is protected too.
    with pytest.raises(ModelRetry):
        asyncio.run(delete_vertex_type(_ctx(deps), DropVertexTypeArgs(node_name="RunMessages")))
    assert not any("DROP TYPE" in sql for sql, _ in db.commands)


def test_delete_edge_type_refuses_protected() -> None:
    db = RecordingClient(existing_types=[{"name": "KNOWS", "type": "edge"}])
    deps = _deps(db)
    with pytest.raises(ModelRetry):
        asyncio.run(delete_edge_type(_ctx(deps), DropEdgeTypeArgs(edge_name="KNOWS")))
    assert not any("DROP TYPE" in sql for sql, _ in db.commands)


def test_delete_type_refuses_missing_and_wrong_category() -> None:
    # Missing type.
    deps = _deps(RecordingClient(existing_types=[]))
    with pytest.raises(ModelRetry):
        asyncio.run(delete_vertex_type(_ctx(deps), DropVertexTypeArgs(node_name="Nope")))

    # Calling delete_vertex_type on an EDGE type must be refused (else DELETE VERTEX/wrong cleanup).
    # (name matches the lookup so this hits the wrong-category branch, not the missing branch.)
    deps = _deps(RecordingClient(existing_types=[{"name": "Partners", "type": "edge"}]))
    with pytest.raises(ModelRetry):
        asyncio.run(delete_vertex_type(_ctx(deps), DropVertexTypeArgs(node_name="Partners")))
    assert not any("DROP TYPE" in sql for sql, _ in deps.db.commands)

    # And delete_edge_type on a VERTEX type must be refused (UNSAFE delete would strip vertices).
    deps = _deps(RecordingClient(existing_types=[{"name": "PROJECT", "type": "vertex"}]))
    with pytest.raises(ModelRetry):
        asyncio.run(delete_edge_type(_ctx(deps), DropEdgeTypeArgs(edge_name="PROJECT")))
    assert not any("DROP TYPE" in sql for sql, _ in deps.db.commands)


# --------------------------------------------------------------------------- #
# Registration / composition with the memory capability
# --------------------------------------------------------------------------- #
def test_ontology_tools_are_registered() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(
        model,
        deps_type=GraphDependencies,
        capabilities=[*build_memory(), *build_ontology()],
    )
    asyncio.run(agent.run("hi", deps=_deps()))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert ONTOLOGY_TOOLS <= names

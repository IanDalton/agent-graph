"""Pydantic models used as agent tool inputs/outputs."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Identifier rules. These are the *only* thing standing between the model and DDL
# injection, because ArcadeDB cannot bind type/property names as parameters.
_PASCAL_CASE_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")  # generic vertex type names: Person, SoftwareFramework
_EDGE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")  # edge type names: USES, WORKS_AT, HAS_NODE
_PROP_NAME_RE = re.compile(r"^[a-z][A-Za-z0-9_]*$")  # property names: name, releasedYear, created_at
_RID_RE = re.compile(r"^#\d+:\d+$")  # ArcadeDB record id, e.g. #29:0
ALLOWED_PROPERTY_TYPES = frozenset(
    {"STRING", "INTEGER", "LONG", "FLOAT", "DOUBLE", "BOOLEAN", "DATETIME", "DATE"}
)


class RawQuery(BaseModel):
    """A read-only ArcadeDB SQL query the agent wants to run directly."""

    query: str = Field(..., description="A read-only ArcadeDB SQL query (must start with SELECT, MATCH, or TRAVERSE).")
    rationale: str = Field(..., description="Why this query is necessary based on the user's request.")


class StoreFactArgs(BaseModel):
    """A durable fact the agent wants to remember about the user."""

    text: str = Field(..., description="The fact to remember, phrased so it is useful in future conversations.")


class MemoryHit(BaseModel):
    """A single retrieved piece of memory (a past message or a stored fact)."""

    kind: str = Field(..., description="'message' or 'fact'.")
    content: str
    created_at: str | None = None
    fact_id: str | None = Field(
        None,
        description="For 'fact' hits: the id to pass to update_fact/delete_fact to revise it in place.",
    )


class MemorySearchResult(BaseModel):
    """Structured result returned by the search_memory tool."""

    hits: list[MemoryHit] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Ontology management (the two-tool node-creator pipeline)
# --------------------------------------------------------------------------- #
class VertexProperty(BaseModel):
    """A single typed property on a proposed vertex type."""

    name: str = Field(..., description="Property name, camelCase or snake_case (e.g. 'releasedYear').")
    type: str = Field(..., description=f"ArcadeDB type, one of: {sorted(ALLOWED_PROPERTY_TYPES)}.")

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _PROP_NAME_RE.match(v):
            raise ValueError("Property name must start lowercase and be alphanumeric/underscore only.")
        return v

    @field_validator("type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        up = v.strip().upper()
        if up not in ALLOWED_PROPERTY_TYPES:
            raise ValueError(f"Property type must be one of {sorted(ALLOWED_PROPERTY_TYPES)}.")
        return up


class ProposeSchemaArgs(BaseModel):
    """Agent's proposal for a new GENERIC vertex type (cognitive layer; no DB write)."""

    node_name: str = Field(
        ...,
        description=(
            "A GENERIC, reusable, PascalCase vertex type — a CATEGORY, never a specific instance. "
            "Good: 'SoftwareFramework', 'Person', 'City', 'ProgrammingLanguage'. "
            "Bad: 'React', 'JohnDoe', 'BuenosAires', 'Python' (those are instances/data, not types)."
        ),
    )
    usage: str = Field(
        ...,
        min_length=1,
        description=(
            "A BRIEF instruction (one or two sentences) describing WHEN to use this vertex type and "
            "what kind of instance belongs in it — e.g. 'Use for software libraries and frameworks "
            "such as React or Django; store the specific framework as a record, not as its own type.' "
            "This is persisted on the type so future runs can read it and reuse the right type."
        ),
    )
    properties: list[VertexProperty] = Field(
        default_factory=list, description="Optional typed properties the generic type should carry."
    )
    rationale: str = Field(
        ...,
        description="Why this type is needed AND why it is general rather than a specific instance.",
    )

    @field_validator("node_name")
    @classmethod
    def _pascal_case(cls, v: str) -> str:
        if not _PASCAL_CASE_RE.match(v):
            raise ValueError("node_name must be PascalCase and alphanumeric (e.g. 'SoftwareFramework').")
        return v


class SchemaProposal(BaseModel):
    """Validated, approved proposal returned by propose_schema_change and consumed by create_vertex_type."""

    approved: bool = Field(..., description="True when the proposal passed structural validation.")
    node_name: str
    usage: str = Field(..., description="The 'when to use this type' instruction persisted on the type.")
    properties: list[VertexProperty] = Field(default_factory=list)
    guidance: str = Field(..., description="Next step for the agent (e.g. call create_vertex_type).")


class VertexTypeInfo(BaseModel):
    """One existing type in the current ontology, as returned by list_vertex_types."""

    name: str
    usage: str | None = Field(None, description="The stored 'when to use' note, if the type has one.")
    properties: list[str] = Field(default_factory=list, description="Names of the type's declared properties.")


class CreateNodeArgs(BaseModel):
    """Create an INSTANCE (a node/record) of an existing generic vertex type."""

    node_type: str = Field(
        ...,
        description=(
            "An EXISTING generic vertex type to create an instance of (e.g. 'SoftwareFramework'). "
            "If it does not exist yet, create it first via propose_schema_change + create_vertex_type."
        ),
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The instance's property values keyed by property name "
            "(e.g. {'name': 'Django', 'releasedYear': 2005}). Values must be scalars."
        ),
    )

    @field_validator("node_type")
    @classmethod
    def _pascal_case(cls, v: str) -> str:
        if not _PASCAL_CASE_RE.match(v):
            raise ValueError("node_type must be PascalCase and alphanumeric (e.g. 'SoftwareFramework').")
        return v

    @field_validator("properties")
    @classmethod
    def _valid_properties(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_scalar_properties(v)


class UpdateNodeArgs(BaseModel):
    """Update property values on an existing instance node, identified by record id."""

    rid: str = Field(..., description="Record id of the node to update, e.g. '#29:0' (from create_node or a query).")
    properties: dict[str, Any] = Field(
        ...,
        description="Property values to set/overwrite, keyed by property name (e.g. {'age': 31}). Scalars only.",
    )

    @field_validator("rid")
    @classmethod
    def _valid_rid(cls, v: str) -> str:
        if not _RID_RE.match(v):
            raise ValueError("rid must look like '#<bucket>:<position>', e.g. '#29:0'.")
        return v

    @field_validator("properties")
    @classmethod
    def _valid_properties(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_scalar_properties(v)


def _validate_scalar_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Ensure property names are safe identifiers and values are scalars (shared by node/edge args)."""
    for key, value in props.items():
        if not _PROP_NAME_RE.match(key):
            raise ValueError(
                f"Invalid property name '{key}': must start lowercase, alphanumeric/underscore only."
            )
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Property '{key}' must be a scalar (string, number, boolean, or null).")
    return props


class ProposeEdgeArgs(BaseModel):
    """Agent's proposal for a new GENERIC edge (relationship) type (cognitive layer; no DB write)."""

    edge_name: str = Field(
        ...,
        description=(
            "A GENERIC, reusable, UPPER_SNAKE_CASE relationship type — a verb/relation, not an instance. "
            "Good: 'USES', 'WORKS_AT', 'FRIEND_OF', 'LOCATED_IN'. Bad: 'usesDjango', 'JohnUsesReact'."
        ),
    )
    usage: str = Field(
        ...,
        min_length=1,
        description=(
            "A BRIEF instruction (one or two sentences) on WHEN to use this relationship and which "
            "kinds of nodes it connects — e.g. 'Connects a Person to a SoftwareFramework they use.' "
            "Persisted on the type so future runs can read it and reuse the right relationship."
        ),
    )
    properties: list[VertexProperty] = Field(
        default_factory=list, description="Optional typed properties carried on the relationship itself."
    )
    rationale: str = Field(..., description="Why this relationship is needed and why it is generic.")

    @field_validator("edge_name")
    @classmethod
    def _upper_snake(cls, v: str) -> str:
        if not _EDGE_NAME_RE.match(v):
            raise ValueError("edge_name must be UPPER_SNAKE_CASE (e.g. 'WORKS_AT').")
        return v


class EdgeProposal(BaseModel):
    """Validated, approved edge proposal returned by propose_edge_type and consumed by create_edge_type."""

    approved: bool = Field(..., description="True when the proposal passed structural validation.")
    edge_name: str
    usage: str = Field(..., description="The 'when to use this relationship' instruction persisted on the type.")
    properties: list[VertexProperty] = Field(default_factory=list)
    guidance: str = Field(..., description="Next step for the agent (e.g. call create_edge_type).")


class CreateEdgeArgs(BaseModel):
    """Create a relationship (edge) between two existing instance nodes, identified by record id."""

    edge_type: str = Field(
        ...,
        description="An EXISTING UPPER_SNAKE_CASE edge type (create it first via propose_edge_type + create_edge_type).",
    )
    from_rid: str = Field(..., description="Record id of the source node, e.g. '#29:0' (from create_node or a query).")
    to_rid: str = Field(..., description="Record id of the target node, e.g. '#31:0'.")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Optional scalar property values carried on the edge."
    )

    @field_validator("edge_type")
    @classmethod
    def _upper_snake(cls, v: str) -> str:
        if not _EDGE_NAME_RE.match(v):
            raise ValueError("edge_type must be UPPER_SNAKE_CASE (e.g. 'WORKS_AT').")
        return v

    @field_validator("from_rid", "to_rid")
    @classmethod
    def _valid_rid(cls, v: str) -> str:
        if not _RID_RE.match(v):
            raise ValueError("Record id must look like '#<bucket>:<position>', e.g. '#29:0'.")
        return v

    @field_validator("properties")
    @classmethod
    def _valid_properties(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_scalar_properties(v)

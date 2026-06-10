"""Pydantic models used as agent tool inputs/outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class MemorySearchResult(BaseModel):
    """Structured result returned by the search_memory tool."""

    hits: list[MemoryHit] = Field(default_factory=list)

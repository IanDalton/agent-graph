"""Structured outputs for the knowledge-base compiler (``backend.kb_compiler``).

The compiler is a deterministic, code-orchestrated pipeline (not a tool-calling agent): each phase is
a single LLM call that returns one of these Pydantic models via ``Agent(output_type=...)``. Mirrors
OpenKB's compile pipeline (summary -> plan -> per-page generation -> summary rewrite), adapted to our
title-based ``[[wikilinks]]`` and ArcadeDB graph.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# OpenKB's entity vocabulary (openkb config `entity_types`); the page's entity kind must be one of these.
_ENTITY_TYPES = ("person", "organization", "place", "product", "work", "event", "other")


class SummaryDraft(BaseModel):
    """A per-source summary page (phase 1) or its rewrite (phase 4)."""

    description: str = Field(description="One sentence (< 100 chars) describing the document.")
    content: str = Field(description="The summary body in Markdown, with [[Title]] wikilinks.")


class ConceptPlan(BaseModel):
    """Concept-page actions the plan phase decides for one document."""

    create: list[str] = Field(default_factory=list, description="New concept page titles to create.")
    update: list[str] = Field(default_factory=list, description="Existing concept titles to update.")
    related: list[str] = Field(default_factory=list, description="Existing concepts merely related.")


class EntityRef(BaseModel):
    """An entity the plan wants to create/update, with its (validated) type."""

    name: str
    type: str = "other"

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        v = (v or "other").strip().lower()
        return v if v in _ENTITY_TYPES else "other"


class EntityPlan(BaseModel):
    """Entity-page actions the plan phase decides for one document."""

    create: list[EntityRef] = Field(default_factory=list)
    update: list[EntityRef] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)


class KbPlan(BaseModel):
    """The plan phase output: which concept/entity pages to create/update from this document."""

    concepts: ConceptPlan = Field(default_factory=ConceptPlan)
    entities: EntityPlan = Field(default_factory=EntityPlan)


class PageDraft(BaseModel):
    """A generated concept page body (phase 3)."""

    description: str = Field(description="One sentence (< 100 chars) defining the concept.")
    content: str = Field(description="The page body in Markdown, with [[Title]] wikilinks.")


class KbConceptOut(BaseModel):
    """One concept page emitted by the single master-synthesis call."""

    title: str = Field(description="The concept's page title.")
    description: str = Field(default="", description="One sentence (< 100 chars) defining it.")
    content: str = Field(default="", description="Markdown body, with [[Title]] wikilinks.")
    sources: list[str] = Field(
        default_factory=list, description="Titles of the source documents this concept draws from."
    )


class KbEntityOut(BaseModel):
    """One entity page emitted by the master-synthesis call (concept + a validated type)."""

    title: str
    type: str = "other"
    description: str = ""
    content: str = ""
    sources: list[str] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        v = (v or "other").strip().lower()
        return v if v in _ENTITY_TYPES else "other"


class KbSynthesis(BaseModel):
    """The single master-synthesis output: the concept + entity 'learnings' across all summaries."""

    concepts: list[KbConceptOut] = Field(default_factory=list)
    entities: list[KbEntityOut] = Field(default_factory=list)


class EntityDraft(BaseModel):
    """A generated entity page body (phase 3) — like PageDraft but carries the entity type."""

    description: str = Field(description="One sentence (< 100 chars) describing the entity.")
    type: str = "other"
    content: str = Field(description="The page body in Markdown, with [[Title]] wikilinks.")

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        v = (v or "other").strip().lower()
        return v if v in _ENTITY_TYPES else "other"

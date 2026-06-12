"""Pydantic models for the Documents capability's tool inputs/outputs.

Kept separate from :mod:`backend.schemas.graph_schemas` (memory/ontology I/O) for clarity.
Documents are agent-authored artifacts (reports, notes, code listings) persisted as ``Document``
vertices and surfaced in the web UI's Documents pane, where text-based ones are user-editable.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

# Conservative media-type shape (type/subtype with the usual suffix chars). The stored value is
# only ever bound as a parameter, so this is a sanity check, not an injection boundary.
_MIME_RE = re.compile(r"^[a-z]+/[a-z0-9][a-z0-9.+_-]*$")


def _normalize_mime(v: str) -> str:
    low = v.strip().lower()
    if not _MIME_RE.match(low):
        raise ValueError("mime_type must look like 'text/markdown' or 'text/x-python'.")
    return low


class CreateDocumentArgs(BaseModel):
    """A new document the agent wants to author for the user."""

    title: str = Field(..., min_length=1, description="A short human-readable document title.")
    content: str = Field(..., description="The full document body (e.g. markdown text).")
    mime_type: str = Field(
        "text/markdown",
        description=(
            "The document's media type. Use 'text/markdown' (default) for prose/reports, "
            "'text/plain' for raw text, 'text/x-python' for Python code, 'text/csv' for tables, "
            "'application/json' for data."
        ),
    )

    @field_validator("mime_type")
    @classmethod
    def _valid_mime(cls, v: str) -> str:
        return _normalize_mime(v)


class UpdateDocumentArgs(BaseModel):
    """Revise an existing document in place (instead of creating a near-duplicate)."""

    document_id: str = Field(..., description="The id returned by create_document or list_documents.")
    title: str | None = Field(None, description="New title; omit to keep the current one.")
    content: str | None = Field(
        None, description="New full body (replaces the old content); omit to keep the current one."
    )


class DocumentInfo(BaseModel):
    """One document's metadata (no body), as returned by list_documents."""

    document_id: str
    conversation_id: str | None = None
    title: str = ""
    mime_type: str = "text/markdown"
    encoding: str = Field(
        "text", description="'text' (content is the literal text) or 'base64' (binary, e.g. a PDF)."
    )
    created_at: str | None = None
    updated_at: str | None = None


class DocumentContent(BaseModel):
    """A full document, as returned by read_document."""

    document_id: str
    title: str = ""
    mime_type: str = "text/markdown"
    encoding: str = Field(
        "text", description="'text' (content is the literal text) or 'base64' (binary, e.g. a PDF)."
    )
    content: str = ""
    created_at: str | None = None
    updated_at: str | None = None

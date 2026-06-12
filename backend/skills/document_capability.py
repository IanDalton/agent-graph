"""Documents capability: let the agent author durable documents the user can read and edit.

Exposed via :func:`build_documents`, dropped into ``Agent(capabilities=...)``. Documents are
persisted as ``Document`` vertices (linked ``Conversation -HAS_DOCUMENT-> Document``) through
:mod:`backend.db.repository`, and surfaced in the web UI's Documents pane — where text-based
documents (markdown, plain text, code) are editable by the user via the HTTP API.

Tools:

- ``create_document`` — author a new document (markdown by default). Returns its document_id.
- ``update_document`` — revise an existing document in place (avoid near-duplicate documents).
- ``read_document`` — read a document's current body (the USER may have edited it since the
  agent last wrote it, so read before revising).
- ``list_documents`` — this conversation's documents (metadata only).
- ``delete_document`` — remove an obsolete document.

Mutating tools raise ``ModelRetry`` on a bad document_id (the standard "fix your input" path,
mirroring update_fact/delete_fact); DB outages propagate like the other persistence tools.
"""

from __future__ import annotations

import logging

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.schemas.document_schemas import (
    CreateDocumentArgs,
    DocumentContent,
    DocumentInfo,
    UpdateDocumentArgs,
)

logger = logging.getLogger("agent_graph.documents")

INSTRUCTIONS = (
    "You can author DOCUMENTS for the user: durable artifacts like reports, plans, notes, code "
    "listings or data files. Documents appear in the user's Documents pane next to the chat, and "
    "text-based ones (markdown, plain text, code) can be EDITED BY THE USER there.\n"
    "Use `create_document` when the user asks for a deliverable that outlives the chat (a report, "
    "a draft, a spec, a saved analysis) or when an answer is long/structured enough that the user "
    "will want to keep or edit it. Prefer markdown ('text/markdown') unless the content is raw "
    "code or data. Keep the chat reply short and point at the document instead of duplicating it.\n"
    "INTERACTIVE UIs: a 'text/html' document renders as a LIVE, interactive preview in the user's "
    "panel (a sandboxed iframe). When the user asks for a small app, widget, game, form or "
    "visualization, create ONE self-contained HTML document: inline ALL CSS in <style> and ALL "
    "JavaScript in <script>, no external URLs (no CDNs, fonts, images — the iframe has no network). "
    "Vanilla JS + DOM APIs work; make it responsive to the panel width.\n"
    "PDFs are NOT created with this tool — generate them with run_python (write to /out); the "
    "file becomes a document automatically.\n"
    "AVOID DUPLICATES: before creating, call `list_documents` — if a document on the same subject "
    "already exists, revise it with `update_document` instead of creating another.\n"
    "THE USER CAN EDIT: a document's content may have changed since you wrote it. ALWAYS call "
    "`read_document` to get the current body before revising or summarizing one — never assume "
    "it still says what you last wrote. `update_document` replaces the full content, so send the "
    "complete new body, not a diff."
)

documents_capability = Capability(id="Documents", instructions=INSTRUCTIONS)


@documents_capability.tool
async def create_document(
    ctx: RunContext[GraphDependencies], args: CreateDocumentArgs
) -> DocumentInfo:
    """Create a new document for the user (markdown by default). Returns its document_id.

    Check list_documents first — revise an existing document on the same subject with
    update_document rather than creating a near-duplicate.
    """
    deps = ctx.deps
    document_id = await repo.create_document(
        deps.db,
        deps.user_id,
        deps.conversation_id,
        title=args.title,
        content=args.content,
        mime_type=args.mime_type,
    )
    return DocumentInfo(
        document_id=document_id,
        conversation_id=deps.conversation_id,
        title=args.title,
        mime_type=args.mime_type,
    )


@documents_capability.tool
async def update_document(ctx: RunContext[GraphDependencies], args: UpdateDocumentArgs) -> str:
    """Revise an existing document in place (use its document_id from create/list_documents).

    Replaces the full content — send the complete new body. Read the current body first with
    read_document: the user may have edited it since you last wrote it.
    """
    if args.title is None and args.content is None:
        raise ModelRetry("Nothing to update: pass a new title and/or content.")
    updated = await repo.update_document(
        ctx.deps.db, ctx.deps.user_id, args.document_id, title=args.title, content=args.content
    )
    if not updated:
        raise ModelRetry(
            f"No document with id {args.document_id!r} for this user. "
            "Use list_documents to find the correct document_id."
        )
    return f"Updated document {args.document_id}."


@documents_capability.tool
async def read_document(ctx: RunContext[GraphDependencies], document_id: str) -> DocumentContent:
    """Read a document's current title and full body (it may include edits made by the user)."""
    row = await repo.get_document(ctx.deps.db, ctx.deps.user_id, document_id)
    if row is None:
        raise ModelRetry(
            f"No document with id {document_id!r} for this user. "
            "Use list_documents to find the correct document_id."
        )
    return DocumentContent(
        document_id=row.get("document_id", document_id),
        title=row.get("title") or "",
        mime_type=row.get("mime_type") or "text/markdown",
        content=row.get("content") or "",
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@documents_capability.tool
async def list_documents(ctx: RunContext[GraphDependencies]) -> list[DocumentInfo]:
    """List this conversation's documents (titles and ids, no bodies), most recently updated first."""
    rows = await repo.list_documents(
        ctx.deps.db, ctx.deps.user_id, conversation_id=ctx.deps.conversation_id
    )
    return [
        DocumentInfo(
            document_id=r.get("document_id", ""),
            conversation_id=r.get("conversation_id"),
            title=r.get("title") or "",
            mime_type=r.get("mime_type") or "text/markdown",
            created_at=r.get("created_at"),
            updated_at=r.get("updated_at"),
        )
        for r in rows
        if r.get("document_id")
    ]


@documents_capability.tool
async def delete_document(ctx: RunContext[GraphDependencies], document_id: str) -> str:
    """Delete an obsolete document by its document_id (from list_documents)."""
    deleted = await repo.delete_document(ctx.deps.db, ctx.deps.user_id, document_id)
    if not deleted:
        raise ModelRetry(
            f"No document with id {document_id!r} for this user. "
            "Use list_documents to find the correct document_id."
        )
    return f"Deleted document {document_id}."


def build_documents() -> list[Capability]:
    """Return the documents capability to add to ``Agent(capabilities=...)``.

    The database connection is supplied per-run through ``GraphDependencies`` (``ctx.deps.db``),
    so nothing needs to be wired in here.
    """
    return [documents_capability]

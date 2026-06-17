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
    DocumentSearchHit,
    UpdateDocumentArgs,
)

logger = logging.getLogger("agent_graph.documents")

# Cap on the excerpt returned per search hit — enough to judge relevance without flooding context.
_SNIPPET_CHARS = 1500


def _row_to_info(row: dict) -> DocumentInfo:
    return DocumentInfo(
        document_id=row.get("document_id", ""),
        conversation_id=row.get("conversation_id"),
        project_id=row.get("project_id"),
        is_global=bool(row.get("is_global")),
        title=row.get("title") or "",
        mime_type=row.get("mime_type") or "text/markdown",
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )

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
    "complete new body, not a diff.\n"
    "PROJECT REFERENCE DOCUMENTS: when this conversation belongs to a project, the user may have "
    "uploaded reference documents (and may have marked some documents 'global', available in every "
    "project). The system prompt lists them. To use them, `search_project_documents(query)` to find "
    "the relevant ones by content, then `read_project_document(document_id)` to read one in full. "
    "Ground answers in these documents rather than guessing, and cite which document you used."
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
    return [_row_to_info(r) for r in rows if r.get("document_id")]


@documents_capability.tool
async def list_project_documents(ctx: RunContext[GraphDependencies]) -> list[DocumentInfo]:
    """List the reference documents available to this conversation (its project's + global ones).

    Returns metadata only (titles + ids). Empty when the conversation is not in a project and there
    are no global documents. Read one with read_project_document or search with
    search_project_documents.
    """
    rows = await repo.list_documents(
        ctx.deps.db,
        ctx.deps.user_id,
        project_id=ctx.deps.project_id,
        include_global=True,
    )
    return [_row_to_info(r) for r in rows if r.get("document_id")]


@documents_capability.tool
async def read_project_document(
    ctx: RunContext[GraphDependencies], document_id: str
) -> DocumentContent:
    """Read a project (or global) reference document's full body by its document_id."""
    row = await repo.get_document(ctx.deps.db, ctx.deps.user_id, document_id)
    if row is None:
        raise ModelRetry(
            f"No document with id {document_id!r} for this user. "
            "Use list_project_documents or search_project_documents to find the correct id."
        )
    return DocumentContent(
        document_id=row.get("document_id", document_id),
        project_id=row.get("project_id"),
        is_global=bool(row.get("is_global")),
        title=row.get("title") or "",
        mime_type=row.get("mime_type") or "text/markdown",
        content=row.get("content") or "",
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@documents_capability.tool
async def search_project_documents(
    ctx: RunContext[GraphDependencies], query: str
) -> list[DocumentSearchHit]:
    """Search the conversation's reference documents (its project's + global) by content.

    Ranks by semantic similarity when embeddings are configured, else by substring match. Returns
    the most relevant documents with a leading excerpt; read one in full with read_project_document.
    Tolerant: any failure returns an empty list rather than aborting the run.
    """
    deps = ctx.deps
    try:
        embedding = None
        if deps.embedder is not None:
            embedding = await deps.embedder.embed(query)
        rows = await repo.search_documents(
            deps.db,
            deps.user_id,
            query,
            embedding=embedding,
            project_id=deps.project_id,
            include_global=True,
        )
    except Exception:  # noqa: BLE001 — search is best-effort; never abort the run.
        logger.warning("search_project_documents failed; returning no hits", exc_info=True)
        return []
    hits: list[DocumentSearchHit] = []
    for r in rows:
        did = r.get("document_id")
        if not did:
            continue
        content = r.get("content") or ""
        hits.append(
            DocumentSearchHit(
                document_id=did,
                title=r.get("title") or "",
                project_id=r.get("project_id"),
                is_global=bool(r.get("is_global")),
                snippet=content[:_SNIPPET_CHARS],
            )
        )
    return hits


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

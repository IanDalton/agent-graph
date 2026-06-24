"""Background knowledge-base compiler — turns a project's uploaded documents into an OpenKB-style
wiki of cross-linked markdown pages, stored as graph nodes.

Design (tuned for slow local models that loop on large contexts):

1. **Extract** each source's text — plain for text docs, via ``pypdf`` for PDFs (best-effort; a
   non-extractable/binary source is skipped).
2. **Summaries — short parallel threads.** One small LLM call per source (sees ONLY that document),
   run concurrently (``KB_COMPILE_MAX_PARALLEL``). Each produces a :class:`SummaryDraft`, stored as a
   ``KbSummary`` with a ``HAS_SUMMARY`` edge from its source. Bounded context ⇒ no looping.
3. **One master synthesis thread.** A single LLM call over the (short) summaries produces the
   connective layer — the cross-document **concepts** and **entities** (the "learnings") as a
   :class:`KbSynthesis`, each page naming the source titles it draws from.
4. **Link + backlink pass (no LLM).** Resolve every ``[[Title]]`` against the page titles, strip
   ghost links, and (re)build the typed edges: ``KB_LINK`` (page↔page references), ``MENTIONS``
   (page→source provenance), and the ``## Sources`` / ``## Related`` backlink sections.

Best-effort throughout (like :mod:`backend.memory_curator`): failures are logged and swallowed so an
upload is never blocked; one bad source/page is skipped, not fatal. Must NOT import
:mod:`backend.main`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import UsageLimits

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient
from backend.embeddings import Embedder
from backend.model_selection import resolve_model
from backend.schemas.kb_schemas import KbSynthesis, SummaryDraft

logger = logging.getLogger("agent_graph.kb_compiler")

# Kill switch — set KB_COMPILE_ENABLED=0 to disable automatic + manual KB compilation entirely.
KB_COMPILE_ENABLED = os.getenv("KB_COMPILE_ENABLED", "1") not in ("0", "false", "False")
# Per-call request cap inside _generate (a structured-output call should need 1).
KB_COMPILER_REQUEST_LIMIT = int(os.getenv("KB_COMPILER_REQUEST_LIMIT", "3"))
# How many summary threads run at once. The local model serializes on one GPU, but overlapping the
# request/parse latency still helps. Tune down to 1 if the backend pressures the model server.
KB_COMPILE_MAX_PARALLEL = int(os.getenv("KB_COMPILE_MAX_PARALLEL", "3"))
# Retries on a transient model error (e.g. the local llama-server reloading a model on demand).
KB_GENERATE_RETRIES = int(os.getenv("KB_GENERATE_RETRIES", "2"))
# Debounce window: rapid successive uploads to one project coalesce into a single compile.
KB_COMPILE_DEBOUNCE_SECONDS = float(os.getenv("KB_COMPILE_DEBOUNCE_SECONDS", "20"))

# The Document subtypes that are *generated* wiki pages (everything else project-scoped is a source).
_KB_PAGE_KINDS = frozenset(
    {"KbPage", "KbSummary", "KbConcept", "KbEntity", "KbExploration", "KbIndex"}
)
# Cap per-source text fed to a summary call, and per-summary text fed to the master — keeps every
# context small so the local model doesn't loop.
_MAX_SOURCE_CHARS = 12000
_MAX_SUMMARY_EXCERPT = 700
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


# --------------------------------------------------------------------------- prompts
SUMMARY_INSTR = (
    "You are compiling a personal knowledge base. Write a faithful, self-contained SUMMARY of the "
    "ONE source document provided. Return a one-sentence `description` (under 100 characters) and a "
    "markdown `content` body capturing the document's key facts, specs, and procedures. Ground "
    "strictly in the document — never invent. Keep it concise."
)
SYNTHESIS_INSTR = (
    "You are building the connective layer of a knowledge base from a set of document SUMMARIES. "
    "Identify the key CONCEPTS (abstract, cross-document ideas, mechanisms, standards, protocols) and "
    "ENTITIES (named things: brands, product families, models, organizations) that recur across the "
    "documents, and write a concise synthesized page for each.\n"
    "Stay FOCUSED: prefer a small number of high-value pages over many fragments. For each page, set "
    "`sources` to the titles of the documents it draws from, and use [[Title]] wikilinks to reference "
    "other concept/entity pages and the document summaries (titled 'Summary: <document>'). Each entity "
    "has a `type` (person/organization/place/product/work/event/other). Ground everything in the "
    "summaries — never invent."
)


# --------------------------------------------------------------------------- LLM seam
async def _generate(
    model: str | None, instructions: str, prompt: str, output_type: type[BaseModel]
) -> BaseModel:
    """Run one structured-output LLM call and return the parsed model. The single test seam.

    Retries a few times on a transient :class:`ModelHTTPError` (e.g. a local server that unloads an
    idle model and reloads it on the next request).
    """
    agent: Agent[None, Any] = Agent(
        resolve_model(model), output_type=output_type, instructions=instructions
    )
    last: Exception | None = None
    for attempt in range(max(1, KB_GENERATE_RETRIES + 1)):
        try:
            result = await agent.run(
                prompt, usage_limits=UsageLimits(request_limit=KB_COMPILER_REQUEST_LIMIT)
            )
            return result.output
        except ModelHTTPError as exc:  # transient model-server hiccup — back off briefly and retry
            last = exc
            logger.warning("KB _generate attempt %d failed: %s", attempt + 1, exc)
            await asyncio.sleep(1.0 * (attempt + 1))
    assert last is not None
    raise last


# --------------------------------------------------------------------------- helpers
def _is_page_kind(kind: str | None) -> bool:
    return bool(kind) and kind in _KB_PAGE_KINDS


def _compose(description: str, content: str) -> str:
    """A page body: the one-line description in bold, then the content."""
    desc = (description or "").strip()
    body = (content or "").strip()
    return f"**{desc}**\n\n{body}" if desc else body


def _pdf_to_text(b64: str) -> str:
    """Extract text from a base64-encoded PDF via pypdf (best-effort; "" on any failure)."""
    try:
        from pypdf import PdfReader  # lazy: only needed when a PDF source is present
    except Exception:  # noqa: BLE001 — pypdf not installed; degrade to no extraction.
        logger.warning("pypdf not available; cannot extract PDF text")
        return ""
    try:
        raw = base64.b64decode(b64, validate=False)
        reader = PdfReader(io.BytesIO(raw))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 — a single bad page shouldn't drop the whole doc.
                continue
        return "\n".join(parts).strip()
    except (binascii.Error, ValueError, Exception):  # noqa: BLE001 — unreadable PDF → no text.
        logger.warning("PDF text extraction failed", exc_info=True)
        return ""


def _source_text(row: dict[str, Any]) -> str:
    """The extractable text of a source document: plain text as-is, PDFs via pypdf, else ""."""
    content = row.get("content") or ""
    if row.get("encoding") == "base64":
        mime = (row.get("mime_type") or "").lower()
        title = (row.get("title") or "").lower()
        if "pdf" in mime or title.endswith(".pdf"):
            return _pdf_to_text(content)
        return ""  # other binary (images, etc.) — no text to summarize
    return content


async def _embed(embedder: Embedder, text: str) -> list[float] | None:
    if embedder is None or not getattr(embedder, "enabled", False):
        return None
    try:
        return await embedder.embed(text)
    except Exception:  # noqa: BLE001 — embedding is best-effort; degrade to LIKE search.
        logger.warning("embedding a KB page failed", exc_info=True)
        return None


def _resolve_wikilinks(body: str, title_to_id: dict[str, str]) -> tuple[str, list[str]]:
    """Strip ghost ``[[links]]`` (targets not in ``title_to_id``) and collect the resolved page ids."""
    linked: list[str] = []

    def _repl(m: "re.Match[str]") -> str:
        name = m.group(1).split("|", 1)[0].strip()
        pid = title_to_id.get(name)
        if pid:
            linked.append(pid)
            return m.group(0)
        return name

    clean = _WIKILINK_RE.sub(_repl, body or "")
    return clean, list(dict.fromkeys(linked))


def _backlink_section(header: str, titles: list[str]) -> str:
    if not titles:
        return ""
    bullets = "\n".join(f"- [[{t}]]" for t in titles)
    return f"\n\n## {header}\n{bullets}"


# --------------------------------------------------------------------------- run-scoped state
class _CompileState:
    """Carries cross-phase maps: provenance (page→sources) and source/summary titles."""

    def __init__(self) -> None:
        self.provenance: dict[str, set[str]] = {}        # page_id -> {source doc ids}
        self.summary_title_of: dict[str, str] = {}        # source id -> its summary page title
        self.source_id_of_title: dict[str, str] = {}      # source doc title -> source id


# --------------------------------------------------------------------------- phases
async def _summarize_source(
    db: ArcadeClient,
    user_id: str,
    project_id: str,
    source: dict[str, Any],
    model: str | None,
    embedder: Embedder,
    state: _CompileState,
) -> dict[str, Any] | None:
    """One short summary thread: extract text, summarize, persist the KbSummary + HAS_SUMMARY edge.

    Returns ``{title, description, content}`` for the master synthesis, or ``None`` if the source has
    no extractable text (skipped).
    """
    src_id = source.get("document_id") or ""
    src_title = source.get("title") or "(untitled)"
    full = await repo.get_document(db, user_id, src_id)
    if not full:
        return None
    text = _source_text(full)[:_MAX_SOURCE_CHARS]
    if not text.strip():
        logger.info("KB: no extractable text for source %r — skipping", src_title)
        return None

    summary = await _generate(
        model, SUMMARY_INSTR, f"Source title: {src_title}\n\nDocument:\n{text}", SummaryDraft
    )
    assert isinstance(summary, SummaryDraft)
    summary_title = f"Summary: {src_title}"
    sbody = _compose(summary.description, summary.content)
    summary_id = await repo.upsert_kb_page(
        db, user_id, project_id, "summary", summary_title, sbody,
        embedding=await _embed(embedder, sbody),
    )
    await repo.clear_kb_edges(db, user_id, src_id, "HAS_SUMMARY")
    await repo.create_kb_edge(db, user_id, "HAS_SUMMARY", src_id, summary_id)
    state.summary_title_of[src_id] = summary_title
    state.source_id_of_title[src_title] = src_id
    state.provenance.setdefault(summary_id, set()).add(src_id)
    return {"title": src_title, "description": summary.description, "content": summary.content}


def _synthesis_prompt(summaries: list[dict[str, Any]]) -> str:
    blocks = []
    for s in summaries:
        excerpt = (s.get("content") or "")[:_MAX_SUMMARY_EXCERPT]
        blocks.append(f"### {s.get('title')}\n{s.get('description', '')}\n{excerpt}")
    return (
        "Document summaries (title — description — excerpt):\n\n"
        + "\n\n".join(blocks)
        + "\n\nSynthesize the concept and entity pages now."
    )


async def _master_synthesis(
    db: ArcadeClient,
    user_id: str,
    project_id: str,
    summaries: list[dict[str, Any]],
    model: str | None,
    embedder: Embedder,
    state: _CompileState,
) -> None:
    """The single master thread: turn all summaries into concept + entity pages (the 'learnings')."""
    if not summaries:
        return
    synthesis = await _generate(model, SYNTHESIS_INSTR, _synthesis_prompt(summaries), KbSynthesis)
    assert isinstance(synthesis, KbSynthesis)

    async def _persist(page_type: str, title: str, description: str, content: str, srcs: list[str]) -> None:
        title = (title or "").strip()
        if not title:
            return
        body = _compose(description, content)
        page_id = await repo.upsert_kb_page(
            db, user_id, project_id, page_type, title, body, embedding=await _embed(embedder, body)
        )
        for st in srcs:
            sid = state.source_id_of_title.get(st)
            if sid:
                state.provenance.setdefault(page_id, set()).add(sid)

    for c in synthesis.concepts:
        await _persist("concept", c.title, c.description, c.content, c.sources)
    for e in synthesis.entities:
        await _persist("entity", e.title, e.description, e.content, e.sources)


# --------------------------------------------------------------------------- link + backlink pass
async def _link_pass(db: ArcadeClient, user_id: str, project_id: str, state: _CompileState) -> None:
    """Resolve every page's ``[[wikilinks]]``, strip ghosts, rebuild KB_LINK/MENTIONS edges, and
    (re)generate the ``## Sources``/``## Related`` backlink sections. No LLM."""
    pages = await repo.list_kb_pages_full(db, user_id, project_id)
    title_to_id = {p.get("title"): p.get("document_id") for p in pages if p.get("title") and p.get("document_id")}
    title_by_id = {pid: t for t, pid in title_to_id.items()}
    kind_by_id = {p.get("document_id"): p.get("kind") for p in pages}
    # source id -> concept/entity page titles derived from it (for a summary's "## Related").
    related_of_source: dict[str, list[str]] = {}
    for pid, srcs in state.provenance.items():
        if kind_by_id.get(pid) in ("KbConcept", "KbEntity"):
            title = title_by_id.get(pid)
            for sid in srcs:
                if title and title not in related_of_source.setdefault(sid, []):
                    related_of_source[sid].append(title)

    for p in pages:
        pid = p.get("document_id")
        body = p.get("content") or ""
        kind = p.get("kind")
        if kind == "KbSummary":
            related: list[str] = []
            for sid in state.provenance.get(pid, set()):
                for t in related_of_source.get(sid, []):
                    if t not in related:
                        related.append(t)
            body += _backlink_section("Related", related)
        elif kind in ("KbConcept", "KbEntity"):
            src_titles = [
                state.summary_title_of[sid]
                for sid in state.provenance.get(pid, set())
                if sid in state.summary_title_of
            ]
            body += _backlink_section("Sources", src_titles)

        clean, linked_ids = _resolve_wikilinks(body, title_to_id)
        linked_ids = [i for i in linked_ids if i != pid]
        if clean != (p.get("content") or ""):
            await repo.update_document(db, user_id, pid, content=clean)
        await repo.clear_kb_edges(db, user_id, pid, "KB_LINK")
        for tid in linked_ids:
            await repo.create_kb_edge(db, user_id, "KB_LINK", pid, tid)
        await repo.clear_kb_edges(db, user_id, pid, "MENTIONS")
        for sid in state.provenance.get(pid, set()):
            await repo.create_kb_edge(db, user_id, "MENTIONS", pid, sid)


async def _refresh_index(
    db: ArcadeClient, user_id: str, project_id: str, embedder: Embedder
) -> None:
    """(Re)build the KbIndex overview page listing the knowledge base's pages. Code-built, no LLM."""
    pages = await repo.list_kb_pages_full(db, user_id, project_id)
    groups: dict[str, list[str]] = {"KbConcept": [], "KbEntity": [], "KbSummary": []}
    for p in pages:
        if p.get("kind") in groups and p.get("title") != "Index":
            groups[p["kind"]].append(p["title"])
    sections = []
    for kind, header in (("KbConcept", "Concepts"), ("KbEntity", "Entities"), ("KbSummary", "Summaries")):
        titles = sorted(groups[kind])
        if titles:
            sections.append(f"## {header}\n" + "\n".join(f"- [[{t}]]" for t in titles))
    if not sections:
        return
    body = "**Overview of this project's knowledge base.**\n\n" + "\n\n".join(sections)
    await repo.upsert_kb_page(
        db, user_id, project_id, "index", "Index", body, embedding=await _embed(embedder, body)
    )


# --------------------------------------------------------------------------- entry point
async def compile_project_kb(
    db: ArcadeClient,
    user_id: str,
    project_id: str,
    model: str | None = None,
    full: bool = False,
) -> None:
    """Compile a project's knowledge base from its source documents. Best-effort: never raises.

    Marks the project ``kb_status='compiling'``, runs short parallel summary threads, one master
    synthesis thread, then a code link/backlink pass, and stamps ``kb_status='idle'`` +
    ``kb_compiled_at`` (or ``'error'`` on failure). Skips quietly when there are no sources.
    """
    if not KB_COMPILE_ENABLED:
        return
    try:
        await repo.set_project_kb_status(db, user_id, project_id, "compiling")
        all_docs = await repo.list_documents(
            db, user_id, project_id=project_id, include_global=True, limit=500
        )
        sources = [d for d in all_docs if not _is_page_kind(d.get("kind"))]
        if not sources:
            await repo.set_project_kb_status(db, user_id, project_id, "idle")
            return
        state = _CompileState()
        async with Embedder.from_env() as embedder:
            # Phase 1 + 2: short summary threads, run in parallel (bounded).
            sem = asyncio.Semaphore(max(1, KB_COMPILE_MAX_PARALLEL))

            async def _one(src: dict[str, Any]) -> dict[str, Any] | None:
                async with sem:
                    try:
                        return await _summarize_source(
                            db, user_id, project_id, src, model, embedder, state
                        )
                    except Exception:  # noqa: BLE001 — one bad source must not sink the compile.
                        logger.warning("summarizing %r failed", src.get("title"), exc_info=True)
                        return None

            results = await asyncio.gather(*[_one(s) for s in sources])
            summaries = [r for r in results if r]
            logger.info(
                "KB: summarized %d/%d sources for project %s", len(summaries), len(sources), project_id
            )

            # Phase 3: one master synthesis thread over the summaries.
            try:
                await _master_synthesis(db, user_id, project_id, summaries, model, embedder, state)
            except Exception:  # noqa: BLE001 — synthesis is the riskiest call; keep the summaries.
                logger.warning("KB master synthesis failed for %s", project_id, exc_info=True)

            # Phase 4: index + link/backlink/edge pass (no LLM).
            await _refresh_index(db, user_id, project_id, embedder)
            await _link_pass(db, user_id, project_id, state)
        await repo.set_project_kb_status(
            db, user_id, project_id, "idle", compiled_at=datetime.now(timezone.utc).isoformat()
        )
        logger.info("compiled knowledge base for project %s (full=%s)", project_id, full)
    except Exception:  # noqa: BLE001 — KB compilation is best-effort; never crash the caller.
        logger.warning("knowledge-base compilation for project %s failed", project_id, exc_info=True)
        try:
            await repo.set_project_kb_status(db, user_id, project_id, "error")
        except Exception:  # noqa: BLE001 — even the status write is best-effort.
            logger.debug("could not mark KB status=error for project %s", project_id, exc_info=True)


# --------------------------------------------------------------------------- background scheduling
# Per-(user, project) debounce: a pending sleep-then-compile task, cancelled+rescheduled by a newer
# upload so a burst of uploads coalesces into one compile. In-process (single uvicorn worker on the
# local network); a second worker would just run its own coalesced compile — harmless (upsert).
_ClientFactory = Callable[..., Any]  # _client_for: (user_id, *, ensure=bool) -> async context manager
_debounce_tasks: dict[tuple[str, str], "asyncio.Task[None]"] = {}


async def _run_compile(
    user_id: str, project_id: str, client_factory: _ClientFactory, full: bool
) -> None:
    """Open a fresh per-user client (API endpoints use short-lived clients) and run one compile."""
    try:
        async with client_factory(user_id, ensure=True) as db:
            await compile_project_kb(db, user_id, project_id, full=full)
    except Exception:  # noqa: BLE001 — background task; swallow so it can't surface anywhere.
        logger.warning("background KB compile for project %s failed", project_id, exc_info=True)


async def _debounced(
    key: tuple[str, str],
    user_id: str,
    project_id: str,
    client_factory: _ClientFactory,
    full: bool,
    delay: float,
) -> None:
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return  # a newer upload rescheduled us during the debounce window
    # Past the debounce window: drop ourselves so a later upload schedules a fresh run (and is not
    # cancelled by us mid-compile).
    _debounce_tasks.pop(key, None)
    await _run_compile(user_id, project_id, client_factory, full)


def schedule_kb_compile(
    user_id: str,
    project_id: str,
    client_factory: _ClientFactory,
    *,
    full: bool = False,
    delay: float | None = None,
) -> None:
    """Schedule a (debounced) background KB compile for a project. No-op when disabled.

    Called best-effort from the upload endpoint. ``delay`` defaults to the debounce window; the
    manual "Rebuild" path passes ``delay=0, full=True`` to run promptly. Requires a running event
    loop (always true under the API server); if none is running it logs and returns.
    """
    if not KB_COMPILE_ENABLED:
        return
    wait = KB_COMPILE_DEBOUNCE_SECONDS if delay is None else delay
    key = (user_id, project_id)
    existing = _debounce_tasks.get(key)
    if existing is not None and not existing.done():
        existing.cancel()
    try:
        task = asyncio.create_task(
            _debounced(key, user_id, project_id, client_factory, full, wait)
        )
    except RuntimeError:  # no running loop (e.g. called outside the server) — nothing to schedule on
        logger.debug("no event loop to schedule KB compile for project %s", project_id)
        return
    _debounce_tasks[key] = task

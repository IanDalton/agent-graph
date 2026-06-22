"""Tests for the background knowledge-base compiler (v2: deterministic OpenKB-style pipeline).

Unit tests use an in-memory ArcadeClient stand-in and monkeypatch the single `_generate` LLM seam to
return canned structured outputs — no database, network, or model. They assert the pipeline writes the
planned pages, builds the typed edges (HAS_SUMMARY / MENTIONS / KB_LINK), strips ghost wikilinks, and
that upload scheduling debounces.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from backend import kb_compiler
from backend.schemas.kb_schemas import (
    KbConceptOut,
    KbEntityOut,
    KbSynthesis,
    SummaryDraft,
)

_KB_PAGE_KINDS = {"KbPage", "KbSummary", "KbConcept", "KbEntity", "KbExploration", "KbIndex"}


class KbClient:
    """In-memory ArcadeClient stand-in with a coherent document + edge store.

    Implements the query/command surface the repo's document/KB/edge helpers use: create/update docs,
    lookup-by-title (KbPage subtree), list (Document vs KbPage), typed edges (create/clear), and the
    Project kb_status writes. Seeded with source documents.
    """

    def __init__(self, sources: list[dict[str, Any]] | None = None) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.edges: list[tuple[str, str, str]] = []  # (edge_type, from_id, to_id)
        self.kb_status: list[str] = []
        self.docs: dict[str, dict[str, Any]] = {}
        for s in sources or []:
            self.docs[s["document_id"]] = {
                "kind": "KbSource", "encoding": "text", "mime_type": "text/markdown",
                "is_global": False, "project_id": "p1", "content": "", **s,
            }

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        p = params or {}
        self.commands.append((sql, p))
        upper = sql.upper()
        if upper.startswith("CREATE VERTEX"):
            vtype = sql.split()[2]
            self.docs[p["did"]] = {
                "document_id": p["did"], "kind": vtype, "title": p.get("title", ""),
                "content": p.get("content", ""), "mime_type": p.get("mime", "text/markdown"),
                "encoding": p.get("enc", "text"), "is_global": False, "project_id": p.get("pid"),
            }
        elif upper.startswith("CREATE EDGE"):
            self.edges.append((sql.split()[2], p.get("fid", ""), p.get("tid", "")))
        elif upper.startswith("DELETE FROM") and "UNSAFE" in upper:  # clear_kb_edges
            etype, fid = sql.split()[2], p.get("fid", "")
            before = len(self.edges)
            self.edges = [e for e in self.edges if not (e[0] == etype and e[1] == fid)]
            return [{"count": before - len(self.edges)}]
        elif upper.startswith("UPDATE DOCUMENT SET") and "did" in p:
            doc = self.docs.get(p["did"])
            if doc is not None and "content" in p:
                doc["content"] = p["content"]
            return [{"count": 1}]
        elif upper.startswith("UPDATE PROJECT SET KB_STATUS"):
            self.kb_status.append(p.get("st", ""))
            return [{"count": 1}]
        elif upper.startswith(("UPDATE", "DELETE")):
            return [{"count": 1}]
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        p = params or {}
        upper = sql.upper()
        if "TITLE = :TITLE" in upper:  # get_kb_page_by_title (KbPage subtree only)
            for d in self.docs.values():
                if d["kind"] in _KB_PAGE_KINDS and d.get("title") == p.get("title") and d.get("project_id") == p.get("pid"):
                    return [{"kind": d["kind"], "document_id": d["document_id"], "title": d["title"]}]
            return []
        if "FROM KBPAGE" in upper:  # list_kb_pages_full / list_kb_page_index (full dicts are fine)
            return [dict(d) for d in self.docs.values() if d["kind"] in _KB_PAGE_KINDS]
        if "DOCUMENT_ID = :DID" in upper:  # get_document
            d = self.docs.get(p.get("did"))
            return [dict(d)] if d else []
        if "FROM DOCUMENT WHERE" in upper:  # list_documents(from_type="Document")
            return [dict(d) for d in self.docs.values()]
        return []

    # convenience for assertions
    def doc_by_title(self, title: str) -> dict[str, Any] | None:
        return next((d for d in self.docs.values() if d.get("title") == title), None)


def _fake_generate():
    """A canned `_generate` replacement keyed by output_type: a summary, then the master synthesis."""

    async def gen(model, instructions, prompt, output_type):  # noqa: ANN001
        if output_type is SummaryDraft:
            return SummaryDraft(description="About Doc A.", content="Doc A is about intercoms.")
        if output_type is KbSynthesis:
            return KbSynthesis(
                concepts=[
                    KbConceptOut(
                        title="Concept X",
                        description="An idea.",
                        content="Relates to [[Entity Y]] and a [[Ghost Link]] that doesn't exist.",
                        sources=["Doc A"],
                    )
                ],
                entities=[
                    KbEntityOut(title="Entity Y", type="person", description="A person.",
                                content="Mentioned in [[Concept X]].", sources=["Doc A"]),
                ],
            )
        raise AssertionError(f"unexpected output_type {output_type}")

    return gen


def test_pipeline_writes_pages_edges_and_strips_ghosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kb_compiler, "_generate", _fake_generate())
    db = KbClient(sources=[{"document_id": "s1", "title": "Doc A", "content": "Body of A."}])

    asyncio.run(kb_compiler.compile_project_kb(db, "u", "p1"))

    # The planned pages were created with the right vertex types (plus the index).
    created = [sql.split()[2] for sql, _ in db.commands if sql.upper().startswith("CREATE VERTEX")]
    assert {"KbSummary", "KbConcept", "KbEntity", "KbIndex"} <= set(created)

    # Typed edges exist: a source->summary HAS_SUMMARY, provenance MENTIONS, and resolved KB_LINKs.
    types = {e[0] for e in db.edges}
    assert "HAS_SUMMARY" in types and "MENTIONS" in types and "KB_LINK" in types
    # MENTIONS goes from a KB page to the source document s1.
    assert any(e[0] == "MENTIONS" and e[2] == "s1" for e in db.edges)

    # Ghost wikilink stripped from the concept body; the valid one kept.
    concept = db.doc_by_title("Concept X")
    assert concept is not None
    assert "[[Ghost Link]]" not in concept["content"] and "Ghost Link" in concept["content"]
    assert "[[Entity Y]]" in concept["content"]
    # The concept got a code-built "## Sources" backlink section; the summary a "## Related" one.
    assert "## Sources" in concept["content"]
    summary = db.doc_by_title("Summary: Doc A")
    assert summary is not None and "## Related" in summary["content"]

    # Status went compiling -> idle, with a compiled_at stamp.
    assert db.kb_status[0] == "compiling" and db.kb_status[-1] == "idle"
    assert any("kb_compiled_at = :ca" in sql for sql, _ in db.commands)


def test_compile_skips_when_no_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    async def _spy(*_a, **_k):  # _generate must never be invoked when there are no sources
        called["n"] += 1
        raise AssertionError("should not generate")

    monkeypatch.setattr(kb_compiler, "_generate", _spy)
    db = KbClient(sources=[])
    asyncio.run(kb_compiler.compile_project_kb(db, "u", "p1"))
    assert called["n"] == 0
    assert not any(sql.upper().startswith("CREATE VERTEX") for sql, _ in db.commands)
    assert db.kb_status and db.kb_status[-1] == "idle"


def test_compile_disabled_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kb_compiler, "KB_COMPILE_ENABLED", False)
    db = KbClient(sources=[{"document_id": "s1", "title": "Doc A", "content": "x"}])
    asyncio.run(kb_compiler.compile_project_kb(db, "u", "p1"))
    assert not db.commands


def test_schedule_debounce_coalesces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two uploads in quick succession schedule one compile (the second cancels the first's wait)."""
    calls: list[tuple[str, str]] = []

    async def _fake_run(user_id, project_id, _factory, _full):  # noqa: ANN001
        calls.append((user_id, project_id))

    monkeypatch.setattr(kb_compiler, "_run_compile", _fake_run)
    monkeypatch.setattr(kb_compiler, "KB_COMPILE_DEBOUNCE_SECONDS", 0.05)
    monkeypatch.setattr(kb_compiler, "KB_COMPILE_ENABLED", True)
    kb_compiler._debounce_tasks.clear()

    @asynccontextmanager
    async def _factory(_uid, *, ensure=False):  # noqa: ANN001
        yield None

    async def go() -> None:
        kb_compiler.schedule_kb_compile("u", "p1", _factory)
        kb_compiler.schedule_kb_compile("u", "p1", _factory)  # cancels + reschedules the first
        await asyncio.sleep(0.2)

    asyncio.run(go())
    assert calls == [("u", "p1")]


def test_schedule_disabled_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kb_compiler, "KB_COMPILE_ENABLED", False)
    kb_compiler._debounce_tasks.clear()

    async def go() -> None:
        kb_compiler.schedule_kb_compile("u", "p1", lambda *_a, **_k: None)
        await asyncio.sleep(0.05)

    asyncio.run(go())
    assert not kb_compiler._debounce_tasks

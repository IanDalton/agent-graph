"""Tests for the Documents capability (agent-authored, user-editable documents).

All unit tests use a duck-typed fake ArcadeClient, so they need no database. The tools are
plain coroutines callable with a hand-built RunContext, like the other capability tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db.dependencies import GraphDependencies
from backend.schemas.document_schemas import CreateDocumentArgs, UpdateDocumentArgs
from backend.skills.document_capability import (
    build_documents,
    create_document,
    delete_document,
    list_documents,
    read_document,
    update_document,
)

EXPECTED_TOOLS = {
    "create_document",
    "update_document",
    "read_document",
    "list_documents",
    "delete_document",
}


class FakeDb:
    """Duck-typed ArcadeClient: records commands, returns canned query rows."""

    def __init__(self, *, rows: list[dict[str, Any]] | None = None, affected: int = 1) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._rows = rows or []
        self._affected = affected

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        if sql.strip().upper().startswith(("UPDATE", "DELETE")):
            return [{"count": self._affected}]
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((sql, params or {}))
        return self._rows


def _ctx(db: FakeDb) -> RunContext[GraphDependencies]:
    deps = GraphDependencies(db=db, user_id="u", conversation_id="c")
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #
def test_tools_are_registered() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, deps_type=GraphDependencies, capabilities=[*build_documents()])
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c")
    asyncio.run(agent.run("hi", deps=deps))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert EXPECTED_TOOLS <= names


# --------------------------------------------------------------------------- #
# create_document
# --------------------------------------------------------------------------- #
def test_create_document_persists_and_links() -> None:
    db = FakeDb()
    info = asyncio.run(
        create_document(
            _ctx(db), CreateDocumentArgs(title="Plan", content="# Plan\n- step 1")
        )
    )
    assert info.document_id
    assert info.mime_type == "text/markdown"  # default
    assert info.conversation_id == "c"
    create = next(p for s, p in db.commands if s.startswith("CREATE VERTEX Document"))
    assert create["title"] == "Plan"
    assert create["uid"] == "u" and create["cid"] == "c"
    assert any(s.startswith("CREATE EDGE HAS_DOCUMENT") for s, _ in db.commands)


def test_create_document_args_normalize_and_reject_mime() -> None:
    assert CreateDocumentArgs(title="t", content="c", mime_type="TEXT/Plain").mime_type == "text/plain"
    with pytest.raises(ValidationError):
        CreateDocumentArgs(title="t", content="c", mime_type="not a mime")


# --------------------------------------------------------------------------- #
# update_document / delete_document
# --------------------------------------------------------------------------- #
def test_update_document_revises_in_place() -> None:
    db = FakeDb(affected=1)
    msg = asyncio.run(
        update_document(_ctx(db), UpdateDocumentArgs(document_id="d1", content="new body"))
    )
    assert "Updated" in msg
    sql, params = next((s, p) for s, p in db.commands if s.startswith("UPDATE Document"))
    assert params["content"] == "new body" and params["did"] == "d1" and params["uid"] == "u"
    assert "title" not in params  # untouched fields are not overwritten


def test_update_document_requires_some_change() -> None:
    with pytest.raises(ModelRetry):
        asyncio.run(update_document(_ctx(FakeDb()), UpdateDocumentArgs(document_id="d1")))


def test_update_document_unknown_id_is_model_retry() -> None:
    db = FakeDb(affected=0)
    with pytest.raises(ModelRetry):
        asyncio.run(
            update_document(_ctx(db), UpdateDocumentArgs(document_id="nope", content="x"))
        )


def test_delete_document_unknown_id_is_model_retry() -> None:
    db = FakeDb(affected=0)
    with pytest.raises(ModelRetry):
        asyncio.run(delete_document(_ctx(db), "nope"))


# --------------------------------------------------------------------------- #
# read_document / list_documents
# --------------------------------------------------------------------------- #
def test_read_document_returns_current_body() -> None:
    row = {
        "document_id": "d1",
        "title": "Plan",
        "mime_type": "text/markdown",
        "content": "# user-edited",
        "created_at": "2026-06-12T00:00:00+00:00",
        "updated_at": "2026-06-12T01:00:00+00:00",
    }
    doc = asyncio.run(read_document(_ctx(FakeDb(rows=[row])), "d1"))
    assert doc.content == "# user-edited"
    assert doc.title == "Plan"


def test_read_document_unknown_id_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        asyncio.run(read_document(_ctx(FakeDb(rows=[])), "nope"))


def test_list_documents_scopes_to_conversation() -> None:
    db = FakeDb(rows=[{"document_id": "d1", "title": "Plan", "mime_type": "text/markdown"}])
    docs = asyncio.run(list_documents(_ctx(db)))
    assert [d.document_id for d in docs] == ["d1"]
    _, params = db.queries[0]
    assert params["cid"] == "c" and params["uid"] == "u"


# --------------------------------------------------------------------------- #
# stream_run's document frames (the UI artifact card + side-panel spotlight)
# --------------------------------------------------------------------------- #
def test_document_events_from_create_result() -> None:
    from backend.main import _document_events
    from backend.schemas.document_schemas import DocumentInfo

    info = DocumentInfo(document_id="d1", title="Plan", mime_type="text/markdown")
    events = _document_events("create_document", info, args=None)
    assert events == [
        {
            "type": "document",
            "action": "created",
            "document_id": "d1",
            "title": "Plan",
            "mime_type": "text/markdown",
        }
    ]


def test_document_events_from_update_args() -> None:
    """update_document returns a string, so the id must come from the call's (nested) args."""
    from backend.main import _document_events

    events = _document_events(
        "update_document", "Updated document d1.", args={"args": {"document_id": "d1"}}
    )
    assert len(events) == 1
    assert events[0]["action"] == "updated" and events[0]["document_id"] == "d1"


def test_document_events_from_run_python_artifacts() -> None:
    """Every /out file the sandbox persisted gets its own 'created' frame."""
    from backend.main import _document_events
    from backend.schemas.document_schemas import DocumentInfo
    from backend.schemas.sandbox_schemas import PythonRunResult

    result = PythonRunResult(
        stdout="",
        exit_code=0,
        documents=[
            DocumentInfo(document_id="d1", title="report.pdf", mime_type="application/pdf"),
            DocumentInfo(document_id="d2", title="data.csv", mime_type="text/csv"),
        ],
    )
    events = _document_events("run_python", result, args=None)
    assert [(e["document_id"], e["action"]) for e in events] == [
        ("d1", "created"),
        ("d2", "created"),
    ]


def test_document_events_ignores_other_tools_and_bad_results() -> None:
    from backend.main import _document_events

    assert _document_events("store_fact", {"text": "x"}, args=None) == []
    assert _document_events("create_document", "unexpected string result", args=None) == []
    assert _document_events("update_document", "msg", args={"args": {}}) == []


def test_jsonable_handles_models_inside_containers() -> None:
    """Regression: list_documents returns list[DocumentInfo]; json.dumps on the raw tool_result
    frame killed the SSE stream (TypeError: DocumentInfo is not JSON serializable)."""
    import json

    from backend.main import _jsonable
    from backend.schemas.document_schemas import DocumentInfo

    payload = [
        DocumentInfo(document_id="d1", title="Plan", mime_type="text/markdown"),
        {"nested": DocumentInfo(document_id="d2", title="B", mime_type="text/plain")},
    ]
    out = _jsonable(payload)
    encoded = json.dumps(out)  # must not raise
    assert '"d1"' in encoded and '"d2"' in encoded

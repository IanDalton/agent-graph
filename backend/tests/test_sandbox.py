"""Tests for the PythonSandbox capability (containerized code execution + file artifacts).

Unit tests inject a fake sandbox via deps.sandbox (or monkeypatch subprocess creation), so they
need no Docker. The integration tests actually run containers and are skipped unless the Docker
daemon is reachable AND the sandbox image is already present locally (so a unit-test run never
pulls or builds an image).
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db.dependencies import GraphDependencies
from backend.sandbox.runner import (
    DEFAULT_IMAGE,
    FALLBACK_IMAGE,
    MAX_FILES,
    PythonSandbox,
    SandboxFile,
    SandboxResult,
    _collect_files,
)
from backend.schemas.sandbox_schemas import RunPythonArgs
from backend.skills.sandbox_capability import _mime_for, build_sandbox, run_python


class FakeSandbox:
    """Duck-typed PythonSandbox returning a canned result (or raising)."""

    def __init__(self, result: SandboxResult | None = None, error: Exception | None = None) -> None:
        self._result = result or SandboxResult(stdout="ok\n", exit_code=0)
        self._error = error
        self.calls: list[tuple[str, float | None]] = []

    async def run(self, code: str, timeout_seconds: float | None = None) -> SandboxResult:
        self.calls.append((code, timeout_seconds))
        if self._error:
            raise self._error
        return self._result


class FakeDb:
    """Duck-typed ArcadeClient recording commands (for the document-persistence path)."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return []


def _ctx(sandbox: Any, db: Any | None = None) -> RunContext[GraphDependencies]:
    deps = GraphDependencies(
        db=db or FakeDb(), user_id="u", conversation_id="c", sandbox=sandbox
    )
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #
def test_tool_is_registered() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, deps_type=GraphDependencies, capabilities=[*build_sandbox()])
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", sandbox=FakeSandbox())
    asyncio.run(agent.run("hi", deps=deps))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert "run_python" in names


# --------------------------------------------------------------------------- #
# run_python tool
# --------------------------------------------------------------------------- #
def test_run_python_returns_captured_output() -> None:
    sandbox = FakeSandbox(SandboxResult(stdout="4\n", stderr="", exit_code=0))
    result = asyncio.run(run_python(_ctx(sandbox), RunPythonArgs(code="print(2+2)")))
    assert result.stdout == "4\n"
    assert result.exit_code == 0
    assert result.error is None
    assert sandbox.calls == [("print(2+2)", None)]


def test_run_python_passes_timeout_override() -> None:
    sandbox = FakeSandbox()
    asyncio.run(run_python(_ctx(sandbox), RunPythonArgs(code="x", timeout_seconds=60)))
    assert sandbox.calls[0][1] == 60.0


def test_run_python_error_is_returned_not_raised() -> None:
    """An unexpected sandbox failure must come back as an error result, never abort the run."""
    sandbox = FakeSandbox(error=RuntimeError("docker exploded"))
    result = asyncio.run(run_python(_ctx(sandbox), RunPythonArgs(code="x")))
    assert result.error and "docker exploded" in result.error
    assert result.stdout == ""


def test_run_python_surfaces_sandbox_error_field() -> None:
    sandbox = FakeSandbox(SandboxResult(timed_out=True, error="Execution exceeded the 30s time limit"))
    result = asyncio.run(run_python(_ctx(sandbox), RunPythonArgs(code="while True: pass")))
    assert result.timed_out is True
    assert result.error and "time limit" in result.error


def test_run_python_persists_out_files_as_documents() -> None:
    """A /out PDF becomes a base64 Document; a /out CSV a text Document; both stream back ids."""
    sandbox = FakeSandbox(
        SandboxResult(
            stdout="",
            exit_code=0,
            files=[
                SandboxFile(name="report.pdf", data=b"%PDF-1.4 fake"),
                SandboxFile(name="data.csv", data=b"a,b\n1,2\n"),
            ],
        )
    )
    db = FakeDb()
    result = asyncio.run(run_python(_ctx(sandbox, db), RunPythonArgs(code="...")))

    by_title = {d.title: d for d in result.documents}
    assert by_title["report.pdf"].mime_type == "application/pdf"
    assert by_title["report.pdf"].encoding == "base64"
    assert by_title["data.csv"].mime_type == "text/csv"
    assert by_title["data.csv"].encoding == "text"
    assert all(d.document_id for d in result.documents)

    creates = [p for s, p in db.commands if s.startswith("CREATE VERTEX Document")]
    assert {p["title"] for p in creates} == {"report.pdf", "data.csv"}
    assert all(p["uid"] == "u" and p["cid"] == "c" for p in creates)
    # The PDF body was base64'd (pure ASCII, decodable), not raw bytes.
    import base64 as b64

    pdf = next(p for p in creates if p["title"] == "report.pdf")
    assert b64.b64decode(pdf["content"]) == b"%PDF-1.4 fake"


def test_run_python_document_persistence_failure_is_a_note_not_a_crash() -> None:
    class BrokenDb(FakeDb):
        async def command(self, sql: str, params: dict[str, Any] | None = None):
            raise RuntimeError("db down")

    sandbox = FakeSandbox(
        SandboxResult(stdout="done\n", exit_code=0, files=[SandboxFile("x.pdf", b"%PDF")])
    )
    result = asyncio.run(run_python(_ctx(sandbox, BrokenDb()), RunPythonArgs(code="...")))
    assert result.stdout == "done\n"  # the code's output survives
    assert result.documents == []
    assert any("could not be saved" in n for n in result.notes)


# --------------------------------------------------------------------------- #
# RunPythonArgs validation / mime mapping
# --------------------------------------------------------------------------- #
def test_args_reject_empty_code_and_bad_timeouts() -> None:
    with pytest.raises(ValidationError):
        RunPythonArgs(code="")
    with pytest.raises(ValidationError):
        RunPythonArgs(code="x", timeout_seconds=0)
    with pytest.raises(ValidationError):
        RunPythonArgs(code="x", timeout_seconds=600)


def test_mime_for_is_deterministic_for_known_artifacts() -> None:
    assert _mime_for("report.pdf") == "application/pdf"
    assert _mime_for("notes.md") == "text/markdown"
    assert _mime_for("page.HTML") == "text/html"
    assert _mime_for("mystery.zzz") == "application/octet-stream"


# --------------------------------------------------------------------------- #
# PythonSandbox runner
# --------------------------------------------------------------------------- #
def test_docker_args_lock_the_container_down(tmp_path: Path) -> None:
    args = PythonSandbox()._docker_args("name1", DEFAULT_IMAGE, str(tmp_path))
    joined = " ".join(args)
    assert args[:3] == ["docker", "run", "--rm"]
    for flag in ("--network none", "--cap-drop ALL", "--read-only", "--pids-limit 128",
                 "--security-opt no-new-privileges", "--user 65534:65534",
                 f"--volume {tmp_path}:/out", "--env OUTPUT_DIR=/out"):
        assert flag in joined
    assert args[-3:] == ["python", "-I", "-"]  # code arrives on stdin, never via a shell


def test_run_reports_docker_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*_a: Any, **_kw: Any):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    result = asyncio.run(PythonSandbox().run("print(1)"))
    assert result.error and "Docker is not available" in result.error
    assert result.exit_code is None


def test_run_caps_oversized_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProc:
        returncode = 0

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            return b"x" * 200_000, b""

    async def fake_exec(*_a: Any, **_kw: Any) -> FakeProc:
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(PythonSandbox().run("print('x' * 200000)"))
    assert result.truncated is True
    assert len(result.stdout) <= 64_000


def test_run_falls_back_to_base_image_when_unbuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing agent-sandbox image retries once on python:3.12-slim with a note."""
    attempts: list[str] = []

    async def fake_run_once(
        self: PythonSandbox, code: str, image: str, timeout: float, skills_dir: str | None = None
    ) -> SandboxResult:
        attempts.append(image)
        if image == DEFAULT_IMAGE:
            return SandboxResult(
                stderr="Unable to find image 'agent-sandbox:latest' locally", exit_code=125
            )
        return SandboxResult(stdout="4\n", exit_code=0)

    monkeypatch.setattr(PythonSandbox, "_run_once", fake_run_once)
    result = asyncio.run(PythonSandbox(image=DEFAULT_IMAGE).run("print(2+2)"))
    assert attempts == [DEFAULT_IMAGE, FALLBACK_IMAGE]
    assert result.exit_code == 0 and result.stdout == "4\n"
    assert any("fallback image" in n for n in result.notes)


def test_collect_files_caps_count_and_size(tmp_path: Path) -> None:
    for i in range(MAX_FILES + 2):
        (tmp_path / f"f{i:02d}.txt").write_bytes(b"data")
    (tmp_path / "a_huge.bin").write_bytes(b"x" * 6_000_000)
    (tmp_path / "subdir").mkdir()  # directories are ignored

    files, notes = _collect_files(str(tmp_path))
    assert len(files) == MAX_FILES
    assert all(f.data == b"data" for f in files if f.name.startswith("f"))
    assert any("exceeded" in n for n in notes)  # the oversized file
    assert any("first" in n for n in notes)  # the overflow


# --------------------------------------------------------------------------- #
# Integration (requires a reachable Docker daemon AND a locally present image)
# --------------------------------------------------------------------------- #
def _image_available(image: str) -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "image", "inspect", image], capture_output=True, timeout=15
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(
    not _image_available(DEFAULT_IMAGE),
    reason=f"Docker daemon or image {DEFAULT_IMAGE} unavailable (docker compose build sandbox)",
)
def test_integration_runs_real_python() -> None:
    result = asyncio.run(PythonSandbox().run("print(2 + 2)"))
    assert result.error is None
    assert result.exit_code == 0
    assert result.stdout.strip() == "4"


@pytest.mark.skipif(
    not _image_available(DEFAULT_IMAGE),
    reason=f"Docker daemon or image {DEFAULT_IMAGE} unavailable (docker compose build sandbox)",
)
def test_integration_generates_a_pdf_artifact() -> None:
    code = (
        "from fpdf import FPDF\n"
        "pdf = FPDF()\n"
        "pdf.add_page()\n"
        "pdf.set_font('Helvetica', size=12)\n"
        "pdf.cell(0, 10, 'hello')\n"
        "pdf.output('/out/report.pdf')\n"
        "print('written')\n"
    )
    result = asyncio.run(PythonSandbox().run(code))
    assert result.error is None and result.exit_code == 0, result.stderr
    assert [f.name for f in result.files] == ["report.pdf"]
    assert result.files[0].data.startswith(b"%PDF")

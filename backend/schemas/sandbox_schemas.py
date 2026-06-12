"""Pydantic models for the PythonSandbox capability's tool inputs/outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.schemas.document_schemas import DocumentInfo


class RunPythonArgs(BaseModel):
    """A Python snippet the agent wants to execute in the sandbox."""

    code: str = Field(
        ...,
        min_length=1,
        description=(
            "A complete, self-contained Python program (standard library only). "
            "print() anything you need to see — only stdout/stderr come back."
        ),
    )
    timeout_seconds: int | None = Field(
        None,
        ge=1,
        le=120,
        description="Optional wall-clock limit override in seconds (default 30, max 120).",
    )


class PythonRunResult(BaseModel):
    """Structured result returned by the run_python tool."""

    stdout: str = Field("", description="Everything the program printed to stdout.")
    stderr: str = Field("", description="Warnings/tracebacks; read this when exit_code != 0.")
    exit_code: int | None = Field(
        None, description="The program's exit code (0 = success); null if it never ran."
    )
    timed_out: bool = Field(False, description="True if the run hit the time limit and was stopped.")
    truncated: bool = Field(False, description="True if stdout/stderr exceeded the size cap and were cut.")
    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description=(
            "Files the program wrote to /out, already saved as documents the user can see "
            "(PDFs, images, HTML, CSVs...). Reference them by title; no need to re-create them."
        ),
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Sandbox-side remarks (dropped/oversized files, image fallback). Read them.",
    )
    error: str | None = Field(
        None,
        description="Set when the sandbox itself failed (Docker offline, timeout); code output is empty.",
    )

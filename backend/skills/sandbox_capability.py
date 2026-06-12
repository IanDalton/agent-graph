"""PythonSandbox capability: let the agent run Python code in a locked-down container.

Exposed via :func:`build_sandbox`, dropped into ``Agent(capabilities=...)``. One tool,
``run_python``, executes a self-contained snippet in a fresh ephemeral Docker container per call
(see :class:`backend.sandbox.runner.PythonSandbox`: no network, capped memory/CPU/time, read-only
root FS, non-root user).

**Artifacts:** files the program writes to ``/out`` come back from the sandbox and are persisted
here as ``Document`` vertices (text files as literal text, binary files — PDFs, images — base64),
so they appear in the web UI's Documents pane like any other agent-authored document. The
capability instructions carry the "how to make a PDF" recipe (fpdf2 ships in the project's
``agent-sandbox`` image).

The sandbox is taken from ``ctx.deps.sandbox`` when present (test injection), else a short-lived
one is built from env for that call — the same standalone/injected pattern as the web tools.

Safety contract (same as ``run_query``/``web_search``): this tool must NEVER abort the run.
Docker being offline, a timeout, or a crashing program is caught and returned as a structured
result the model can read and react to, not raised.
"""

from __future__ import annotations

import base64
import logging
import mimetypes

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.sandbox.runner import PythonSandbox, SandboxFile
from backend.schemas.document_schemas import DocumentInfo
from backend.schemas.sandbox_schemas import PythonRunResult, RunPythonArgs

logger = logging.getLogger("agent_graph.sandbox")

INSTRUCTIONS = (
    "You can EXECUTE PYTHON CODE in a secure sandbox with `run_python`. Use it for anything that "
    "benefits from real computation instead of mental arithmetic: math, data wrangling, parsing, "
    "date calculations, simulations, validating an algorithm, checking your own claims — and for "
    "PRODUCING FILE ARTIFACTS like PDFs.\n"
    "HOW IT WORKS — each call runs in a FRESH, isolated container:\n"
    "  - STATELESS: no variables, files or imports survive between calls. Send a complete, "
    "self-contained program every time.\n"
    "  - NO NETWORK. For internet data use web_search/fetch_url and paste what you need into the "
    "code as a literal.\n"
    "  - LIBRARIES: the standard library plus `fpdf` (the fpdf2 PDF library). No other "
    "third-party packages and no pip.\n"
    "  - OUTPUT: only stdout/stderr come back — print() every result you need. A non-zero "
    "exit_code means the program crashed; read the traceback in stderr, fix the code, retry.\n"
    "  - FILES: anything your program writes to the /out directory is AUTOMATICALLY saved as a "
    "document and shown to the user (returned in the result's `documents` with their ids). Write "
    "flat files with proper extensions (/out/report.pdf, /out/data.csv); max 8 files, 5MB each. "
    "Do NOT also call create_document for them — they are already documents.\n"
    "  - LIMITS: ~30s wall clock (raise timeout_seconds up to 120 for genuinely heavy work), "
    "capped memory and output size. Read `notes` in the result for anything the sandbox dropped.\n"
    "CREATING A PDF — use fpdf2 and write to /out. The recipe:\n"
    "    from fpdf import FPDF\n"
    "    pdf = FPDF()                     # A4 portrait, unit=mm\n"
    "    pdf.set_auto_page_break(auto=True, margin=15)\n"
    "    pdf.add_page()\n"
    "    pdf.set_font('Helvetica', 'B', 16)\n"
    "    pdf.cell(0, 10, 'Title', new_x='LMARGIN', new_y='NEXT')\n"
    "    pdf.set_font('Helvetica', size=11)\n"
    "    pdf.multi_cell(0, 6, 'Body paragraph text...')   # wraps long text\n"
    "    pdf.output('/out/report.pdf')\n"
    "  Guidelines: one logical section per heading (bold cell) + multi_cell body; use "
    "pdf.ln(4) between sections; tables via pdf.cell(w, h, txt, border=1) per column with "
    "new_y='LAST' until the row's final cell. Helvetica/Times/Courier need no font files. "
    "Keep text ASCII-safe or pass it through .encode('latin-1','replace').decode('latin-1') — "
    "the built-in fonts are not unicode.\n"
    "If the result is worth keeping beyond the artifact, save a store_fact too. If the tool "
    "returns an `error`, the sandbox itself is unavailable — say so honestly and fall back to "
    "reasoning it through."
)

sandbox_capability = Capability(id="PythonSandbox", instructions=INSTRUCTIONS)

# Extensions whose mimetypes.guess_type answer is missing/platform-dependent (Windows reads the
# registry), pinned so artifact rendering in the UI is deterministic.
_MIME_BY_EXT = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".html": "text/html",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".py": "text/x-python",
}
# Non-text/* mimes that are still text on the wire.
_TEXT_MIMES = {"application/json", "image/svg+xml"}


def _mime_for(filename: str) -> str:
    dot = filename.rfind(".")
    ext = filename[dot:].lower() if dot >= 0 else ""
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


async def _persist_files(
    ctx: RunContext[GraphDependencies], files: list[SandboxFile]
) -> tuple[list[DocumentInfo], list[str]]:
    """Save each /out file as a Document (text as-is, binary base64). Best-effort per file."""
    deps = ctx.deps
    documents: list[DocumentInfo] = []
    notes: list[str] = []
    for f in files:
        mime = _mime_for(f.name)
        if mime.startswith("text/") or mime in _TEXT_MIMES:
            content, encoding = f.data.decode("utf-8", errors="replace"), "text"
        else:
            content, encoding = base64.b64encode(f.data).decode("ascii"), "base64"
        try:
            document_id = await repo.create_document(
                deps.db,
                deps.user_id,
                deps.conversation_id,
                title=f.name,
                content=content,
                mime_type=mime,
                encoding=encoding,
            )
        except Exception:  # noqa: BLE001 — a DB hiccup must not void the code's stdout/stderr.
            logger.warning("failed to persist sandbox file %r as a document", f.name, exc_info=True)
            notes.append(f"Output file {f.name!r} could not be saved as a document.")
            continue
        documents.append(
            DocumentInfo(
                document_id=document_id,
                conversation_id=deps.conversation_id,
                title=f.name,
                mime_type=mime,
                encoding=encoding,
            )
        )
    return documents, notes


@sandbox_capability.tool
async def run_python(ctx: RunContext[GraphDependencies], args: RunPythonArgs) -> PythonRunResult:
    """Execute a self-contained Python program in an isolated container and return its output.

    Stateless between calls; stdlib + fpdf (fpdf2); no network. print() what you need to see;
    write file artifacts (e.g. PDFs) to /out — they are saved as documents automatically.
    """
    sandbox = ctx.deps.sandbox or PythonSandbox()
    try:
        result = await sandbox.run(
            args.code,
            timeout_seconds=float(args.timeout_seconds) if args.timeout_seconds else None,
        )
        documents, persist_notes = await _persist_files(ctx, result.files)
    except Exception as exc:  # noqa: BLE001 — never abort the run on a sandbox failure.
        logger.warning("run_python failed: %s", exc, exc_info=True)
        return PythonRunResult(error=f"Sandbox execution failed: {exc}")
    return PythonRunResult(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        truncated=result.truncated,
        documents=documents,
        notes=[*result.notes, *persist_notes],
        error=result.error,
    )


def build_sandbox() -> list[Capability]:
    """Return the Python-sandbox capability to add to ``Agent(capabilities=...)``.

    The sandbox is supplied per-run through ``GraphDependencies.sandbox`` (or built from env on
    demand), so nothing needs to be wired in here.
    """
    return [sandbox_capability]

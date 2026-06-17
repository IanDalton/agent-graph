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
import os
import shutil
import tempfile

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
    "  - SKILLS: when a skill is enabled and ships files, they are mounted READ-ONLY at "
    "$SKILLS_DIR/<skill-name>/ (call load_skill first to see what a skill provides and how to use "
    "it). Read/import them from there; there is no network, so a skill can't install packages.\n"
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


def _make_world_readable(root: str) -> None:
    """Make ``root`` and everything under it readable by the sandbox's non-root user (Linux).

    The container runs as 'nobody' and the /skills mount is read-only, so files need o+r and dirs
    o+rx for it to traverse them. A no-op on Windows/Docker Desktop mounts.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        try:
            os.chmod(dirpath, 0o755)
        except OSError:
            pass
        for fn in filenames:
            try:
                os.chmod(os.path.join(dirpath, fn), 0o644)
            except OSError:
                pass


async def _materialize_skills(
    ctx: RunContext[GraphDependencies], names: list[str]
) -> tuple[str | None, list[str]]:
    """Write the enabled skills' files into a fresh host temp dir for read-only mounting.

    Each skill lands under ``<dir>/<name>/`` (its SKILL.md body as ``SKILL.md`` plus its bundled
    files at their relative paths, base64-decoded as needed). Returns ``(dir | None, notes)``; the
    caller mounts ``dir`` at ``/skills`` and MUST remove it afterwards. Tolerant: any failure
    returns ``(None, notes)`` so the code still runs — just without the skill files. Uses the
    default ``TMPDIR`` (the same shared bind ``/out`` uses) so the path is reachable by the daemon.
    """
    deps = ctx.deps
    notes: list[str] = []
    skills_dir = tempfile.mkdtemp(prefix="agent-skills-")
    wrote_any = False
    try:
        for name in names:
            skill = await repo.get_skill(deps.db, deps.user_id, name)
            if not skill:
                continue
            base = os.path.join(skills_dir, name)
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "SKILL.md"), "w", encoding="utf-8") as fh:
                fh.write(str(skill.get("body") or ""))
            for relpath, info in (skill.get("files") or {}).items():
                dest = os.path.normpath(os.path.join(base, relpath))
                # Defense-in-depth: never let a crafted relpath escape the skill's dir.
                if os.path.commonpath([base, dest]) != base:
                    notes.append(f"Skipped skill file {relpath!r} (path outside skill dir).")
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                content = info.get("content", "") if isinstance(info, dict) else ""
                if isinstance(info, dict) and info.get("encoding") == "base64":
                    with open(dest, "wb") as fh:
                        fh.write(base64.b64decode(content))
                else:
                    with open(dest, "w", encoding="utf-8") as fh:
                        fh.write(content)
            wrote_any = True
        if not wrote_any:
            shutil.rmtree(skills_dir, ignore_errors=True)
            return None, notes
        _make_world_readable(skills_dir)
        return skills_dir, notes
    except Exception:  # noqa: BLE001 — skill files are best-effort; run without them on failure.
        logger.warning("failed to materialize skill files for the sandbox", exc_info=True)
        shutil.rmtree(skills_dir, ignore_errors=True)
        notes.append("Could not prepare skill files for the sandbox; ran without them.")
        return None, notes


@sandbox_capability.tool
async def run_python(ctx: RunContext[GraphDependencies], args: RunPythonArgs) -> PythonRunResult:
    """Execute a self-contained Python program in an isolated container and return its output.

    Stateless between calls; stdlib + fpdf (fpdf2); no network. print() what you need to see;
    write file artifacts (e.g. PDFs) to /out — they are saved as documents automatically. Files
    shipped by enabled skills are mounted read-only under $SKILLS_DIR/<name>/.
    """
    sandbox = ctx.deps.sandbox or PythonSandbox()
    timeout = float(args.timeout_seconds) if args.timeout_seconds else None
    skills_dir: str | None = None
    skill_notes: list[str] = []
    if ctx.deps.enabled_skills:
        skills_dir, skill_notes = await _materialize_skills(ctx, ctx.deps.enabled_skills)
    try:
        # Pass skills_dir only when set, so a sandbox without that parameter (test fakes) still works.
        run_kwargs: dict[str, object] = {"timeout_seconds": timeout}
        if skills_dir:
            run_kwargs["skills_dir"] = skills_dir
        result = await sandbox.run(args.code, **run_kwargs)
        documents, persist_notes = await _persist_files(ctx, result.files)
    except Exception as exc:  # noqa: BLE001 — never abort the run on a sandbox failure.
        logger.warning("run_python failed: %s", exc, exc_info=True)
        return PythonRunResult(error=f"Sandbox execution failed: {exc}", notes=skill_notes)
    finally:
        if skills_dir:
            shutil.rmtree(skills_dir, ignore_errors=True)
    return PythonRunResult(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        truncated=result.truncated,
        documents=documents,
        notes=[*result.notes, *persist_notes, *skill_notes],
        error=result.error,
    )


def build_sandbox() -> list[Capability]:
    """Return the Python-sandbox capability to add to ``Agent(capabilities=...)``.

    The sandbox is supplied per-run through ``GraphDependencies.sandbox`` (or built from env on
    demand), so nothing needs to be wired in here.
    """
    return [sandbox_capability]

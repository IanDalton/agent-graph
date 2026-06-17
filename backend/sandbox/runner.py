"""Containerized Python execution via ephemeral Docker containers.

:class:`PythonSandbox` runs a snippet of Python inside a fresh, locked-down container per call
(``docker run --rm``): no network, capped memory/CPU/pids, read-only root filesystem with a small
writable ``/tmp``, non-root user, and a hard wall-clock timeout. The code is piped over stdin
(``python -I -``), so nothing is interpolated into a shell and there is no command-length limit.

**File output:** a host temp directory is mounted at ``/out`` (also exported as ``$OUTPUT_DIR``).
Files the program writes there — PDFs, images, CSVs, HTML — are collected after the run and
returned as :class:`SandboxFile` blobs (count/size-capped), so the caller can persist them as
documents. This is how the agent produces real artifacts (e.g. fpdf2 PDFs) from sandboxed code.

The sandbox talks to the host's Docker CLI (the same daemon that runs the compose services), so
it needs no extra Python dependency. The default image is the project's own ``agent-sandbox``
(python:3.12-slim + fpdf2 for PDF generation — build it with ``docker compose build sandbox``);
when that image hasn't been built, the run transparently retries on plain ``python:3.12-slim``
(stdlib only). Configuration comes from the environment with safe defaults:

- ``SANDBOX_IMAGE``            (default ``agent-sandbox``)
- ``SANDBOX_TIMEOUT_SECONDS``  (default ``30``)
- ``SANDBOX_MEMORY``           (default ``256m``)
- ``SANDBOX_CPUS``             (default ``1``)
- ``SANDBOX_NETWORK``          (default ``none`` — no internet; the agent has web_search for that)

Failure contract: :meth:`PythonSandbox.run` never raises for expected problems (Docker missing,
non-zero exit, timeout) — it returns a :class:`SandboxResult` whose fields describe what happened.
That keeps the agent-facing tool (``run_python``) trivially tolerant.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("agent_graph.sandbox")

# The project image (see docker/sandbox/Dockerfile): python:3.12-slim + fpdf2 for PDFs.
DEFAULT_IMAGE = "agent-sandbox"
# Stdlib-only fallback when the project image hasn't been built yet.
FALLBACK_IMAGE = "python:3.12-slim"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY = "256m"
DEFAULT_CPUS = "1"
DEFAULT_NETWORK = "none"
# Per-stream cap on captured output, so a print loop can't flood the model's context.
MAX_OUTPUT_BYTES = 64_000
# Caps on collected /out files: enough for a report + a few assets, small enough to store.
MAX_FILES = 8
MAX_FILE_BYTES = 5_000_000


@dataclass
class SandboxFile:
    """One file the program wrote to ``/out``."""

    name: str
    data: bytes


@dataclass
class SandboxResult:
    """Outcome of one sandboxed execution. ``error`` is set for infrastructure problems
    (Docker unavailable, timeout); a non-zero ``exit_code`` with a traceback in ``stderr``
    is a normal result the model can read and fix. ``files`` holds whatever the program
    wrote to ``/out`` (post-run, count/size-capped — drops are noted in ``notes``)."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    truncated: bool = False
    error: str | None = None
    files: list[SandboxFile] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _cap(data: bytes) -> tuple[str, bool]:
    """Decode a captured stream, truncating it to MAX_OUTPUT_BYTES."""
    truncated = len(data) > MAX_OUTPUT_BYTES
    if truncated:
        data = data[:MAX_OUTPUT_BYTES]
    return data.decode("utf-8", errors="replace"), truncated


def _collect_files(out_dir: str) -> tuple[list[SandboxFile], list[str]]:
    """Read the regular files the program left in the mounted output dir (capped, flat).

    Subdirectories are ignored (the instructions say to write flat files to /out); oversized
    files and overflow beyond MAX_FILES are skipped with a human-readable note so the model
    learns why an artifact didn't come through.
    """
    files: list[SandboxFile] = []
    notes: list[str] = []
    try:
        entries = sorted(p for p in Path(out_dir).iterdir() if p.is_file())
    except OSError:
        logger.warning("could not list sandbox output dir %s", out_dir, exc_info=True)
        return files, notes
    for path in entries:
        if len(files) >= MAX_FILES:
            notes.append(f"Only the first {MAX_FILES} output files were kept; the rest were dropped.")
            break
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                notes.append(f"Output file {path.name!r} exceeded {MAX_FILE_BYTES // 1_000_000}MB and was skipped.")
                continue
            files.append(SandboxFile(name=path.name, data=path.read_bytes()))
        except OSError:
            logger.warning("could not read sandbox output file %s", path, exc_info=True)
            notes.append(f"Output file {path.name!r} could not be read and was skipped.")
    return files, notes


class PythonSandbox:
    """Runs Python snippets in ephemeral, resource-limited Docker containers."""

    def __init__(
        self,
        image: str | None = None,
        timeout_seconds: float | None = None,
        memory: str | None = None,
        cpus: str | None = None,
        network: str | None = None,
    ) -> None:
        self.image = image or os.getenv("SANDBOX_IMAGE", DEFAULT_IMAGE)
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("SANDBOX_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
        self.memory = memory or os.getenv("SANDBOX_MEMORY", DEFAULT_MEMORY)
        self.cpus = cpus or os.getenv("SANDBOX_CPUS", DEFAULT_CPUS)
        self.network = network or os.getenv("SANDBOX_NETWORK", DEFAULT_NETWORK)

    def _docker_args(
        self, name: str, image: str, out_dir: str, skills_dir: str | None = None
    ) -> list[str]:
        """The full ``docker run`` argv for one execution (code arrives on stdin).

        When ``skills_dir`` is given, the enabled skills' files are bind-mounted READ-ONLY at
        ``/skills`` (exported as ``$SKILLS_DIR``). The mount is ``:ro`` and adds no capability, so
        every other hardening flag (no network, dropped caps, read-only root, non-root user,
        resource caps) is unchanged.
        """
        args = [
            "docker", "run", "--rm", "-i",
            "--name", name,
            "--network", self.network,
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", "128",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            # Immutable container; /tmp (scratch) and /out (artifacts) are the only writable spots.
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "--volume", f"{out_dir}:/out",
        ]
        if skills_dir is not None:
            args += ["--volume", f"{skills_dir}:/skills:ro", "--env", "SKILLS_DIR=/skills"]
        args += [
            "--workdir", "/tmp",
            "--user", "65534:65534",  # nobody
            "--env", "HOME=/tmp",
            "--env", "OUTPUT_DIR=/out",
            image,
            "python", "-I", "-",  # isolated mode, program from stdin
        ]
        return args

    async def run(
        self,
        code: str,
        timeout_seconds: float | None = None,
        skills_dir: str | None = None,
    ) -> SandboxResult:
        """Execute ``code`` in a fresh container and return its captured output + /out files.

        Never raises for expected failures — see the module docstring's failure contract. When
        the project image (``agent-sandbox``) isn't built yet, retries once on the stdlib-only
        base image so plain Python still works. ``skills_dir`` (when given) is mounted read-only at
        ``/skills`` so enabled skills' bundled scripts/assets are available to the code.
        """
        timeout = min(timeout_seconds or self.timeout_seconds, 120.0)
        result = await self._run_once(code, self.image, timeout, skills_dir)
        if (
            self.image == DEFAULT_IMAGE
            and result.exit_code not in (0, None)
            and "Unable to find image" in result.stderr
        ):
            logger.warning(
                "sandbox image %r not built (docker compose build sandbox); "
                "falling back to %r (stdlib only, no PDF libs)",
                self.image, FALLBACK_IMAGE,
            )
            result = await self._run_once(code, FALLBACK_IMAGE, timeout, skills_dir)
            result.notes.append(
                f"Ran on the fallback image {FALLBACK_IMAGE} (stdlib only): the {DEFAULT_IMAGE} "
                "image with PDF support is not built. Third-party imports like fpdf will fail."
            )
        return result

    async def _run_once(
        self, code: str, image: str, timeout: float, skills_dir: str | None = None
    ) -> SandboxResult:
        name = f"agent-sandbox-{uuid.uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory(prefix="agent-sandbox-out-") as out_dir:
            # The container runs as 'nobody'; on Linux hosts the bind-mounted dir must be
            # world-writable for it to create files (no-op on Windows/Docker Desktop mounts).
            try:
                os.chmod(out_dir, 0o777)
            except OSError:
                pass
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self._docker_args(name, image, out_dir, skills_dir),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                return SandboxResult(
                    error=(
                        "Docker is not available on this host, so sandboxed Python cannot run. "
                        "Tell the user the code-execution sandbox is offline."
                    )
                )
            except OSError as exc:
                return SandboxResult(error=f"Could not start the sandbox container: {exc}")

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=code.encode("utf-8")), timeout=timeout
                )
            except asyncio.TimeoutError:
                # Killing the docker CLI does not stop the container (the daemon owns it),
                # so remove it by name as well.
                proc.kill()
                await self._force_remove(name)
                return SandboxResult(
                    timed_out=True,
                    error=(
                        f"Execution exceeded the {timeout:.0f}s time limit and was stopped. "
                        "Make the code finish faster (smaller input, fewer iterations)."
                    ),
                )

            out_text, out_trunc = _cap(stdout)
            err_text, err_trunc = _cap(stderr)
            files, notes = _collect_files(out_dir)
            return SandboxResult(
                stdout=out_text,
                stderr=err_text,
                exit_code=proc.returncode,
                truncated=out_trunc or err_trunc,
                files=files,
                notes=notes,
            )

    async def _force_remove(self, name: str) -> None:
        """Best-effort ``docker rm -f`` of a timed-out container."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
        except Exception:  # noqa: BLE001 — cleanup is best-effort; --rm reaps it eventually.
            logger.warning("failed to force-remove sandbox container %s", name, exc_info=True)

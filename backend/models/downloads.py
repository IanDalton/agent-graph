"""In-process manager for GGUF downloads that survive client disconnects.

A download runs as a **detached** background task (not tied to the SSE request that started it), so
refreshing the page or closing the Model Manager doesn't cancel it (the old endpoint cancelled the
task in its ``finally``). Each download fans out progress frames to any number of subscribed SSE
streams; a client that reconnects after a refresh **re-attaches** to the still-running task instead of
starting a duplicate. ``active_snapshot`` backs ``GET /api/models/downloads`` so the UI can repopulate
its progress bars.

Dependency-light + unit-testable: the actual HuggingFace download is injected as a ``downloader``
coroutine (``downloader(progress) -> done_payload``), so this module imports neither httpx nor the HF
client — mirroring how the rest of ``backend.models`` keeps the network at the edges.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger("agent_graph.downloads")

# Keep a finished entry around briefly so a UI that reconnects right after completion still sees the
# terminal frame; then it's reaped (the model is already in the library by then).
_GRACE_SECONDS = 30.0

Frame = dict[str, Any] | None
Downloader = Callable[[Callable[[int, int], Awaitable[None]]], Awaitable[dict[str, Any]]]


@dataclass
class _Download:
    key: str
    repo_id: str
    file_path: str
    quant: str = ""
    revision: str = "main"
    filename: str = ""
    downloaded: int = 0
    total: int = 0
    status: str = "downloading"  # "downloading" | "done" | "error"
    message: str = ""
    label: str = ""
    path: str = ""
    # Smoothed transfer rate (bytes/sec) + ETA (seconds) for the UI's speed/time-left readout.
    speed_bps: float = 0.0
    eta_seconds: float | None = None
    # Internal sampling state for the EMA speed estimate (not serialized).
    last_t: float = 0.0
    last_bytes: int = 0
    task: "asyncio.Task[None] | None" = None
    subscribers: set["asyncio.Queue[Frame]"] = field(default_factory=set)

    def observe(self, downloaded: int, total: int) -> None:
        """Record a progress sample and update the smoothed speed + ETA."""
        now = time.monotonic()
        if self.last_t and now > self.last_t:
            inst = (downloaded - self.last_bytes) / (now - self.last_t)
            if inst >= 0:  # ignore a backwards jump (e.g. a resume re-seek)
                self.speed_bps = inst if self.speed_bps <= 0 else 0.4 * inst + 0.6 * self.speed_bps
        self.last_t = now
        self.last_bytes = downloaded
        self.downloaded, self.total = downloaded, total
        self.eta_seconds = (
            max(0, total - downloaded) / self.speed_bps if self.speed_bps > 0 and total else None
        )

    def progress_frame(self) -> dict[str, Any]:
        return {
            "type": "progress",
            "downloaded": self.downloaded,
            "total": self.total,
            "speed_bps": self.speed_bps,
            "eta_seconds": self.eta_seconds,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "quant": self.quant,
            "filename": self.filename,
            "downloaded": self.downloaded,
            "total": self.total,
            "status": self.status,
            "message": self.message,
            "speed_bps": self.speed_bps,
            "eta_seconds": self.eta_seconds,
        }

    def terminal_frames(self) -> list[Frame]:
        """The frame(s) a late subscriber to a finished download should receive immediately."""
        if self.status == "done":
            return [
                {"type": "done", "filename": self.filename, "path": self.path,
                 "size_bytes": self.total, "label": self.label},
                None,
            ]
        return [{"type": "error", "message": self.message or "download failed"}, None]


_active: dict[str, _Download] = {}


def key_for(repo_id: str, file_path: str) -> str:
    return f"{repo_id}/{file_path}"


def active_snapshot() -> list[dict[str, Any]]:
    """The current download set (in-progress + recently-finished) for the repopulate endpoint."""
    return [d.snapshot() for d in _active.values()]


def _fanout(dl: _Download, item: Frame) -> None:
    for q in list(dl.subscribers):
        q.put_nowait(item)


async def _run(dl: _Download, downloader: Downloader) -> None:
    async def progress(downloaded: int, total: int) -> None:
        dl.observe(downloaded, total)
        _fanout(dl, dl.progress_frame())

    try:
        payload = await downloader(progress)
        dl.status = "done"
        dl.filename = payload.get("filename", dl.filename)
        dl.path = payload.get("path", "")
        dl.label = payload.get("label", "")
        dl.total = int(payload.get("size_bytes", dl.total) or 0)
        dl.downloaded = dl.total
        _fanout(dl, {"type": "done", **payload})
    except asyncio.CancelledError:
        dl.status = "error"
        dl.message = "cancelled"
        _fanout(dl, {"type": "error", "message": "cancelled"})
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced as an error frame, never raised to a caller.
        logger.warning("model download failed: %s", dl.key, exc_info=True)
        dl.status = "error"
        dl.message = f"{type(exc).__name__}: {exc}"
        _fanout(dl, {"type": "error", "message": dl.message})
    finally:
        _fanout(dl, None)  # sentinel: end every attached stream
        asyncio.ensure_future(_expire(dl.key))


async def _expire(key: str, delay: float = _GRACE_SECONDS) -> None:
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:  # pragma: no cover - shutdown
        return
    _active.pop(key, None)


def start(key: str, meta: dict[str, Any], downloader: Downloader) -> _Download:
    """Start a download for ``key`` — or return the one already in flight (no duplicates)."""
    existing = _active.get(key)
    if existing is not None and existing.status == "downloading":
        return existing
    dl = _Download(key=key, **meta)
    _active[key] = dl
    dl.task = asyncio.create_task(_run(dl, downloader))
    return dl


def subscribe(dl: _Download) -> "asyncio.Queue[Frame]":
    """A queue that receives ``dl``'s frames, seeded with the current state so a late joiner's bar
    shows immediately. A finished download emits its terminal frame(s) right away."""
    q: "asyncio.Queue[Frame]" = asyncio.Queue()
    if dl.status == "downloading":
        dl.subscribers.add(q)
        if dl.total:
            q.put_nowait(dl.progress_frame())
    else:
        for frame in dl.terminal_frames():
            q.put_nowait(frame)
    return q


def unsubscribe(dl: _Download, q: "asyncio.Queue[Frame]") -> None:
    dl.subscribers.discard(q)


__all__ = ["key_for", "active_snapshot", "start", "subscribe", "unsubscribe"]

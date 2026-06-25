"""Tests for the detached download manager (no network — the downloader is injected)."""

from __future__ import annotations

import asyncio

from backend.models import downloads as d


def _meta(key: str) -> dict:
    repo, _, path = key.partition("/")
    return {"repo_id": repo, "file_path": path, "quant": "Q4", "revision": "main", "filename": path}


def _drain(q) -> list:
    frames = []
    while True:
        item = q.get_nowait() if not q.empty() else None
        if item is None:
            break
        frames.append(item)
    return frames


def test_completion_fans_out_progress_then_done() -> None:
    async def downloader(progress):
        await progress(50, 100)
        await progress(100, 100)
        return {"filename": "m.gguf", "path": "/models/m.gguf", "size_bytes": 100, "label": "local/m"}

    async def main():
        d._active.clear()
        dl = d.start("r/m.gguf", _meta("r/m.gguf"), downloader)
        q = d.subscribe(dl)
        types = []
        while True:
            item = await q.get()
            if item is None:
                break
            types.append(item["type"])
        return types, dl.status

    types, status = asyncio.run(main())
    assert types == ["progress", "progress", "done"]
    assert status == "done"


def test_start_dedups_in_flight() -> None:
    started = 0

    async def downloader(progress):
        nonlocal started
        started += 1
        await asyncio.sleep(0)  # stay "downloading" long enough for the second start to dedup
        await progress(100, 100)
        return {"filename": "m.gguf", "path": "/m", "size_bytes": 100, "label": "local/m"}

    async def main():
        d._active.clear()
        dl1 = d.start("r/x.gguf", _meta("r/x.gguf"), downloader)
        dl2 = d.start("r/x.gguf", _meta("r/x.gguf"), downloader)  # same key → same object, one task
        same = dl1 is dl2
        await dl1.task
        return same, started

    same, started = asyncio.run(main())
    assert same is True
    assert started == 1


def test_subscribe_to_finished_gets_terminal_frames() -> None:
    async def downloader(progress):
        return {"filename": "m.gguf", "path": "/m", "size_bytes": 7, "label": "local/m"}

    async def main():
        d._active.clear()
        dl = d.start("r/done.gguf", _meta("r/done.gguf"), downloader)
        await dl.task  # let it finish
        late = d.subscribe(dl)  # a client that connects AFTER completion
        frames = []
        while True:
            item = await late.get()
            if item is None:
                break
            frames.append(item)
        return frames

    frames = asyncio.run(main())
    assert frames and frames[0]["type"] == "done"


def test_observe_computes_speed_and_eta() -> None:
    dl = d._Download(key="r/m.gguf", repo_id="r", file_path="m.gguf")
    dl.observe(0, 1000)  # first sample only seeds the baseline — no rate yet
    assert dl.speed_bps == 0
    assert dl.eta_seconds is None
    dl.last_t -= 1.0  # pretend the previous sample was ~1s ago
    dl.observe(100, 1000)  # +100 bytes over ~1s → ~100 B/s, ~9s left
    assert dl.speed_bps > 0
    assert dl.eta_seconds is not None and dl.eta_seconds > 0
    # The frame + snapshot both carry the readout fields the UI displays.
    assert dl.progress_frame()["speed_bps"] == dl.speed_bps
    snap = dl.snapshot()
    assert snap["speed_bps"] == dl.speed_bps and snap["eta_seconds"] == dl.eta_seconds


def test_active_snapshot_reports_status() -> None:
    async def downloader(progress):
        await progress(10, 20)
        return {"filename": "m.gguf", "path": "/m", "size_bytes": 20, "label": "local/m"}

    async def main():
        d._active.clear()
        dl = d.start("r/snap.gguf", _meta("r/snap.gguf"), downloader)
        await dl.task
        return d.active_snapshot()

    snap = asyncio.run(main())
    assert len(snap) == 1
    assert snap[0]["repo_id"] == "r"
    assert snap[0]["status"] == "done"

"""Tests for hardware detection + the editable profile (no real GPU/subprocess needed)."""

from __future__ import annotations

import asyncio

from backend.models import hardware as hw


def test_detect_nvidia_absent(monkeypatch) -> None:
    async def boom(*_a, **_k):
        raise FileNotFoundError()

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    assert asyncio.run(hw.detect_nvidia()) == []


def test_detect_nvidia_parses(monkeypatch) -> None:
    class FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"NVIDIA GeForce RTX 4090, 24564\nNVIDIA RTX A6000, 49140\n", b"")

    async def fake_exec(*_a, **_k):
        return FakeProc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    gpus = asyncio.run(hw.detect_nvidia())
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert gpus[0].vram_mb == 24564
    assert gpus[1].vram_mb == 49140


def test_detect_nvidia_nonzero_exit(monkeypatch) -> None:
    class FakeProc:
        returncode = 1

        async def communicate(self):
            return (b"", b"no devices")

    async def fake_exec(*_a, **_k):
        return FakeProc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    assert asyncio.run(hw.detect_nvidia()) == []


def test_detect_system_smoke() -> None:
    ram_mb, threads = hw.detect_system()
    assert ram_mb >= 0
    assert threads >= 0


def test_profile_roundtrip(tmp_path) -> None:
    p = hw.HardwareProfile(
        gpus=[hw.GpuInfo("RTX 4090", 24564)],
        system_ram_mb=64000,
        cpu_threads=16,
        source="manual",
        updated_at="t",
    )
    path = tmp_path / "hw.json"
    hw.save_profile(path, p)
    loaded = hw.load_profile(path)
    assert loaded is not None
    assert loaded.vram_total_mb == 24564
    assert loaded.gpu_count == 1
    assert loaded.source == "manual"
    assert loaded.gpus[0].name == "RTX 4090"


def test_load_missing_profile(tmp_path) -> None:
    assert hw.load_profile(tmp_path / "missing.json") is None


def test_to_dict_includes_derived() -> None:
    p = hw.HardwareProfile(gpus=[hw.GpuInfo("g", 8000), hw.GpuInfo("g", 8000)], source="manual")
    d = p.to_dict()
    assert d["gpu_count"] == 2
    assert d["vram_total_mb"] == 16000

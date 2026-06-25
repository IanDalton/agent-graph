"""Hardware profile + best-effort NVIDIA detection for model recommendations.

The llama-server is **external** and may live on another machine, so the backend container generally
can't see the real GPU. The hardware *profile* the user enters is therefore the **source of truth**
for recommendations; ``nvidia-smi`` auto-detection is a best-effort convenience that only works when
a GPU is visible to this process (co-located / GPU-passthrough). Modeled on
:mod:`backend.sandbox.runner`'s subprocess + graceful-failure pattern. Stdlib only — no ``psutil``.

The profile is persisted as a small JSON file on the shared models volume (next to the GGUFs), so a
fresh backend container reloads it and so a future separate-machine setup can carry it alongside the
models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("agent_graph.hardware")

_NVIDIA_SMI_TIMEOUT = 5.0


@dataclass
class GpuInfo:
    name: str
    vram_mb: int


# A small catalog of common NVIDIA cards → VRAM (MB), so the UI can auto-fill VRAM the moment a known
# GPU name is typed (manual entry is otherwise easy to leave blank — the cause of the historical
# ``vram_mb: 0`` profiles). Keys are matched case-insensitively as substrings of the entered name, so
# "NVIDIA GeForce RTX 3090" matches "rtx 3090". Not exhaustive — auto-detect is the primary fill.
KNOWN_GPUS: dict[str, int] = {
    "rtx 5090": 32768,
    "rtx 4090": 24564,
    "rtx 3090 ti": 24576,
    "rtx 3090": 24576,
    "rtx 4080": 16376,
    "rtx 3080 ti": 12288,
    "rtx 3080": 10240,
    "rtx 4070 ti": 12288,
    "rtx 4070": 12288,
    "rtx 3070": 8192,
    "rtx 3060": 12288,
    "rtx a6000": 49140,
    "rtx a5000": 24564,
    "rtx a4000": 16376,
    "a100": 40960,
    "h100": 81920,
    "l40s": 49152,
    "l4": 24576,
    "t4": 16384,
    "v100": 16384,
}


def vram_for_name(name: str) -> int | None:
    """VRAM (MB) for a known GPU name, or ``None``. Substring match, longest key first (so
    "rtx 3090 ti" wins over "rtx 3090")."""
    lowered = (name or "").lower()
    for key in sorted(KNOWN_GPUS, key=len, reverse=True):
        if key in lowered:
            return KNOWN_GPUS[key]
    return None


@dataclass
class HardwareProfile:
    """A description of the machine that will run llama-server (the recommendation target)."""

    gpus: list[GpuInfo] = field(default_factory=list)
    system_ram_mb: int = 0
    cpu_threads: int = 0
    source: str = "default"  # "manual" | "auto" | "default"
    updated_at: str = ""

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def vram_total_mb(self) -> int:
        return sum(g.vram_mb for g in self.gpus)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["gpu_count"] = self.gpu_count
        data["vram_total_mb"] = self.vram_total_mb
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareProfile":
        gpus = [
            GpuInfo(name=str(g.get("name", "")), vram_mb=int(g.get("vram_mb", 0) or 0))
            for g in (data.get("gpus") or [])
        ]
        return cls(
            gpus=gpus,
            system_ram_mb=int(data.get("system_ram_mb", 0) or 0),
            cpu_threads=int(data.get("cpu_threads", 0) or 0),
            source=str(data.get("source", "manual")),
            updated_at=str(data.get("updated_at", "")),
        )


def iso_now() -> str:
    """Current UTC time as an ISO-8601 string (used to stamp a saved profile's ``updated_at``)."""
    return datetime.now(timezone.utc).isoformat()


async def detect_nvidia() -> list[GpuInfo]:
    """Best-effort list of NVIDIA GPUs via ``nvidia-smi``; ``[]`` when it's absent/fails.

    Mirrors the sandbox's tolerant subprocess handling: a missing binary (``FileNotFoundError``),
    timeout, or non-zero exit all degrade to ``[]`` rather than raising — the manual profile is the
    fallback.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError):
        return []
    try:
        stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=_NVIDIA_SMI_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return []
    if proc.returncode != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, mem = line.rpartition(",")
        try:
            gpus.append(GpuInfo(name=name.strip(), vram_mb=int(float(mem.strip()))))
        except ValueError:
            continue
    return gpus


def detect_system() -> tuple[int, int]:
    """Return ``(system_ram_mb, cpu_threads)`` from ``/proc/meminfo`` + ``os.cpu_count()``.

    Tolerant: a non-Linux host (no ``/proc/meminfo``) yields ``ram=0``; the user fills it in.
    """
    ram_mb = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram_mb = int(line.split()[1]) // 1024  # kB → MB
                    break
    except OSError:
        ram_mb = 0
    return ram_mb, os.cpu_count() or 0


async def auto_profile() -> HardwareProfile:
    """Detect the *current* host's hardware (GPUs + RAM/threads). ``source`` is ``auto`` if a GPU was
    found, else ``default`` (no GPU visible — likely the containerized/remote case)."""
    gpus = await detect_nvidia()
    ram_mb, threads = detect_system()
    return HardwareProfile(
        gpus=gpus,
        system_ram_mb=ram_mb,
        cpu_threads=threads,
        source="auto" if gpus else "default",
        updated_at=iso_now(),
    )


def default_profile() -> HardwareProfile:
    """A conservative no-GPU profile (RAM/threads from this host) used when nothing is saved."""
    ram_mb, threads = detect_system()
    return HardwareProfile(
        gpus=[], system_ram_mb=ram_mb, cpu_threads=threads, source="default", updated_at=iso_now()
    )


def load_profile(path: str | Path) -> HardwareProfile | None:
    """Load a saved profile from JSON, or ``None`` if it's absent/unreadable."""
    path = Path(path)
    try:
        if not path.exists():
            return None
        return HardwareProfile.from_dict(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError):
        logger.warning("could not read hardware profile %s", path, exc_info=True)
        return None


def save_profile(path: str | Path, profile: HardwareProfile) -> None:
    """Persist a profile to JSON (atomic write), creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(profile.to_dict(), indent=2), "utf-8")
    os.replace(tmp, path)


__all__ = [
    "GpuInfo",
    "HardwareProfile",
    "KNOWN_GPUS",
    "vram_for_name",
    "detect_nvidia",
    "detect_system",
    "auto_profile",
    "default_profile",
    "load_profile",
    "save_profile",
]

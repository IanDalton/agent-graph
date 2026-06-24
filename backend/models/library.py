"""Filesystem registry of downloaded GGUF models (the local model library).

Models are a **machine-level** resource the external llama-server must read off disk, so — unlike the
per-user conversational data in ArcadeDB — they live on the shared ``LLAMACPP_MODELS_DIR`` volume,
with a small JSON manifest beside them holding the metadata the filesystem doesn't (source repo,
revision, download time). :func:`scan_models_dir` reconciles the manifest against the actual ``.gguf``
files on every read, so files copied in out-of-band still appear and entries whose file vanished are
dropped — the manifest can never lie about what's installed.

A model's UI label is ``local/<filename-stem>`` (parallel to the old ``ollama/<name>`` convention);
that label flows through :func:`backend.model_selection.resolve_model` to the llama.cpp provider.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models.huggingface import parse_quant

logger = logging.getLogger("agent_graph.model_library")

# Sharded GGUFs: ``model-00001-of-00003.gguf``. Only the first shard is surfaced as the model.
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

DEFAULT_MODELS_DIR = "/models"
_META_DIRNAME = ".agent-graph"
_MANIFEST_NAME = "manifest.json"
HARDWARE_FILENAME = "hardware.json"
LOCAL_PREFIX = "local/"


def models_dir() -> Path:
    """The directory holding downloaded GGUFs (``LLAMACPP_MODELS_DIR``, default ``/models``)."""
    return Path(os.getenv("LLAMACPP_MODELS_DIR", DEFAULT_MODELS_DIR))


def _meta_dir(base: Path) -> Path:
    return base / _META_DIRNAME


def manifest_path(base: Path | None = None) -> Path:
    return _meta_dir(base or models_dir()) / _MANIFEST_NAME


def hardware_path(base: Path | None = None) -> Path:
    return _meta_dir(base or models_dir()) / HARDWARE_FILENAME


def model_label(filename: str) -> str:
    """The UI/model label for a GGUF file: ``local/<stem>`` (``.gguf`` extension stripped)."""
    stem = filename[:-5] if filename.lower().endswith(".gguf") else filename
    return f"{LOCAL_PREFIX}{stem}"


def load_manifest(base: Path | None = None) -> dict[str, Any]:
    """Load the manifest dict (``{"version":1,"models":[...]}``); empty default if absent/unreadable."""
    path = manifest_path(base)
    try:
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("models"), list):
                return data
    except (OSError, ValueError):
        logger.warning("could not read model manifest %s", path, exc_info=True)
    return {"version": 1, "models": []}


def save_manifest(manifest: dict[str, Any], base: Path | None = None) -> None:
    """Persist the manifest (atomic write), creating the meta dir as needed."""
    path = manifest_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), "utf-8")
    os.replace(tmp, path)


def scan_models_dir(base: Path | None = None) -> list[dict[str, Any]]:
    """List installed models, reconciling on-disk ``.gguf`` files with the manifest metadata.

    Returns ``[{filename, path, size_bytes, quant, repo_id, revision, downloaded_at, label}]`` sorted
    by filename. Skips the per-shard companions of sharded models (only the first shard is listed) and
    the ``.part`` files of in-flight downloads. Tolerant: an unreadable directory yields ``[]``.
    """
    base = base or models_dir()
    by_name = {m.get("filename"): m for m in load_manifest(base).get("models", []) if m.get("filename")}
    out: list[dict[str, Any]] = []
    try:
        entries = sorted(p for p in base.iterdir() if p.is_file())
    except OSError:
        return []
    for path in entries:
        name = path.name
        if not name.lower().endswith(".gguf"):
            continue
        # Sharded models: only surface the first shard; llama-server loads the siblings itself.
        shard = _SHARD_RE.search(name)
        if shard and shard.group(1) != "00001":
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        meta = by_name.get(name, {})
        out.append(
            {
                "filename": name,
                "path": str(path),
                "size_bytes": int(meta.get("size_bytes") or size),
                "quant": meta.get("quant") or parse_quant(name),
                "repo_id": meta.get("repo_id", ""),
                "revision": meta.get("revision", ""),
                "downloaded_at": meta.get("downloaded_at", ""),
                "label": model_label(name),
            }
        )
    return out


def add_to_manifest(entry: dict[str, Any], base: Path | None = None) -> None:
    """Upsert a downloaded-model record (keyed by filename) into the manifest."""
    base = base or models_dir()
    manifest = load_manifest(base)
    entry = dict(entry)
    entry.setdefault("downloaded_at", datetime.now(timezone.utc).isoformat())
    models = [m for m in manifest.get("models", []) if m.get("filename") != entry.get("filename")]
    models.append(entry)
    manifest["models"] = models
    save_manifest(manifest, base)


def remove_model(filename: str, base: Path | None = None) -> bool:
    """Delete a GGUF file and its manifest entry. Returns ``True`` if a file or entry was removed.

    The caller (API layer) is responsible for the path-traversal guard; here ``filename`` is treated
    as a bare name within ``base``.
    """
    base = base or models_dir()
    removed = False
    target = base / filename
    try:
        if target.exists():
            target.unlink()
            removed = True
    except OSError:
        logger.warning("could not delete model file %s", target, exc_info=True)
    manifest = load_manifest(base)
    models = manifest.get("models", [])
    kept = [m for m in models if m.get("filename") != filename]
    if len(kept) != len(models):
        manifest["models"] = kept
        save_manifest(manifest, base)
        removed = True
    return removed


def local_model_labels(base: Path | None = None) -> list[str]:
    """The ``local/<name>`` labels for the model picker (one per installed GGUF)."""
    return [m["label"] for m in scan_models_dir(base)]


__all__ = [
    "models_dir",
    "manifest_path",
    "hardware_path",
    "model_label",
    "load_manifest",
    "save_manifest",
    "scan_models_dir",
    "add_to_manifest",
    "remove_model",
    "local_model_labels",
]

"""Tests for the filesystem model library/manifest (reconcile with on-disk GGUFs)."""

from __future__ import annotations

from backend.models import library as lib


def test_scan_and_manifest(tmp_path) -> None:
    (tmp_path / "a.gguf").write_bytes(b"x" * 10)
    lib.add_to_manifest(
        {"filename": "a.gguf", "size_bytes": 10, "quant": "Q4_K_M", "repo_id": "r/x", "revision": "main"},
        base=tmp_path,
    )
    models = lib.scan_models_dir(base=tmp_path)
    assert len(models) == 1
    assert models[0]["quant"] == "Q4_K_M"
    assert models[0]["repo_id"] == "r/x"
    assert models[0]["label"] == "local/a"


def test_file_without_manifest_appears(tmp_path) -> None:
    (tmp_path / "b-Q8_0.gguf").write_bytes(b"y" * 5)
    models = lib.scan_models_dir(base=tmp_path)
    assert len(models) == 1
    assert models[0]["filename"] == "b-Q8_0.gguf"
    assert models[0]["quant"] == "Q8_0"  # parsed from the filename


def test_manifest_entry_without_file_dropped(tmp_path) -> None:
    lib.add_to_manifest({"filename": "gone.gguf", "size_bytes": 1}, base=tmp_path)
    assert lib.scan_models_dir(base=tmp_path) == []


def test_sharded_only_first(tmp_path) -> None:
    for i in (1, 2, 3):
        (tmp_path / f"m-0000{i}-of-00003.gguf").write_bytes(b"x")
    models = lib.scan_models_dir(base=tmp_path)
    assert len(models) == 1
    assert models[0]["filename"] == "m-00001-of-00003.gguf"


def test_remove(tmp_path) -> None:
    (tmp_path / "a.gguf").write_bytes(b"x")
    lib.add_to_manifest({"filename": "a.gguf", "size_bytes": 1}, base=tmp_path)
    assert lib.remove_model("a.gguf", base=tmp_path) is True
    assert not (tmp_path / "a.gguf").exists()
    assert lib.scan_models_dir(base=tmp_path) == []
    assert lib.remove_model("nope.gguf", base=tmp_path) is False


def test_empty_dir_tolerant(tmp_path) -> None:
    assert lib.scan_models_dir(base=tmp_path) == []
    assert lib.local_model_labels(base=tmp_path) == []


def test_model_label() -> None:
    assert lib.model_label("Qwen3-8B-Q4_K_M.gguf") == "local/Qwen3-8B-Q4_K_M"

"""Smoke tests for the Model Manager API endpoints (services monkeypatched; no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from backend.api import app


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    # Point the model library + hardware profile at an isolated temp dir per test.
    monkeypatch.setenv("LLAMACPP_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://localhost:1/v1")  # unreachable on purpose
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    return TestClient(app)


def test_list_models_empty(client) -> None:
    assert client.get("/api/models").json() == []


def test_config_reports_llamacpp(client) -> None:
    cfg = client.get("/api/config").json()
    assert cfg["provider"] == "llamacpp"
    assert isinstance(cfg["models"], list)
    assert "llamacpp_base_url" in cfg


def test_hardware_put_then_get(client) -> None:
    saved = client.put(
        "/api/hardware",
        json={"gpus": [{"name": "RTX 4090", "vram_mb": 24564}], "system_ram_mb": 64000, "cpu_threads": 16},
    ).json()
    assert saved["source"] == "manual"
    assert saved["vram_total_mb"] == 24564
    assert client.get("/api/hardware").json()["source"] == "manual"


def test_search_monkeypatched(client, monkeypatch) -> None:
    async def fake_search(query: str, **_k: Any):
        return [{"repo_id": "unsloth/Qwen3-8B-GGUF", "downloads": 1, "likes": 0, "last_modified": "", "gated": False}]

    monkeypatch.setattr("backend.api.hf_models.search", fake_search)
    rows = client.get("/api/models/search?query=qwen").json()
    assert rows[0]["repo_id"] == "unsloth/Qwen3-8B-GGUF"


def test_repo_files_have_fit_badges(client, monkeypatch) -> None:
    async def fake_list(repo: str, revision: str = "main"):
        return [{"path": "m.gguf", "filename": "m.gguf", "quant": "Q4_K_M", "size_bytes": int(4.9e9), "shards": 1}]

    async def fake_cfg(repo: str, revision: str = "main"):
        return {
            "num_hidden_layers": 36,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
        }

    monkeypatch.setattr("backend.api.hf_models.list_files", fake_list)
    monkeypatch.setattr("backend.api.hf_models.fetch_config", fake_cfg)
    # A 24 GB profile so the small model fits fully on GPU (deterministic, not host-dependent).
    client.put(
        "/api/hardware",
        json={"gpus": [{"name": "g", "vram_mb": 24564}], "system_ram_mb": 64000, "cpu_threads": 16},
    )
    rows = client.get("/api/models/repo/unsloth/Qwen3-8B-GGUF/files").json()
    assert rows[0]["fit"] == "gpu"
    assert "recommendation" in rows[0]


def test_delete_missing_is_404(client) -> None:
    assert client.delete("/api/models/nope.gguf").status_code == 404


def test_delete_traversal_guard(client) -> None:
    # A filename containing ".." reaches the handler and is rejected (400). (A bare "/api/models/.."
    # is normalized away by the client/router to a non-route 404 before the handler — also safe.)
    assert client.delete("/api/models/evil..gguf").status_code == 400


def test_download_sse_then_listed(client, monkeypatch) -> None:
    class FakeHF:
        async def stream_download(self, repo, file_path, dest_dir, revision="main", progress=None):
            p = Path(dest_dir) / file_path
            p.write_bytes(b"data")
            if progress:
                await progress(2, 4)
                await progress(4, 4)
            return p

        async def aclose(self):
            pass

    monkeypatch.setattr("backend.api.hf_models.HuggingFaceClient", lambda *a, **k: FakeHF())
    resp = client.post(
        "/api/models/download", json={"repo_id": "r/x", "file_path": "m.gguf", "quant": "Q4_K_M"}
    )
    body = resp.text
    assert "progress" in body
    assert "done" in body
    # The downloaded model is now in the library.
    assert any(m["filename"] == "m.gguf" for m in client.get("/api/models").json())


def test_active_downloads_endpoint(client) -> None:
    # No downloads in flight → an empty list (and the route exists). Clear the process-wide registry
    # first since other tests may have left (not-yet-expired) entries in it.
    from backend.models import downloads as _downloads

    _downloads._active.clear()
    assert client.get("/api/models/downloads").json() == []


def test_llamacpp_status_unreachable_is_tolerant(client) -> None:
    status = client.get("/api/llamacpp/status").json()
    assert status["reachable"] is False
    assert status["served_model"] is None


def test_config_exposes_known_gpus(client) -> None:
    cfg = client.get("/api/config").json()
    assert "known_gpus" in cfg
    assert cfg["known_gpus"].get("rtx 3090") == 24576


def test_recommend_includes_model_max_ctx(client) -> None:
    body = {"size_bytes": int(4.9e9), "quant": "Q4_K_M", "filename": "m.gguf"}
    rec = client.post("/api/models/recommend", json=body).json()["recommendation"]
    assert "model_max_ctx" in rec and rec["model_max_ctx"] > 0


def test_recommend_includes_vram_breakdown(client) -> None:
    body = {"size_bytes": int(4.9e9), "quant": "Q4_K_M", "filename": "Qwen3-8B-Q4_K_M.gguf"}
    rec = client.post("/api/models/recommend", json=body).json()["recommendation"]
    for key in ("weights_mb", "kv_cache_mb", "overhead_mb"):
        assert key in rec
    if rec["fit"] in {"gpu", "partial"}:
        total = rec["weights_mb"] + rec["kv_cache_mb"] + rec["overhead_mb"]
        assert abs(total - rec["est_vram_mb"]) <= 1


def test_detect_tolerant_when_no_gpu(client, monkeypatch) -> None:
    # No local nvidia-smi AND no docker → a valid (empty-GPU) profile, never a 500.
    async def boom(*_a, **_k):
        raise FileNotFoundError()

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    resp = client.post("/api/hardware/detect")
    assert resp.status_code == 200
    assert resp.json()["gpus"] == []


def test_load_missing_file_is_404(client) -> None:
    assert client.post("/api/llamacpp/load", json={"filename": "nope.gguf"}).status_code == 404


def test_load_traversal_guard(client) -> None:
    assert client.post("/api/llamacpp/load", json={"filename": "evil..gguf"}).status_code == 400


def test_load_unmanaged_without_docker(client, tmp_path, monkeypatch) -> None:
    # A real library file exists, but docker is unavailable → tolerant {ok:false, unmanaged:true}.
    (tmp_path / "m.gguf").write_bytes(b"x")

    async def boom(*_a, **_k):
        raise FileNotFoundError()

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    resp = client.post("/api/llamacpp/load", json={"filename": "m.gguf"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["unmanaged"] is True

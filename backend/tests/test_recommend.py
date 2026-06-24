"""Tests for the pure llama.cpp recommendation math (no network/DB/Docker)."""

from __future__ import annotations

from backend.models import recommend as rec
from backend.models.hardware import GpuInfo, HardwareProfile

CFG_8B = {
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 40960,
}
CFG_70B = {
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
}


def _gpu(vram_mb: int) -> HardwareProfile:
    return HardwareProfile(
        gpus=[GpuInfo("GPU", vram_mb)], system_ram_mb=64000, cpu_threads=16, source="manual"
    )


def test_fits_on_gpu() -> None:
    r = rec.recommend(
        _gpu(24564), size_bytes=int(4.9e9), quant="Q4_K_M", config=CFG_8B, filename="Qwen3-8B-Q4_K_M.gguf"
    )
    assert r.fit == "gpu"
    assert r.n_gpu_layers == 999
    assert r.confidence == "high"
    assert r.kv_cache_type == "f16"


def test_partial_offload() -> None:
    r = rec.recommend(
        _gpu(24564), size_bytes=int(42e9), quant="Q4_K_M", config=CFG_70B, filename="L70-Q4_K_M.gguf"
    )
    assert r.fit == "partial"
    assert 0 < r.n_gpu_layers < CFG_70B["num_hidden_layers"]


def test_cpu_only() -> None:
    p = HardwareProfile(gpus=[], system_ram_mb=32000, cpu_threads=8, source="manual")
    r = rec.recommend(p, size_bytes=int(4.9e9), quant="Q4_K_M", config=CFG_8B)
    assert r.fit == "cpu"
    assert r.n_gpu_layers == 0
    assert r.threads == 8


def test_too_big() -> None:
    p = HardwareProfile(gpus=[], system_ram_mb=2000, cpu_threads=4, source="manual")
    r = rec.recommend(p, size_bytes=int(42e9), quant="Q4_K_M", config=CFG_70B)
    assert r.fit == "too_big"


def test_heuristic_low_confidence_no_config() -> None:
    r = rec.recommend(
        _gpu(24564), size_bytes=int(4.9e9), quant="Q4_K_M", config=None, filename="mistral-7b.Q4_K_M.gguf"
    )
    assert r.confidence == "low"
    assert r.fit in {"gpu", "partial", "cpu"}


def test_requested_context_capped_to_model_max() -> None:
    r = rec.recommend(
        _gpu(24564), size_bytes=int(4.9e9), config=CFG_70B, filename="L70.gguf", requested_context=999999
    )
    assert r.context_length <= CFG_70B["max_position_embeddings"]


def test_kv_cache_bytes_formula() -> None:
    assert rec.kv_cache_bytes(32, 8, 128, 4096, "f16") == 2 * 32 * 8 * 128 * 4096 * 2


def test_parse_params_billions() -> None:
    assert rec.parse_params_billions("Qwen3-8B-Q4_K_M.gguf") == 8.0
    assert rec.parse_params_billions("Llama-3.1-70B.gguf") == 70.0
    assert rec.parse_params_billions("model.gguf") is None


def test_launch_command_flags() -> None:
    r = rec.recommend(_gpu(24564), size_bytes=int(4.9e9), config=CFG_8B, filename="m.gguf")
    cmd = rec.launch_command("/models/m.gguf", r, alias="m")
    assert cmd.startswith("llama-server")
    assert "-m /models/m.gguf" in cmd
    assert "-ngl 999" in cmd
    assert f"-c {r.context_length}" in cmd
    assert "--flash-attn on" in cmd
    assert "--alias m" in cmd

    cmd_hf = rec.launch_command_hf("unsloth/Qwen3-8B-GGUF", "Q4_K_M", r)
    assert "-hf unsloth/Qwen3-8B-GGUF:Q4_K_M" in cmd_hf

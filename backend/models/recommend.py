"""Hardware-aware llama.cpp configuration recommendations (pure, network-free, unit-testable).

Given a :class:`backend.models.hardware.HardwareProfile`, a GGUF's on-disk size + quant, and
(optionally) the model's HuggingFace ``config.json``, classify how the model fits the NVIDIA hardware
and produce concrete ``llama-server`` settings: GPU layers (``-ngl``), context length (``-c``),
KV-cache quant (``-ctk``/``-ctv``), flash-attention, batch sizes, and threads — plus the exact launch
command to copy-run.

The numbers are deliberately approximations (real VRAM use depends on the build, the compute buffers,
and the exact KV layout); they're meant to get the user into the right ballpark, with the advanced
panel for fine-tuning. The KV-cache formula is the standard GQA one used by llama.cpp:

    kv_bytes = 2 (K and V) * n_layers * n_kv_heads * head_dim * context * bytes_per_element

where ``head_dim = hidden_size / n_attention_heads`` and ``bytes_per_element`` depends on the KV
cache type (f16≈2, q8_0≈1, q4_0≈0.5 bytes — approximate).
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any

_MIB = 1024 * 1024
_GIB = 1024 ** 3

# KV-cache bytes per stored element by cache type (approximate). f16 is exact; the quantized types
# carry a small per-block scale overhead we round off.
_KV_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}

# VRAM safety margin held back when deciding what fits (CUDA context + headroom). Scales with the
# card, so it is a *fit budget* only — never added to the reported estimate (which must reflect the
# model's footprint, not how big your GPU is).
_MIN_RESERVE = 1 * _GIB
# Realistic, GPU-size-independent compute/output buffer counted in the *reported* VRAM estimate.
_GPU_OVERHEAD = 512 * _MIB
# CPU-side runtime overhead beyond weights + KV (rough).
_CPU_OVERHEAD = 512 * _MIB

# Fallback architecture brackets when there's no config.json: (max_billions, n_layers, hidden,
# n_heads, n_kv_heads, max_ctx). Picked by the parameter count parsed from the filename. Rough — only
# used to put the estimate in the right order of magnitude (confidence="low").
_ARCH_BRACKETS = [
    (2.0, 24, 2048, 16, 4, 32768),
    (4.0, 28, 3072, 24, 8, 32768),
    (9.0, 32, 4096, 32, 8, 32768),
    (16.0, 40, 5120, 40, 8, 32768),
    (35.0, 64, 5120, 40, 8, 32768),
    (80.0, 80, 8192, 64, 8, 32768),
    (float("inf"), 96, 12288, 96, 8, 32768),
]
_DEFAULT_BRACKET = (32, 4096, 32, 8, 32768)  # used when no param count can be parsed at all

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


@dataclass
class Recommendation:
    fit: str  # "gpu" | "partial" | "cpu" | "too_big"
    n_gpu_layers: int
    context_length: int
    kv_cache_type: str
    flash_attn: bool
    batch_size: int
    ubatch_size: int
    threads: int
    est_vram_mb: int
    est_ram_mb: int
    confidence: str  # "high" (from config.json) | "low" (size-class heuristic)
    # VRAM estimate broken into its parts (sums to est_vram_mb) so the UI can show how each piece
    # contributes. Zero on CPU/too_big where nothing is offloaded.
    weights_mb: int = 0
    kv_cache_mb: int = 0
    overhead_mb: int = 0
    notes: list[str] = field(default_factory=list)
    # The model's own maximum context (from config.json's max_position_embeddings, else the size-class
    # bracket). The advanced UI uses it as the upper bound of the context-length slider.
    model_max_ctx: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def parse_params_billions(text: str) -> float | None:
    """Extract a parameter count in billions from a filename/repo id (``…-7B-…`` → ``7.0``)."""
    match = _PARAM_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _arch(config: dict[str, Any] | None, filename: str) -> dict[str, Any]:
    """Resolve architecture params (layers, KV heads, head_dim, max context) + a confidence level."""
    if config:
        hidden = int(config.get("hidden_size") or config.get("n_embd") or 4096)
        n_heads = int(config.get("num_attention_heads") or config.get("n_head") or 32)
        n_kv_heads = int(config.get("num_key_value_heads") or n_heads)
        n_layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 32)
        head_dim = int(config.get("head_dim") or (hidden // max(1, n_heads)))
        max_ctx = int(config.get("max_position_embeddings") or 32768)
        return {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "max_ctx": max_ctx,
            "confidence": "high",
        }
    params = parse_params_billions(filename)
    if params is None:
        n_layers, hidden, n_heads, n_kv_heads, max_ctx = _DEFAULT_BRACKET
    else:
        for max_b, n_layers, hidden, n_heads, n_kv_heads, max_ctx in _ARCH_BRACKETS:
            if params <= max_b:
                break
    return {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": hidden // max(1, n_heads),
        "max_ctx": max_ctx,
        "confidence": "low",
    }


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, context: int, kv_type: str) -> int:
    """KV-cache size in bytes for ``context`` tokens (the standard GQA formula)."""
    per_elem = _KV_BYTES.get(kv_type, 2.0)
    return int(2 * n_layers * n_kv_heads * head_dim * context * per_elem)


def _max_context(
    weights: int, usable_vram: int, n_layers: int, n_kv_heads: int, head_dim: int, kv_type: str, ceil: int
) -> int:
    """Largest context (≤ ``ceil``) whose weights + KV cache fit in ``usable_vram``; 0 if none do."""
    if weights >= usable_vram:
        return 0
    lo, hi, best = 512, ceil, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        need = weights + kv_cache_bytes(n_layers, n_kv_heads, head_dim, mid, kv_type)
        if need <= usable_vram:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _to_mb(byte_count: float) -> int:
    return int(byte_count / _MIB)


def recommend(
    profile,
    *,
    size_bytes: int,
    quant: str = "",
    config: dict[str, Any] | None = None,
    filename: str = "",
    requested_context: int | None = None,
    kv_cache_type: str | None = None,
) -> Recommendation:
    """Classify how a GGUF fits ``profile``'s NVIDIA hardware and recommend llama.cpp settings.

    ``requested_context``/``kv_cache_type`` let the advanced UI panel pin those and recompute. When a
    requested context is given it is honored (capped at the model's max); otherwise the recommender
    maximizes the context that fits on the GPU.
    """
    arch = _arch(config, filename or quant)
    n_layers, n_kv_heads, head_dim = arch["n_layers"], arch["n_kv_heads"], arch["head_dim"]
    model_max_ctx = arch["max_ctx"]
    confidence = arch["confidence"]

    weights = max(0, int(size_bytes))
    vram = profile.vram_total_mb * _MIB
    ram = profile.system_ram_mb * _MIB
    threads = profile.cpu_threads or os.cpu_count() or 8
    # Default to the model's own max and let the GPU path maximize the context that actually fits;
    # a pinned requested_context is honored (capped at the model max).
    target = min(requested_context or model_max_ctx, model_max_ctx)
    target = max(target, 512)

    notes = ["VRAM/KV-cache figures are estimates — fine-tune with the advanced controls."]
    if confidence == "low":
        notes.append("No config.json found; sizes estimated from the parameter count in the name.")

    # CPU-only machine (no GPU in the profile).
    if vram <= 0:
        kv = kv_cache_type or "f16"
        kv_b = kv_cache_bytes(n_layers, n_kv_heads, head_dim, target, kv)
        est_ram = weights + kv_b + _CPU_OVERHEAD
        fit = "cpu" if (ram <= 0 or est_ram <= ram) else "too_big"
        if fit == "too_big":
            notes.append("Model + KV cache exceed system RAM; pick a smaller quant.")
        else:
            notes.append("No GPU in the profile — runs on CPU + system RAM (slower).")
        return Recommendation(
            fit=fit, n_gpu_layers=0, context_length=target, kv_cache_type=kv, flash_attn=True,
            batch_size=512, ubatch_size=512, threads=threads, est_vram_mb=0,
            est_ram_mb=_to_mb(est_ram), confidence=confidence, notes=notes,
            model_max_ctx=model_max_ctx,
        )

    reserve = max(_MIN_RESERVE, int(0.1 * vram))
    usable = vram - reserve

    # Full GPU offload — try progressively cheaper KV cache until a useful context fits, maximizing it.
    kv_candidates = [kv_cache_type] if kv_cache_type else ["f16", "q8_0", "q4_0"]
    for kv in kv_candidates:
        max_fit = _max_context(weights, usable, n_layers, n_kv_heads, head_dim, kv, model_max_ctx)
        if max_fit >= min(target, 2048):
            ctx = min(max(target, 2048), max_fit, model_max_ctx)
            kv_b = kv_cache_bytes(n_layers, n_kv_heads, head_dim, ctx, kv)
            weights_mb, kv_mb, overhead_mb = _to_mb(weights), _to_mb(kv_b), _to_mb(_GPU_OVERHEAD)
            if kv != "f16":
                notes.append(f"Using {kv} KV cache so it fits at a useful context.")
            return Recommendation(
                fit="gpu", n_gpu_layers=999, context_length=ctx, kv_cache_type=kv, flash_attn=True,
                batch_size=2048, ubatch_size=512, threads=threads,
                est_vram_mb=weights_mb + kv_mb + overhead_mb, est_ram_mb=0, confidence=confidence,
                weights_mb=weights_mb, kv_cache_mb=kv_mb, overhead_mb=overhead_mb, notes=notes,
                model_max_ctx=model_max_ctx,
            )

    # Partial offload — lower the context, then offload as many layers as fit.
    kv = kv_cache_type or "q8_0"
    ctx = min(target, 4096, model_max_ctx)
    kv_total = kv_cache_bytes(n_layers, n_kv_heads, head_dim, ctx, kv)
    per_layer = weights / n_layers + kv_total / n_layers
    ngl = int((vram - reserve) / per_layer) if per_layer > 0 else 0
    ngl = max(0, min(ngl, n_layers))
    if ngl >= 1:
        est_ram = (n_layers - ngl) * per_layer
        weights_mb = _to_mb(ngl * weights / n_layers)
        kv_mb = _to_mb(ngl * kv_total / n_layers)
        overhead_mb = _to_mb(_GPU_OVERHEAD)
        notes.append(
            f"Partial offload: ~{ngl}/{n_layers} layers on GPU. Lower the context or KV quant to "
            "raise it."
        )
        return Recommendation(
            fit="partial", n_gpu_layers=ngl, context_length=ctx, kv_cache_type=kv, flash_attn=True,
            batch_size=512, ubatch_size=512, threads=threads,
            est_vram_mb=weights_mb + kv_mb + overhead_mb, est_ram_mb=_to_mb(est_ram),
            confidence=confidence, weights_mb=weights_mb, kv_cache_mb=kv_mb, overhead_mb=overhead_mb,
            notes=notes, model_max_ctx=model_max_ctx,
        )

    # Nothing offloads usefully → CPU if it fits RAM, else too big.
    kv = kv_cache_type or "f16"
    kv_b = kv_cache_bytes(n_layers, n_kv_heads, head_dim, ctx, kv)
    est_ram = weights + kv_b + _CPU_OVERHEAD
    if ram <= 0 or est_ram <= ram:
        notes.append("Too large for VRAM; will run on CPU + system RAM (slower).")
        fit = "cpu"
    else:
        notes.append("Exceeds both VRAM and system RAM; pick a smaller quant.")
        fit = "too_big"
    return Recommendation(
        fit=fit, n_gpu_layers=0, context_length=ctx, kv_cache_type=kv, flash_attn=True,
        batch_size=512, ubatch_size=512, threads=threads, est_vram_mb=0, est_ram_mb=_to_mb(est_ram),
        confidence=confidence, notes=notes, model_max_ctx=model_max_ctx,
    )


def launch_argv(
    model_path: str,
    rec: Recommendation,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    alias: str = "",
    api_key: str = "",
) -> list[str]:
    """The ``llama-server`` argv for ``rec`` as a raw (un-shell-quoted) token list.

    This is the structured form :func:`launch_command` renders for display AND the form the model
    manager feeds (minus the leading ``llama-server`` token) to ``docker create`` when auto-loading a
    model — the server-cuda image's entrypoint *is* ``llama-server``, so the container command is the
    flags only.
    """
    parts = [
        "llama-server",
        "-m", model_path,
        "-c", str(rec.context_length),
        "-ngl", str(rec.n_gpu_layers),
        "-ctk", rec.kv_cache_type,
        "-ctv", rec.kv_cache_type,
        "-b", str(rec.batch_size),
        "-ub", str(rec.ubatch_size),
        "-t", str(rec.threads),
    ]
    if rec.flash_attn:
        parts += ["--flash-attn", "on"]
    parts += ["--host", host, "--port", str(port)]
    if alias:
        parts += ["--alias", alias]
    if api_key:
        parts += ["--api-key", api_key]
    return parts


def launch_command(
    model_path: str,
    rec: Recommendation,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    alias: str = "",
    api_key: str = "",
) -> str:
    """Build the exact ``llama-server`` command for ``rec`` (the user runs this on the GPU machine)."""
    argv = launch_argv(model_path, rec, host=host, port=port, alias=alias, api_key=api_key)
    return " ".join(shlex.quote(p) for p in argv)


def launch_command_hf(
    repo_id: str,
    quant: str,
    rec: Recommendation,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    api_key: str = "",
) -> str:
    """A ``-hf`` variant: llama-server downloads the GGUF itself (handy for a remote server)."""
    target = f"{repo_id}:{quant}" if quant else repo_id
    parts = [
        "llama-server",
        "-hf", shlex.quote(target),
        "-c", str(rec.context_length),
        "-ngl", str(rec.n_gpu_layers),
        "-ctk", rec.kv_cache_type,
        "-ctv", rec.kv_cache_type,
        "-b", str(rec.batch_size),
        "-ub", str(rec.ubatch_size),
        "-t", str(rec.threads),
    ]
    if rec.flash_attn:
        parts += ["--flash-attn", "on"]
    parts += ["--host", host, "--port", str(port)]
    if api_key:
        parts += ["--api-key", shlex.quote(api_key)]
    return " ".join(parts)


__all__ = [
    "Recommendation",
    "recommend",
    "kv_cache_bytes",
    "launch_argv",
    "launch_command",
    "launch_command_hf",
    "parse_params_billions",
]

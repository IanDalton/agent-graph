"""Local model management for the llama.cpp provider.

These modules back the Model Manager UI — they are **not** Pydantic AI capabilities and are
system-level (machine-wide), not per-user:

- :mod:`backend.models.huggingface` — discover/inspect/download GGUF models from HuggingFace.
- :mod:`backend.models.hardware` — the editable NVIDIA hardware profile (the source of truth for
  recommendations) + best-effort ``nvidia-smi`` detection.
- :mod:`backend.models.recommend` — pure VRAM/KV-cache math → fit classification + ``llama-server``
  launch command.
- :mod:`backend.models.library` — the filesystem manifest of downloaded GGUFs (shared with the
  external llama-server via ``LLAMACPP_MODELS_DIR``).

Distinct from :mod:`backend.model_selection`, which resolves the *active* Pydantic AI model.
"""

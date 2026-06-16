"""Shared env-driven model selection.

Used by the main agent (``build_agent``) and the ontology evaluator sub-agent so both resolve a
model the same way: an explicit Pydantic AI model string from ``primary_env`` (e.g.
``AGENT_MODEL=openai:gpt-5.2``), or else a local Ollama model named by ``fallback_env``.

This lives in a dependency-free leaf module on purpose: ``backend.main`` imports
``backend.skills.ontology_capability``, so the evaluator cannot import its model selection from
``main`` without creating an import cycle.
"""

from __future__ import annotations

import os

from pydantic_ai.models import Model


def select_model(
    primary_env: str, fallback_env: str = "OLLAMA_MODEL", fallback_default: str = "qwen3"
) -> Model | str:
    """Resolve a model from env: ``primary_env`` if set, else a local Ollama model.

    Returns a model *string* (passed straight to ``Agent``) when ``primary_env`` is set, otherwise
    an ``OllamaModel`` instance. The Ollama import is lazy so importing this module never requires
    the local-model dependency when a hosted provider is configured.
    """
    model_string = os.getenv(primary_env)
    if model_string:
        return model_string
    from pydantic_ai.models.ollama import OllamaModel

    return OllamaModel(os.getenv(fallback_env, fallback_default))


# The built-in dropdown choices when ``AGENT_MODELS`` is unset. Labels follow the same convention
# as :func:`default_model_label` / :func:`resolve_model`: a Pydantic AI model string for a hosted
# provider, or ``ollama/<name>`` for a local Ollama model.
DEFAULT_MODELS = [
    "anthropic:claude-opus-4-8",
    "anthropic:claude-sonnet-4-6",
    "openai:gpt-5.2",
    "ollama/qwen3",
]


def default_model_label() -> str:
    """The currently-configured default model as a UI-facing label.

    ``AGENT_MODEL`` verbatim when set (e.g. ``openai:gpt-5.2``), else ``ollama/<OLLAMA_MODEL>``.
    This is what the agent uses when a request carries no explicit model override.
    """
    agent_model = os.getenv("AGENT_MODEL")
    return agent_model or f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen3')}"


def available_models() -> list[str]:
    """The selectable model labels for the UI dropdown.

    From ``AGENT_MODELS`` (comma-separated) when set, else :data:`DEFAULT_MODELS`. The current
    :func:`default_model_label` is always included (prepended if missing) so the configured model
    is never an unselectable option.
    """
    raw = os.getenv("AGENT_MODELS")
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        models = list(DEFAULT_MODELS)
    default = default_model_label()
    if default not in models:
        models.insert(0, default)
    return models


# Approximate context-window sizes (in tokens) for the known model labels, used by the
# context-window usage meter. These are deliberately rough — exact numbers vary by provider tier and
# change over time — and fully overridable via the ``MODEL_CONTEXT_WINDOWS`` env var (a
# comma-separated list of ``label=size`` pairs, e.g. ``"openai:gpt-5.2=400000,ollama/qwen3=32768"``).
CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic:claude-opus-4-8": 200_000,
    "anthropic:claude-sonnet-4-6": 200_000,
    "openai:gpt-5.2": 400_000,
    "ollama/qwen3": 32_768,
}

# Fallback window for an unknown model label (a conservative modern default).
DEFAULT_CONTEXT_WINDOW = 128_000


def _env_context_windows() -> dict[str, int]:
    """Parse ``MODEL_CONTEXT_WINDOWS`` (``label=size`` pairs) into a mapping; ignore bad entries."""
    raw = os.getenv("MODEL_CONTEXT_WINDOWS")
    if not raw:
        return {}
    out: dict[str, int] = {}
    for pair in raw.split(","):
        label, _, size = pair.partition("=")
        label = label.strip()
        try:
            if label and size.strip():
                out[label] = int(size.strip())
        except ValueError:
            continue  # a malformed override entry is skipped, not fatal
    return out


def context_window_for(label: str) -> int:
    """The context-window size (tokens) for a UI model label.

    Env overrides (``MODEL_CONTEXT_WINDOWS``) win over the built-in :data:`CONTEXT_WINDOWS`; an
    unknown label falls back to :data:`DEFAULT_CONTEXT_WINDOW`.
    """
    label = (label or "").strip()
    overrides = _env_context_windows()
    if label in overrides:
        return overrides[label]
    return CONTEXT_WINDOWS.get(label, DEFAULT_CONTEXT_WINDOW)


# Model-label prefixes assumed NOT to support image/PDF (vision) input. Local Ollama models are
# treated as text-only by default; a multimodal local model can opt in via the VISION_MODELS env
# (a comma-separated list of labels that ARE vision-capable). Used only for a soft warning in
# stream_run, so an over/under-guess is harmless — it never blocks a turn.
NON_VISION_PREFIXES = ("ollama/",)


def is_vision_capable(label: str) -> bool:
    """Best-effort guess at whether a UI model label can read image/PDF input.

    The hosted defaults (Anthropic, OpenAI) are vision-capable; local Ollama labels are assumed
    text-only unless listed in ``VISION_MODELS``. This drives only the "this model may not be able
    to see images" note attached to uploads, never any hard behaviour.
    """
    label = (label or "").strip()
    overrides = {m.strip() for m in os.getenv("VISION_MODELS", "").split(",") if m.strip()}
    if label in overrides:
        return True
    return not any(label.startswith(prefix) for prefix in NON_VISION_PREFIXES)


def resolve_model(model: str | None) -> Model | str:
    """Resolve an explicit UI-selected model label, falling back to the env default.

    ``model`` uses the same labels as :func:`available_models`: a Pydantic AI model string passed
    straight to ``Agent`` (``openai:gpt-5.2``), or ``ollama/<name>`` for a local Ollama model. When
    blank/None, defers to the env-driven :func:`select_model` (``AGENT_MODEL`` or Ollama fallback).
    """
    model = (model or "").strip()
    if not model:
        return select_model("AGENT_MODEL")
    if model.startswith("ollama/"):
        from pydantic_ai.models.ollama import OllamaModel

        return OllamaModel(model[len("ollama/") :])
    return model

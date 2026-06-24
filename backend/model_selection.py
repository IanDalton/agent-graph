"""Shared env-driven model selection (llama.cpp local provider).

Used by the main agent (``build_agent``) and the leaf write-time agents (title/summary/curator,
ontology evaluator) so they all resolve a model the same way: an explicit Pydantic AI model string
from ``primary_env`` (e.g. ``AGENT_MODEL=openai:gpt-5.2``, a power-user escape hatch), or else a
**local llama.cpp model** served over llama-server's OpenAI-compatible API.

The agent connects to an *external* ``llama-server`` (the app does not manage that process — see the
Model Manager UI). ``LLAMACPP_BASE_URL`` points at its ``/v1`` endpoint (e.g.
``http://localhost:8080/v1``). llama-server typically serves one model at a time and loosely matches
the request's ``model`` field, so the model *name* we send is mostly informational; the real model
is whatever the server was launched with (surfaced to the UI via ``/v1/models``).

This lives in a dependency-free leaf module on purpose: ``backend.main`` imports
``backend.skills.ontology_capability``, so the evaluator cannot import its model selection from
``main`` without creating an import cycle. The provider/model classes are imported lazily inside the
builder so importing this module stays cheap.
"""

from __future__ import annotations

import os

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

# llama-server's default OpenAI-compatible endpoint. The compose service defaults to
# ``http://llamacpp:8080/v1``; a standalone server on the host is ``http://localhost:8080/v1``.
DEFAULT_LLAMACPP_BASE_URL = "http://localhost:8080/v1"

# A neutral name used when no specific model name is configured. llama-server ignores/loosely
# matches the request ``model`` for a single-model server, so this is informational only.
DEFAULT_LOCAL_MODEL = "local-model"

# The UI label prefix for a local llama.cpp model (parallels the old ``ollama/`` convention). A label
# is either a hosted Pydantic AI string (``provider:model`` — only via the AGENT_MODEL escape hatch)
# or ``local/<name>`` for a llama.cpp model.
LOCAL_PREFIX = "local/"


def _local_settings() -> ModelSettings | None:
    """Build per-request model settings for the local model from env, or ``None``.

    Local reasoning models can get cut off mid-chain-of-thought when the output budget is too small,
    which leaves the answer trapped on the thinking channel (see ``main.stream_run``'s fallback).
    ``LLAMACPP_NUM_PREDICT`` → ``max_tokens`` (the max output/answer tokens) widens that budget.

    There is deliberately no per-request context-window knob: llama-server's context size is fixed at
    launch by its ``-c/--ctx-size`` flag (the Model Manager generates that command), not negotiated
    per request. Returns ``None`` when unset, so default behaviour is unchanged.
    """
    settings: ModelSettings = {}
    num_predict = os.getenv("LLAMACPP_NUM_PREDICT")
    if num_predict:
        try:
            settings["max_tokens"] = int(num_predict)
        except ValueError:
            pass  # a malformed override is ignored, not fatal
    return settings or None


def _llamacpp_model(model_name: str) -> Model:
    """Build an ``OpenAIChatModel`` pointed at the local llama-server's OpenAI-compatible API.

    The api_key is set **explicitly** (``LLAMACPP_API_KEY`` or a placeholder): ``OpenAIProvider``
    only injects a placeholder when ``OPENAI_API_KEY`` is *also* unset, but this app sets
    ``OPENAI_API_KEY`` for embeddings — so without an explicit key the OpenAI key would leak into the
    llama.cpp requests. The imports are local so this leaf module stays import-cheap.
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    base_url = os.getenv("LLAMACPP_BASE_URL", DEFAULT_LLAMACPP_BASE_URL)
    api_key = os.getenv("LLAMACPP_API_KEY") or "api-key-not-set"
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model_name or DEFAULT_LOCAL_MODEL, provider=provider, settings=_local_settings())


def select_model(
    primary_env: str, fallback_env: str = "LLAMACPP_MODEL", fallback_default: str = DEFAULT_LOCAL_MODEL
) -> Model | str:
    """Resolve a model from env: ``primary_env`` if set, else a local llama.cpp model.

    Returns a model *string* (passed straight to ``Agent``) when ``primary_env`` is set (the hosted
    escape hatch, e.g. ``AGENT_MODEL=openai:gpt-5.2``), otherwise an ``OpenAIChatModel`` instance
    pointed at llama-server (named by ``fallback_env``/``fallback_default``).
    """
    model_string = os.getenv(primary_env)
    if model_string:
        return model_string
    return _llamacpp_model(os.getenv(fallback_env, fallback_default))


# The built-in dropdown choices when ``AGENT_MODELS`` is unset. In a llama.cpp-only setup the real
# picker list is the user's *downloaded* GGUF models, merged in by the API layer (``/api/config``);
# this leaf module is filesystem-free, so it only contributes the configured default label.
DEFAULT_MODELS: list[str] = []


def default_model_label() -> str:
    """The currently-configured default model as a UI-facing label.

    ``AGENT_MODEL`` verbatim when set (the hosted escape hatch), else ``local/<LLAMACPP_MODEL>``. This
    is what the agent uses when a request carries no explicit model override.
    """
    agent_model = os.getenv("AGENT_MODEL")
    return agent_model or f"{LOCAL_PREFIX}{os.getenv('LLAMACPP_MODEL') or DEFAULT_LOCAL_MODEL}"


def available_models() -> list[str]:
    """The selectable model labels for the UI dropdown (fallback set).

    From ``AGENT_MODELS`` (comma-separated) when set, else :data:`DEFAULT_MODELS`. The current
    :func:`default_model_label` is always included (prepended if missing). The API layer overlays the
    discovered local GGUF models on top of this (see ``backend.api`` ``/api/config``), so this is the
    fallback used only when no models have been downloaded yet.
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


# Approximate context-window sizes (in tokens) for known model labels, used by the context-window
# usage meter. For a *local* model the real window is the server's launched ``-c`` value, so locals
# fall back to :data:`DEFAULT_LOCAL_CONTEXT_WINDOW` (overridable via ``MODEL_CONTEXT_WINDOWS``, a
# comma-separated list of ``label=size`` pairs, e.g. ``"local/qwen3-8b=32768"``). The hosted entries
# are kept only so the meter sizes correctly when the ``AGENT_MODEL`` escape hatch is in use.
CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic:claude-opus-4-8": 200_000,
    "anthropic:claude-sonnet-4-6": 200_000,
    "openai:gpt-5.2": 400_000,
}

# Fallback window for an unknown local model label (llama-server's common default ``-c`` is small;
# 8192 is a safe, conservative meter default).
DEFAULT_LOCAL_CONTEXT_WINDOW = 8192
DEFAULT_CONTEXT_WINDOW = DEFAULT_LOCAL_CONTEXT_WINDOW


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
    unknown label falls back to :data:`DEFAULT_LOCAL_CONTEXT_WINDOW`.
    """
    label = (label or "").strip()
    overrides = _env_context_windows()
    if label in overrides:
        return overrides[label]
    return CONTEXT_WINDOWS.get(label, DEFAULT_LOCAL_CONTEXT_WINDOW)


# Model-label prefixes assumed NOT to support image/PDF (vision) input. Local llama.cpp models are
# treated as text-only by default; a multimodal GGUF (which needs an ``--mmproj`` file at launch) can
# opt in via the VISION_MODELS env (a comma-separated list of labels that ARE vision-capable). Used
# only for a soft warning in stream_run, so an over/under-guess is harmless — it never blocks a turn.
NON_VISION_PREFIXES = (LOCAL_PREFIX,)


def is_vision_capable(label: str) -> bool:
    """Best-effort guess at whether a UI model label can read image/PDF input.

    Local llama.cpp labels are assumed text-only unless listed in ``VISION_MODELS``; a hosted label
    (only reachable via the ``AGENT_MODEL`` escape hatch) is assumed vision-capable. This drives only
    the "this model may not be able to see images" note attached to uploads, never any hard behaviour.
    """
    label = (label or "").strip()
    overrides = {m.strip() for m in os.getenv("VISION_MODELS", "").split(",") if m.strip()}
    if label in overrides:
        return True
    return not any(label.startswith(prefix) for prefix in NON_VISION_PREFIXES)


def resolve_model(model: str | None) -> Model | str:
    """Resolve an explicit UI-selected model label, falling back to the env default.

    ``model`` uses the same labels as :func:`available_models`: ``local/<name>`` for a local
    llama.cpp model, or a hosted Pydantic AI string (``openai:gpt-5.2`` — only via the AGENT_MODEL
    escape hatch). A bare name with no provider ``:`` is treated as a local model name. When
    blank/None, defers to the env-driven :func:`select_model` (``AGENT_MODEL`` or llama.cpp fallback).
    """
    model = (model or "").strip()
    if not model:
        return select_model("AGENT_MODEL")
    if model.startswith(LOCAL_PREFIX):
        return _llamacpp_model(model[len(LOCAL_PREFIX) :])
    if ":" not in model:
        # A bare model name (no hosted ``provider:model`` form) is a local llama.cpp model.
        return _llamacpp_model(model)
    return model

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

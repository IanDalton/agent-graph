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

"""Token counting for the context-window usage meter.

A small, tolerant helper used by the ``/api/conversations/{id}/context`` endpoint to estimate how
many tokens a conversation's system prompt, tool definitions, and message history consume. Uses
``tiktoken`` for precise counts when available, and falls back to a character heuristic (``chars/4``)
if ``tiktoken`` is missing or its BPE files can't be loaded (e.g. offline on first use — tiktoken
downloads the encoding the first time). The fallback means token counting can never raise, mirroring
the tolerance contract of the rest of the codebase (``run_query``/``web_search`` never abort a run).

The chosen encoder is reported back as a ``counter`` label (e.g. ``"tiktoken:o200k_base"`` or
``"heuristic:chars/4"``) so the UI can show whether the numbers are precise or estimated.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agent_graph.token_count")

# The modern OpenAI encoding; a reasonable generic approximation for non-OpenAI models (Claude,
# Ollama) too, since we don't ship their proprietary tokenizers.
_DEFAULT_ENCODING = "o200k_base"

# Cache loaded encoders by name so we pay tiktoken's load cost at most once per encoding.
_encoders: dict[str, object] = {}
# Once tiktoken proves unavailable/unloadable we stop retrying and stay on the heuristic.
_tiktoken_broken = False


def _encoding_name_for(model_label: str) -> str:
    """Pick a tiktoken encoding name for a UI model label (e.g. ``openai:gpt-5.2``)."""
    label = (model_label or "").lower()
    if "gpt-4o" in label or "o200k" in label:
        return "o200k_base"
    # OpenAI's older chat models used cl100k_base; default everything else to the modern encoding.
    if "gpt-3.5" in label or "gpt-4" in label:
        return "cl100k_base"
    return _DEFAULT_ENCODING


def _get_encoder(name: str):
    """Return a cached tiktoken encoder, or ``None`` if tiktoken can't be used."""
    global _tiktoken_broken
    if _tiktoken_broken:
        return None
    cached = _encoders.get(name)
    if cached is not None:
        return cached
    try:
        import tiktoken

        encoder = tiktoken.get_encoding(name)
    except Exception:  # noqa: BLE001 — missing dep or offline BPE fetch must not break counting.
        logger.warning("tiktoken unavailable; falling back to heuristic token counting", exc_info=True)
        _tiktoken_broken = True
        return None
    _encoders[name] = encoder
    return encoder


def count_tokens(text: str, model_label: str = "") -> tuple[int, str]:
    """Count tokens in ``text`` for ``model_label``.

    Returns ``(token_count, counter_label)`` where ``counter_label`` is ``"tiktoken:<encoding>"`` on
    the precise path or ``"heuristic:chars/4"`` on the fallback. Never raises.
    """
    if not text:
        # Still report which counter we *would* use, so the API surfaces a stable label.
        name = _encoding_name_for(model_label)
        return 0, f"tiktoken:{name}" if _get_encoder(name) is not None else "heuristic:chars/4"
    name = _encoding_name_for(model_label)
    encoder = _get_encoder(name)
    if encoder is not None:
        try:
            return len(encoder.encode(text)), f"tiktoken:{name}"
        except Exception:  # noqa: BLE001 — a pathological string must not break the meter.
            logger.warning("tiktoken encode failed; using heuristic", exc_info=True)
    # Heuristic: ~4 characters per token is the usual rough rule of thumb.
    return max(1, len(text) // 4), "heuristic:chars/4"

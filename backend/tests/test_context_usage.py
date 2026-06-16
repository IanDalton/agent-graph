"""Unit tests for the context-window usage meter's pure helpers.

These need no DB/network/Docker: they exercise the token counter, the model→window mapping, and the
tool-definition introspection that backs ``GET /api/conversations/{id}/context``.
"""

from __future__ import annotations

import backend.token_count as token_count
from backend import main
from backend.model_selection import (
    DEFAULT_CONTEXT_WINDOW,
    context_window_for,
)


def test_count_tokens_precise_path() -> None:
    count, counter = token_count.count_tokens("hello world, this is a test", "openai:gpt-5.2")
    assert count > 0
    assert counter.startswith("tiktoken:")


def test_count_tokens_empty_is_zero() -> None:
    count, counter = token_count.count_tokens("", "openai:gpt-5.2")
    assert count == 0
    assert counter  # a stable label is still reported


def test_count_tokens_heuristic_fallback(monkeypatch) -> None:
    # Force the tiktoken path to be unavailable and confirm we degrade to the char heuristic
    # rather than raising. Reset the module's cache/flag so the patch is observed.
    monkeypatch.setattr(token_count, "_encoders", {})
    monkeypatch.setattr(token_count, "_tiktoken_broken", False)

    def _boom(_name: str):
        return None  # simulate "encoder unavailable"

    monkeypatch.setattr(token_count, "_get_encoder", _boom)
    text = "a" * 40
    count, counter = token_count.count_tokens(text, "anything")
    assert counter == "heuristic:chars/4"
    assert count == 10  # 40 chars // 4


def test_context_window_known_label() -> None:
    assert context_window_for("anthropic:claude-opus-4-8") == 200_000


def test_context_window_unknown_falls_back() -> None:
    assert context_window_for("made-up:model") == DEFAULT_CONTEXT_WINDOW


def test_context_window_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CONTEXT_WINDOWS", "made-up:model=4096, openai:gpt-5.2=999")
    assert context_window_for("made-up:model") == 4096
    assert context_window_for("openai:gpt-5.2") == 999  # override wins over the built-in default


def test_context_window_env_override_ignores_bad_entries(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CONTEXT_WINDOWS", "good:m=10,bad-entry,also:bad=notanint")
    assert context_window_for("good:m") == 10
    # A malformed entry doesn't crash; the unknown label just falls back.
    assert context_window_for("also:bad") == DEFAULT_CONTEXT_WINDOW


def test_tool_definitions_json_regular_has_tools() -> None:
    serialized = main.tool_definitions_json("regular")
    assert "search_memory" in serialized
    assert "web_search" in serialized


def test_tool_definitions_json_swarm_is_leaner_than_regular() -> None:
    # The swarm orchestrator is a pure router with far fewer "doing" tools, so its serialized tool
    # definitions must be smaller than the full regular profile's.
    regular = main.tool_definitions_json("regular")
    swarm = main.tool_definitions_json("swarm")
    assert 0 < len(swarm) < len(regular)
    assert "run_python" not in swarm  # sandbox tool is not granted to the orchestrator


def test_message_history_text_flattens_parts() -> None:
    # message_history_text reuses _to_message_history; a corrupt/empty blob just yields no text.
    assert main.message_history_text([]) == ""
    assert main.message_history_text([{"raw": ""}]) == ""

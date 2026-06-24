"""Tests for the llama.cpp-oriented model selection (provider switch + escape hatch)."""

from __future__ import annotations

from pydantic_ai.models.openai import OpenAIChatModel

from backend import model_selection as m


def test_resolve_local_builds_openai_chat_model(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-embed-secret")  # must NOT leak into the llama.cpp client
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://host:8080/v1")
    monkeypatch.delenv("LLAMACPP_API_KEY", raising=False)
    mdl = m.resolve_model("local/qwen3-8b")
    assert isinstance(mdl, OpenAIChatModel)
    assert str(mdl.client.base_url) == "http://host:8080/v1/"
    assert mdl.client.api_key == "api-key-not-set"  # explicit placeholder, not the OpenAI key
    assert mdl.model_name == "qwen3-8b"


def test_explicit_llamacpp_api_key(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://host:8080/v1")
    monkeypatch.setenv("LLAMACPP_API_KEY", "secret-key")
    mdl = m.resolve_model("local/x")
    assert mdl.client.api_key == "secret-key"


def test_bare_name_is_local(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://h:8080/v1")
    mdl = m.resolve_model("foo")
    assert isinstance(mdl, OpenAIChatModel)
    assert mdl.model_name == "foo"


def test_default_label_local(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("LLAMACPP_MODEL", "qwen3-8b")
    assert m.default_model_label() == "local/qwen3-8b"


def test_agent_model_escape_hatch(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-5.2")
    assert m.resolve_model("") == "openai:gpt-5.2"
    assert m.default_model_label() == "openai:gpt-5.2"
    assert m.resolve_model("openai:gpt-5.2") == "openai:gpt-5.2"  # hosted passthrough


def test_vision_default_off_for_local(monkeypatch) -> None:
    monkeypatch.delenv("VISION_MODELS", raising=False)
    assert m.is_vision_capable("local/x") is False
    monkeypatch.setenv("VISION_MODELS", "local/x")
    assert m.is_vision_capable("local/x") is True


def test_context_window_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_CONTEXT_WINDOWS", raising=False)
    assert m.context_window_for("local/anything") == m.DEFAULT_LOCAL_CONTEXT_WINDOW
    monkeypatch.setenv("MODEL_CONTEXT_WINDOWS", "local/big=131072")
    assert m.context_window_for("local/big") == 131072

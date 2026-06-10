"""Tests for the WebSearch capability (SearXNG search + page fetch).

All unit tests use a duck-typed fake WebClient (injected via deps.web) or monkeypatch the real
client's httpx call, so they need no network. The tools are plain coroutines callable with a
hand-built RunContext, like the other capability tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend.db.dependencies import GraphDependencies
from backend.schemas.search_schemas import FetchUrlArgs, WebSearchArgs
from backend.skills.search_capability import build_search, fetch_url, web_search
from backend.web.client import WebClient, html_to_text

EXPECTED_TOOLS = {"web_search", "fetch_url"}


class FakeWebClient:
    """Duck-typed stand-in for WebClient returning canned results (or raising)."""

    def __init__(
        self,
        *,
        results: list[dict[str, Any]] | None = None,
        page: tuple[str, bool] = ("page text", False),
        error: Exception | None = None,
    ) -> None:
        self._results = results or []
        self._page = page
        self._error = error

    async def search(self, query: str, **_kw: Any) -> list[dict[str, Any]]:
        if self._error:
            raise self._error
        return self._results

    async def fetch(self, url: str) -> tuple[str, bool]:
        if self._error:
            raise self._error
        return self._page


def _ctx(deps: GraphDependencies) -> RunContext[GraphDependencies]:
    """Minimal RunContext for invoking the web tool coroutines directly."""
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def _deps(web: Any) -> GraphDependencies:
    return GraphDependencies(db=object(), user_id="u", conversation_id="c", web=web)


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #
def test_tools_are_registered() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, deps_type=GraphDependencies, capabilities=[*build_search()])
    deps = _deps(FakeWebClient())
    asyncio.run(agent.run("hi", deps=deps))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert EXPECTED_TOOLS <= names


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #
def test_web_search_parses_hits() -> None:
    rows = [
        {"title": "ArcadeDB", "url": "https://arcadedb.com", "content": "A graph DB", "engine": "duckduckgo"},
        {"title": "Other", "url": "https://example.com", "content": "blah"},
    ]
    deps = _deps(FakeWebClient(results=rows))
    result = asyncio.run(web_search(_ctx(deps), WebSearchArgs(query="arcadedb")))
    assert result.error is None
    assert [h.title for h in result.hits] == ["ArcadeDB", "Other"]
    assert result.hits[0].snippet == "A graph DB"
    assert result.hits[0].engine == "duckduckgo"
    assert result.hits[1].engine is None  # missing engine -> None, not crash


def test_web_search_error_is_returned_not_raised() -> None:
    """A search failure must come back as an error result, never abort the run."""
    req = httpx.Request("GET", "http://localhost:8085/search")
    err = httpx.HTTPStatusError("boom", request=req, response=httpx.Response(500, request=req))
    deps = _deps(FakeWebClient(error=err))
    result = asyncio.run(web_search(_ctx(deps), WebSearchArgs(query="x")))
    assert result.hits == []
    assert result.error and "Search failed" in result.error


# --------------------------------------------------------------------------- #
# fetch_url
# --------------------------------------------------------------------------- #
def test_fetch_url_returns_text_and_truncated_flag() -> None:
    deps = _deps(FakeWebClient(page=("the page body", True)))
    result = asyncio.run(fetch_url(_ctx(deps), FetchUrlArgs(url="https://example.com")))
    assert result.error is None
    assert result.text == "the page body"
    assert result.truncated is True


def test_fetch_url_error_is_returned_not_raised() -> None:
    deps = _deps(FakeWebClient(error=httpx.ConnectError("no route")))
    result = asyncio.run(fetch_url(_ctx(deps), FetchUrlArgs(url="https://example.com")))
    assert result.text == ""
    assert result.error and "Fetch failed" in result.error


# --------------------------------------------------------------------------- #
# FetchUrlArgs scheme validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", ["https://ok.com", "http://ok.com/path?q=1"])
def test_fetch_url_args_accepts_http(url: str) -> None:
    assert FetchUrlArgs(url=url).url == url


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "javascript:alert(1)", ""])
def test_fetch_url_args_rejects_non_http(url: str) -> None:
    with pytest.raises(ValidationError):
        FetchUrlArgs(url=url)


# --------------------------------------------------------------------------- #
# html_to_text
# --------------------------------------------------------------------------- #
def test_html_to_text_strips_and_collapses() -> None:
    html = (
        "<html><head><title>T</title><style>.x{color:red}</style></head>"
        "<body><script>var a=1;</script><h1>Hello   World</h1>"
        "<p>Some\n\n  text</p></body></html>"
    )
    text = html_to_text(html)
    assert "Hello World" in text
    assert "Some text" in text
    assert "var a=1" not in text  # script content dropped
    assert "color:red" not in text  # style content dropped


# --------------------------------------------------------------------------- #
# WebClient.search wiring (monkeypatched httpx, no network)
# --------------------------------------------------------------------------- #
def test_web_client_search_sends_json_and_trims() -> None:
    async def main() -> None:
        async with WebClient(searxng_url="http://searx.local") as client:
            captured: dict[str, Any] = {}
            payload = {
                "results": [
                    {"title": f"r{i}", "url": f"https://e/{i}", "content": "c"} for i in range(5)
                ]
            }

            async def fake_get(url: str, params: dict[str, str] | None = None) -> httpx.Response:
                captured["url"] = url
                captured["params"] = params or {}
                return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

            client._client.get = fake_get  # type: ignore[assignment]
            rows = await client.search("hello", max_results=2)
            assert captured["url"] == "http://searx.local/search"
            assert captured["params"]["format"] == "json"
            assert captured["params"]["q"] == "hello"
            assert len(rows) == 2  # trimmed to max_results

    asyncio.run(main())


def test_web_client_fetch_caps_content() -> None:
    async def main() -> None:
        async with WebClient(max_content_bytes=10) as client:
            big = "<p>" + ("x" * 100) + "</p>"

            async def fake_get(url: str, params: dict[str, str] | None = None) -> httpx.Response:
                return httpx.Response(200, text=big, request=httpx.Request("GET", url))

            client._client.get = fake_get  # type: ignore[assignment]
            text, truncated = await client.fetch("https://example.com")
            assert truncated is True  # body exceeded the 10-byte cap
            assert isinstance(text, str)

    asyncio.run(main())

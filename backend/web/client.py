"""Async HTTP client for web access: SearXNG search + page fetching.

Two responsibilities, both thin wrappers over a single reused :class:`httpx.AsyncClient`:

- :meth:`WebClient.search` → SearXNG's JSON API (``GET /search?format=json``). SearXNG is the
  self-hosted metasearch engine shipped in ``docker-compose.yml`` (the ``searxng`` service on
  ``:8085``). Its JSON format and bot-limiter must be enabled in ``docker/searxng/settings.yml``.
- :meth:`WebClient.fetch` → download an arbitrary result page and strip it to readable text via
  :func:`html_to_text`.

Connection settings come from the environment (``SEARXNG_URL``) with the docker-compose default
baked in, so a fresh checkout works out of the box. Modeled on
:class:`backend.db.arcade_db.ArcadeClient`: env-driven config, a single reused client, an async
context manager, and capped exponential-backoff retries on transient (transport / 5xx) failures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from html.parser import HTMLParser

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agent_graph.web")

# Default mirrors docker-compose.yml (searxng service, host port 8085).
DEFAULT_SEARXNG_URL = "http://localhost:8085"
# A browser-like UA: some SearXNG instances and target sites reject empty/obviously-bot agents.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; agent-graph/1.0; +https://github.com/agent-graph) "
    "AppleWebKit/537.36 (KHTML, like Gecko)"
)
_WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# HTML -> text extraction (stdlib only; no BeautifulSoup/readability dependency)
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    """Collect visible text, skipping the content of non-visible elements."""

    _SKIP_TAGS = frozenset({"script", "style", "head", "noscript", "template", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = _WHITESPACE_RE.sub(" ", data).strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


def html_to_text(html: str) -> str:
    """Strip an HTML document to its visible text, collapsing whitespace.

    Drops the content of script/style/head/etc. Pure and network-free, so it is unit-testable
    in isolation. Malformed markup is tolerated rather than raised.
    """
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 — malformed HTML must not crash extraction.
        logger.debug("html_to_text: parser error on malformed markup; returning partial text")
    return parser.text()


class WebClient:
    """Thin async wrapper for web search (SearXNG) and page fetching.

    Reuses a single :class:`httpx.AsyncClient`. Call :meth:`aclose` on shutdown, or use the
    client as an async context manager.
    """

    def __init__(
        self,
        searxng_url: str | None = None,
        *,
        timeout: float = 20.0,
        max_content_bytes: int = 2_000_000,
        max_retries: int = 2,
        retry_base_delay: float = 0.1,
        retry_max_delay: float = 2.0,
    ) -> None:
        self.searxng_url = (searxng_url or os.getenv("SEARXNG_URL", DEFAULT_SEARXNG_URL)).rstrip("/")
        self.max_content_bytes = max_content_bytes
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
        )

    async def __aenter__(self) -> "WebClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff for ``attempt`` (0-based), capped at ``retry_max_delay``."""
        return min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)

    async def _get_with_retry(self, url: str, *, params: dict[str, str] | None = None) -> httpx.Response:
        """GET ``url``, retrying transient failures (transport errors, 5xx) with capped backoff.

        A final 4xx/5xx is raised via ``raise_for_status``; the caller decides how to surface it.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "web transport error on %s (attempt %d/%d): %s; retrying",
                        url, attempt + 1, self.max_retries + 1, exc,
                    )
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                break
            if resp.status_code >= 500 and attempt < self.max_retries:
                last_exc = httpx.HTTPStatusError(
                    f"{resp.status_code} from {url}", request=resp.request, response=resp
                )
                logger.warning(
                    "web %d on %s (attempt %d/%d); retrying",
                    resp.status_code, url, attempt + 1, self.max_retries + 1,
                )
                await asyncio.sleep(self._backoff(attempt))
                continue
            resp.raise_for_status()
            return resp
        assert last_exc is not None  # the loop ran at least once
        raise last_exc

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        categories: str | None = None,
        time_range: str | None = None,
    ) -> list[dict[str, object]]:
        """Run a SearXNG search and return up to ``max_results`` raw result dicts.

        Each dict carries SearXNG's fields (``title``, ``url``, ``content``, ``engine``, ...).
        Requires the JSON output format to be enabled on the SearXNG instance.
        """
        params: dict[str, str] = {"q": query, "format": "json", "safesearch": "1"}
        if categories:
            params["categories"] = categories
        if time_range:
            params["time_range"] = time_range
        resp = await self._get_with_retry(f"{self.searxng_url}/search", params=params)
        results = resp.json().get("results", []) or []
        return results[:max_results]

    async def fetch(self, url: str) -> tuple[str, bool]:
        """Download ``url`` and return ``(readable_text, truncated)``.

        The body is capped at ``max_content_bytes`` (``truncated`` is True if the cap was hit),
        then stripped to visible text via :func:`html_to_text`.
        """
        resp = await self._get_with_retry(url)
        content = resp.content
        truncated = len(content) > self.max_content_bytes
        if truncated:
            content = content[: self.max_content_bytes]
        html = content.decode(resp.encoding or "utf-8", errors="replace")
        return html_to_text(html), truncated

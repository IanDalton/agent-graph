"""WebSearch capability: let the agent query the internet via SearXNG and read result pages.

Exposed via :func:`build_search`, dropped into ``Agent(capabilities=...)``. Two tools:

- ``web_search`` — run a SearXNG search, returning ranked title/URL/snippet results.
- ``fetch_url`` — download a chosen result page and return its readable text.

Both go through :class:`backend.web.client.WebClient`. The client is taken from
``ctx.deps.web`` when present (wired in ``main.run``), else a short-lived one is built from env
for that call — so the tools work standalone (tests/CLI) and injected (production) alike.

Safety contract (copied from ``graph_capability.run_query``): these tools must NEVER abort the
run. A SearXNG outage, HTTP error, or unreachable page is caught and returned as a structured
``error`` result the model can read and react to, not raised.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability

from backend.db.dependencies import GraphDependencies
from backend.schemas.search_schemas import (
    FetchPageResult,
    FetchUrlArgs,
    WebSearchArgs,
    WebSearchHit,
    WebSearchResult,
)
from backend.web.client import WebClient

logger = logging.getLogger("agent_graph.search")

INSTRUCTIONS = (
    "You can query the live internet. Use `web_search` for current or external information that "
    "the user's memory graph cannot answer (recent events, facts you are unsure of, anything "
    "beyond your training). It returns ranked results with a title, url and short snippet. "
    "When a snippet is not enough, call `fetch_url` with a result's url to read the full page "
    "text. Always keep and cite the source url for anything you report from the web. "
    "If a search or fetch returns an `error`, say so honestly and do not invent results. "
    "When you learn a durable fact worth remembering across conversations, consider saving it "
    "with store_fact (and cite the url).\n"
    "WRITING THE QUERY — search engines match keywords, not sentences. Compose the `query` like a "
    "search expert, not a chatbot:\n"
    "  - Use a few precise KEYWORDS, not a full natural-language question. Strip filler words "
    "('what is', 'how do I', 'please tell me'). Bad: 'what is the latest version of ArcadeDB?' "
    "Good: 'ArcadeDB latest version release'.\n"
    "  - Keep the most DISTINCTIVE, specific terms: proper nouns, product/library names, error "
    "codes, technical terms, and dates or version numbers. These narrow results fastest.\n"
    "  - Add qualifiers when they help: a year for recency ('2026'), 'documentation'/'changelog'/"
    "'tutorial' for the kind of source, or wrap an exact phrase in double quotes (\"...\") to "
    "require it verbatim.\n"
    "  - For time-sensitive topics, set `time_range` ('day'/'week'/'month'/'year') instead of "
    "stuffing 'latest'/'recent' into the keywords.\n"
    "  - Prefer SEVERAL focused searches over one long query. If results are weak or off-topic, "
    "reformulate: swap in synonyms, make terms more specific, or split a multi-part question into "
    "separate searches — don't just re-run the same words."
)

web_capability = Capability(id="WebSearch", instructions=INSTRUCTIONS)


@asynccontextmanager
async def _client_for(deps: GraphDependencies) -> AsyncIterator[WebClient]:
    """Yield the run-scoped WebClient if injected, else a short-lived one built from env."""
    if deps.web is not None:
        yield deps.web
    else:
        async with WebClient() as client:
            yield client


@web_capability.tool
async def web_search(ctx: RunContext[GraphDependencies], args: WebSearchArgs) -> WebSearchResult:
    """Search the web via SearXNG. Returns ranked results (title, url, snippet)."""
    try:
        async with _client_for(ctx.deps) as client:
            rows = await client.search(
                args.query,
                max_results=args.max_results,
                categories=args.categories,
                time_range=args.time_range,
            )
    except Exception as exc:  # noqa: BLE001 — never abort the run on a search failure.
        logger.warning("web_search failed for %r: %s", args.query, exc, exc_info=True)
        return WebSearchResult(query=args.query, error=f"Search failed: {exc}")
    hits = [
        WebSearchHit(
            title=str(r.get("title") or ""),
            url=str(r.get("url") or ""),
            snippet=str(r.get("content") or ""),
            engine=(str(r["engine"]) if r.get("engine") else None),
        )
        for r in rows
    ]
    return WebSearchResult(query=args.query, hits=hits)


@web_capability.tool
async def fetch_url(ctx: RunContext[GraphDependencies], args: FetchUrlArgs) -> FetchPageResult:
    """Download a page (an http(s) url, e.g. a web_search result) and return its readable text."""
    try:
        async with _client_for(ctx.deps) as client:
            text, truncated = await client.fetch(args.url)
    except Exception as exc:  # noqa: BLE001 — never abort the run on a fetch failure.
        logger.warning("fetch_url failed for %r: %s", args.url, exc, exc_info=True)
        return FetchPageResult(url=args.url, error=f"Fetch failed: {exc}")
    return FetchPageResult(url=args.url, text=text, truncated=truncated)


def build_search() -> list[Capability]:
    """Return the web-search capability to add to ``Agent(capabilities=...)``.

    The WebClient is supplied per-run through ``GraphDependencies.web`` (or built on demand),
    so nothing needs to be wired in here.
    """
    return [web_capability]

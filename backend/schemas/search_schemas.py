"""Pydantic models for the web-search capability's tool inputs/outputs.

Kept separate from :mod:`backend.schemas.graph_schemas` (the memory/ontology I/O) for clarity.
The ``FetchUrlArgs`` scheme validator is this capability's safety boundary: it refuses anything
that isn't an ``http``/``https`` URL.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

# SearXNG accepts these relative windows for the ``time_range`` filter.
_ALLOWED_TIME_RANGES = frozenset({"day", "week", "month", "year"})


class WebSearchArgs(BaseModel):
    """A web search the agent wants to run via SearXNG."""

    query: str = Field(
        ...,
        min_length=1,
        description=(
            "A keyword search query — concise, distinctive terms, NOT a full sentence/question. "
            "Drop filler words; keep proper nouns, technical terms and version numbers. "
            "E.g. 'ArcadeDB latest version release', not 'what is the newest ArcadeDB?'. "
            "Use double quotes for an exact phrase; prefer time_range over the word 'latest'."
        ),
    )
    max_results: int = Field(
        5, ge=1, le=10, description="Maximum number of results to return (1-10)."
    )
    time_range: str | None = Field(
        None,
        description="Optional recency filter: one of 'day', 'week', 'month', 'year'.",
    )
    categories: str | None = Field(
        None,
        description="Optional SearXNG category filter (e.g. 'general', 'news', 'science').",
    )

    @field_validator("time_range")
    @classmethod
    def _valid_time_range(cls, v: str | None) -> str | None:
        if v is None:
            return None
        low = v.strip().lower()
        if low not in _ALLOWED_TIME_RANGES:
            raise ValueError(f"time_range must be one of {sorted(_ALLOWED_TIME_RANGES)}.")
        return low


class WebSearchHit(BaseModel):
    """A single search result."""

    title: str = ""
    url: str = ""
    snippet: str = Field("", description="The result's summary/snippet text.")
    engine: str | None = Field(None, description="Which engine returned this result.")


class WebSearchResult(BaseModel):
    """Structured result returned by the web_search tool."""

    query: str
    hits: list[WebSearchHit] = Field(default_factory=list)
    error: str | None = Field(
        None,
        description="Set when the search could not be completed (e.g. SearXNG unreachable); "
        "hits will be empty.",
    )


class FetchUrlArgs(BaseModel):
    """A page the agent wants to download and read (typically a web_search result URL)."""

    url: str = Field(..., description="An http(s) URL to fetch, e.g. a web_search result's url.")

    @field_validator("url")
    @classmethod
    def _http_only(cls, v: str) -> str:
        scheme = urlparse(v.strip()).scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError("url must be an http or https URL.")
        return v.strip()


class FetchPageResult(BaseModel):
    """Structured result returned by the fetch_url tool."""

    url: str
    text: str = Field("", description="The page's readable text (HTML stripped).")
    truncated: bool = Field(False, description="True if the page was longer than the byte cap.")
    error: str | None = Field(
        None,
        description="Set when the page could not be fetched (e.g. 404, unreachable); "
        "text will be empty.",
    )

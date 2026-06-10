"""Async embedding client: turn text into a vector for semantic memory search.

Env-driven and **tolerant**, modeled on :class:`backend.web.client.WebClient` and
:class:`backend.db.arcade_db.ArcadeClient` (single reused ``httpx`` client, async context manager,
capped exponential-backoff retries). :meth:`Embedder.embed` NEVER raises — any failure (provider
down, bad response, no model configured) is logged and returns ``None``, so semantic recall simply
degrades to the existing ``LIKE`` substring search rather than aborting the run (same contract as
``run_query`` and the web tools).

Configuration (all via env, with ``.env`` loaded):

- ``EMBED_MODEL``   — embedding model name. **Unset ⇒ embeddings disabled** (``embed`` returns
  ``None`` and no schema/index is created); set it to turn semantic search on.
- ``EMBED_PROVIDER``— ``openai`` or ``ollama``. Default inferred: ``ollama`` (the project's local
  fallback) unless ``OPENAI_API_KEY``/``EMBED_API_KEY`` is set.
- ``EMBED_BASE_URL``— API base. Default per provider (OpenAI ``https://api.openai.com/v1``, Ollama
  ``http://localhost:11434``).
- ``EMBED_API_KEY`` — bearer token for OpenAI-style providers (falls back to ``OPENAI_API_KEY``).
- ``EMBED_DIM``     — vector dimension. Used by :func:`backend.db.arcade_db.ensure_schema` to size
  the ``LSM_VECTOR`` index; must match the model's output (e.g. 1536 for
  ``text-embedding-3-small``, 768 for ``nomic-embed-text``).
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agent_graph.embeddings")

_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
_OLLAMA_DEFAULT_BASE = "http://localhost:11434"


def embeddings_enabled() -> bool:
    """True when an embedding model is configured (``EMBED_MODEL`` set)."""
    return bool(os.getenv("EMBED_MODEL"))


def embedding_dimension() -> int | None:
    """The configured embedding dimension (``EMBED_DIM``), or ``None`` if unset/invalid.

    Read by :func:`backend.db.arcade_db.ensure_schema` to decide whether to create the vector
    property + ``LSM_VECTOR`` index and how to size it. ``None`` ⇒ no vector schema is created.
    """
    raw = os.getenv("EMBED_DIM")
    if not raw:
        return None
    try:
        dim = int(raw)
    except ValueError:
        logger.warning("EMBED_DIM=%r is not an integer; ignoring (vector index disabled)", raw)
        return None
    return dim if dim > 0 else None


def _infer_provider() -> str:
    provider = os.getenv("EMBED_PROVIDER")
    if provider:
        return provider.strip().lower()
    # Default to OpenAI when an API key is present, else the project's local Ollama fallback.
    if os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


class Embedder:
    """Async text → vector embedder. Disabled (``embed`` returns ``None``) when no model is set."""

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.1,
        retry_max_delay: float = 2.0,
    ) -> None:
        self.model = model if model is not None else os.getenv("EMBED_MODEL")
        self.provider = (provider or _infer_provider()).lower()
        default_base = _OPENAI_DEFAULT_BASE if self.provider == "openai" else _OLLAMA_DEFAULT_BASE
        self.base_url = (base_url or os.getenv("EMBED_BASE_URL", default_base)).rstrip("/")
        self.api_key = api_key or os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None  # lazy: a disabled embedder opens nothing.

    @classmethod
    def from_env(cls) -> "Embedder":
        """Build an embedder from the environment (inert when ``EMBED_MODEL`` is unset)."""
        return cls()

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    async def __aenter__(self) -> "Embedder":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)
        return self._client

    def _backoff(self, attempt: int) -> float:
        return min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)

    async def _post(self, path: str, body: dict[str, object]) -> httpx.Response:
        """POST ``body`` to ``{base_url}{path}``, retrying transient failures with capped backoff."""
        client = self._ensure_client()
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await client.post(url, json=body)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                break
            if resp.status_code >= 500 and attempt < self.max_retries:
                last_exc = httpx.HTTPStatusError(
                    f"{resp.status_code} from {url}", request=resp.request, response=resp
                )
                await asyncio.sleep(self._backoff(attempt))
                continue
            resp.raise_for_status()
            return resp
        assert last_exc is not None
        raise last_exc

    async def embed(self, text: str) -> list[float] | None:
        """Embed ``text`` into a vector, or return ``None`` if disabled or on any failure.

        Tolerant by contract: callers fall back to substring search when this returns ``None``, so
        a missing model / provider outage degrades recall instead of crashing the agent run.
        """
        if not self.enabled or not text:
            return None
        try:
            if self.provider == "ollama":
                resp = await self._post("/api/embeddings", {"model": self.model, "prompt": text})
                vector = resp.json().get("embedding")
            else:  # openai-compatible
                resp = await self._post("/embeddings", {"model": self.model, "input": text})
                data = resp.json().get("data") or []
                vector = data[0].get("embedding") if data else None
            if not isinstance(vector, list) or not vector:
                logger.warning("embedder returned no vector for input; falling back to LIKE search")
                return None
            return [float(x) for x in vector]
        except Exception:  # noqa: BLE001 — embeddings are best-effort; never abort the run.
            logger.warning("embedding failed; falling back to LIKE search", exc_info=True)
            return None

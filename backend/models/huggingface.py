"""HuggingFace GGUF model discovery + download for the local llama.cpp provider.

Modeled on :mod:`backend.marketplace` and :class:`backend.web.client.WebClient`: an env-driven
``httpx`` wrapper, one reused client, async context manager, capped-backoff retry. Used by the Model
Manager API to let the user search HuggingFace for GGUF models, inspect a repo's quantizations +
sizes, read a model's ``config.json`` (for precise VRAM/KV-cache estimates), and stream-download a
chosen ``.gguf`` to the shared models directory.

**Tolerant contract** (same as ``web_search``/``run_query``): the module-level :func:`search`,
:func:`list_files` and :func:`fetch_config` NEVER raise — a network or API hiccup degrades to an
empty/``None`` result. :meth:`HuggingFaceClient.stream_download` *does* propagate errors, because its
single caller (the SSE download handler) frames them as an ``error`` event — mirroring
``api.chat_stream``.

An optional ``HF_TOKEN`` env lifts rate limits and reaches gated/private repos.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agent_graph.hf_models")

HF_API = "https://huggingface.co/api"
HF_BASE = "https://huggingface.co"
_USER_AGENT = "agent-graph/1.0 (+https://github.com/agent-graph) hf-models"
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB
_PROGRESS_INTERVAL = 0.25  # seconds between progress emissions (plus a guaranteed final one)

# Quantization tags as they appear in GGUF filenames, longest/most-specific first so e.g. ``Q4_K_M``
# wins over a bare ``Q4``. Covers k-quants, i-quants, legacy ``Q4_0`` and full-precision tags.
_QUANT_RE = re.compile(
    r"(IQ\d+_[A-Z]+|Q\d+_K_[A-Z]+|Q\d+_K|Q\d+_\d+|Q\d+|BF16|F16|F32)",
    re.IGNORECASE,
)
# Sharded GGUFs: ``model-00001-of-00003.gguf``. We group shards into one logical model.
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

# In-process cache for searches so re-opening the Discover tab / re-typing the same query is instant.
_SEARCH_TTL_SECONDS = 900
_search_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def parse_quant(filename: str) -> str:
    """Extract the quantization label from a GGUF filename (e.g. ``Q4_K_M``), uppercased, or ``""``."""
    match = _QUANT_RE.search(filename)
    return match.group(1).upper() if match else ""


def _shard_group_key(filename: str) -> str:
    """Map a GGUF filename to a shard-independent key so the N shards of one model group together."""
    return _SHARD_RE.sub(".gguf", filename)


class HuggingFaceClient:
    """Fetches GGUF model metadata and files from the HuggingFace Hub HTTP API."""

    def __init__(
        self,
        token: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.2,
        retry_max_delay: float = 2.0,
    ) -> None:
        self.token = token if token is not None else os.getenv("HF_TOKEN")
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        # A generous read timeout: big GGUFs stream for a long time, but a stalled socket should still
        # eventually error rather than hang forever.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=timeout, read=120.0, write=timeout, pool=timeout),
            follow_redirects=True,
            headers=headers,
        )

    async def __aenter__(self) -> "HuggingFaceClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _backoff(self, attempt: int) -> float:
        return min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)

    async def _get_with_retry(self, url: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET ``url``, retrying transport errors / 5xx with capped backoff; raises a final 4xx/5xx."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "hf transport error on %s (attempt %d/%d): %s; retrying",
                        url, attempt + 1, self.max_retries + 1, exc,
                    )
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

    async def search_gguf(
        self, query: str, *, limit: int = 30, sort: str = "downloads"
    ) -> list[dict[str, Any]]:
        """Search the Hub for GGUF models matching ``query``, newest/most-downloaded first.

        Returns ``[{repo_id, downloads, likes, last_modified, gated}]``. ``sort`` is one of
        ``downloads``/``likes``/``lastModified``/``createdAt``/``trendingScore``. An empty ``query``
        omits the search term, so the result is a pure browse list ranked by ``sort`` (e.g. the most
        popular or most recent GGUF models) — that's what the Discover tab shows before you search.
        """
        params: dict[str, Any] = {
            "filter": "gguf",
            "sort": sort,
            "direction": "-1",
            "limit": str(limit),
        }
        if query.strip():
            params["search"] = query.strip()
        resp = await self._get_with_retry(f"{HF_API}/models", params=params)
        items = resp.json() or []
        out: list[dict[str, Any]] = []
        for item in items:
            repo_id = item.get("id") or item.get("modelId")
            if not repo_id:
                continue
            out.append(
                {
                    "repo_id": repo_id,
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "last_modified": item.get("lastModified") or item.get("last_modified") or "",
                    "gated": bool(item.get("gated")),
                }
            )
        return out

    async def list_gguf_files(self, repo_id: str, revision: str = "main") -> list[dict[str, Any]]:
        """List a repo's GGUF files (shards grouped), with byte sizes and quant labels.

        Uses the git-tree API (``?recursive=true``). GGUFs are Git-LFS, so the true size is in
        ``entry["lfs"]["size"]`` (the plain ``size`` is the LFS pointer). Sharded models
        (``…-00001-of-000NN.gguf``) collapse to one entry whose ``path`` is the first shard
        (llama-server loads the rest automatically) and whose ``size_bytes`` sums the shards.
        """
        resp = await self._get_with_retry(
            f"{HF_API}/models/{repo_id}/tree/{revision}", params={"recursive": "true"}
        )
        tree = resp.json() or []
        groups: dict[str, dict[str, Any]] = {}
        for entry in tree:
            if entry.get("type") != "file":
                continue
            path = entry.get("path") or ""
            if not path.lower().endswith(".gguf"):
                continue
            size = int((entry.get("lfs") or {}).get("size") or entry.get("size") or 0)
            filename = path.rsplit("/", 1)[-1]
            key = _shard_group_key(path)
            shard = _SHARD_RE.search(filename)
            group = groups.get(key)
            if group is None:
                group = {
                    "path": path,
                    "filename": filename,
                    "quant": parse_quant(filename),
                    "size_bytes": 0,
                    "shards": 0,
                }
                groups[key] = group
            group["size_bytes"] += size
            group["shards"] += 1
            # The first shard (…-00001-of-…) is the canonical path llama-server is given.
            if shard and shard.group(1) == "00001":
                group["path"] = path
                group["filename"] = filename
        return sorted(groups.values(), key=lambda g: g["size_bytes"])

    async def fetch_model_config(self, repo_id: str, revision: str = "main") -> dict[str, Any] | None:
        """Fetch the repo's ``config.json`` (architecture params for precise KV-cache math), or ``None``.

        Tolerant: a missing config (a pure-GGUF repo with no HF config) or any error returns ``None``,
        and the recommender falls back to a size-class heuristic.
        """
        try:
            resp = await self._get_with_retry(f"{HF_BASE}/{repo_id}/resolve/{revision}/config.json")
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001 — no config is the normal case for many GGUF repos.
            logger.debug("no config.json for %s@%s", repo_id, revision, exc_info=True)
            return None

    async def stream_download(
        self,
        repo_id: str,
        file_path: str,
        dest_dir: str | Path,
        *,
        revision: str = "main",
        progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> Path:
        """Stream-download one GGUF to ``dest_dir`` with byte progress; resumable; atomic on finish.

        Writes to a ``<file>.part`` sibling and ``os.replace``s it onto the final name only when the
        whole file has arrived, so an interrupted download never looks complete. If a ``.part`` exists
        it resumes via a ``Range`` request. ``progress(downloaded, total)`` is awaited at most every
        ``_PROGRESS_INTERVAL`` seconds (plus a guaranteed final call). Raises on failure (the SSE
        handler frames it).
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = file_path.rsplit("/", 1)[-1]
        dest = dest_dir / filename
        part = dest.with_name(dest.name + ".part")
        existing = part.stat().st_size if part.exists() else 0
        url = f"{HF_BASE}/{repo_id}/resolve/{revision}/{file_path}"
        headers = {"Range": f"bytes={existing}-"} if existing else {}

        async with self._client.stream("GET", url, headers=headers) as resp:
            if resp.status_code == 416:  # Requested range not satisfiable → the .part is already whole.
                await resp.aread()
                os.replace(part, dest)
                if progress:
                    size = dest.stat().st_size
                    await progress(size, size)
                return dest
            resp.raise_for_status()
            resuming = resp.status_code == 206
            downloaded = existing if resuming else 0
            total = _content_total(resp, downloaded)
            # Free-space guard before committing to a multi-GB write.
            remaining = max(0, (total or 0) - downloaded)
            free = shutil.disk_usage(dest_dir).free
            if total and remaining + (64 << 20) > free:  # keep a 64 MiB margin
                raise RuntimeError(
                    f"Not enough free space for {filename}: need ~{remaining // (1 << 20)} MiB, "
                    f"have {free // (1 << 20)} MiB free in {dest_dir}."
                )
            mode = "ab" if resuming and existing else "wb"
            if mode == "wb":
                downloaded = 0
            last_emit = 0.0
            with open(part, mode) as fh:
                if progress:
                    await progress(downloaded, total)
                async for chunk in resp.aiter_bytes(_DOWNLOAD_CHUNK):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if progress and (now - last_emit) >= _PROGRESS_INTERVAL:
                        last_emit = now
                        await progress(downloaded, total)
            if progress:
                await progress(downloaded, total or downloaded)
        os.replace(part, dest)
        return dest


def _content_total(resp: httpx.Response, downloaded: int) -> int:
    """Total byte size of the download from a streamed response (handles 206 ``Content-Range``)."""
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        try:
            return int(cr.rsplit("/", 1)[-1])
        except ValueError:
            pass
    cl = resp.headers.get("Content-Length")
    if cl:
        try:
            return downloaded + int(cl)
        except ValueError:
            pass
    return 0


# ---- Module-level tolerant helpers (build a client from env; never raise) -------------------------


async def search(
    query: str, *, limit: int = 30, sort: str = "downloads", client: HuggingFaceClient | None = None
) -> list[dict[str, Any]]:
    """Search GGUF models. Tolerant → ``[]`` on any failure. Cached in-process (15 min TTL)."""
    key = f"{query}|{sort}|{limit}"
    cached = _search_cache.get(key)
    if cached is not None and (time.time() - cached[0]) < _SEARCH_TTL_SECONDS:
        return cached[1]
    own = client is None
    hc = client or HuggingFaceClient()
    try:
        results = await hc.search_gguf(query, limit=limit, sort=sort)
        _search_cache[key] = (time.time(), results)
        return results
    except Exception:  # noqa: BLE001 — a search hiccup must not break the UI.
        logger.warning("hf search failed for %r", query, exc_info=True)
        return []
    finally:
        if own:
            await hc.aclose()


async def list_files(
    repo_id: str, revision: str = "main", *, client: HuggingFaceClient | None = None
) -> list[dict[str, Any]]:
    """List a repo's GGUF files. Tolerant → ``[]`` on any failure."""
    own = client is None
    hc = client or HuggingFaceClient()
    try:
        return await hc.list_gguf_files(repo_id, revision)
    except Exception:  # noqa: BLE001
        logger.warning("hf list_files failed for %r", repo_id, exc_info=True)
        return []
    finally:
        if own:
            await hc.aclose()


async def fetch_config(
    repo_id: str, revision: str = "main", *, client: HuggingFaceClient | None = None
) -> dict[str, Any] | None:
    """Fetch a repo's ``config.json``. Tolerant → ``None``."""
    own = client is None
    hc = client or HuggingFaceClient()
    try:
        return await hc.fetch_model_config(repo_id, revision)
    finally:
        if own:
            await hc.aclose()


__all__ = [
    "HuggingFaceClient",
    "search",
    "list_files",
    "fetch_config",
    "parse_quant",
]

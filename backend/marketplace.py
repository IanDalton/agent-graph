"""Sync skills from the Anthropic Agent Skills marketplace into a user's database.

A *skill* is one folder in the public ``anthropics/skills`` repo: a ``SKILL.md`` (YAML frontmatter
``name``/``description`` + a markdown instructions body) plus optional bundled ``scripts/``,
``references/`` and ``assets/`` files. This module fetches them from GitHub and stores each as a
``Skill`` vertex (see :func:`backend.db.repository.upsert_skill`), so the agent can later inject a
skill's description (progressive disclosure), load its body on demand, and mount its files into the
``run_python`` sandbox.

Fetching is cheap on the GitHub API: ONE call to the git-trees API lists the whole repo tree, then
each file is pulled from ``raw.githubusercontent.com`` (which does *not* count against the API rate
limit). An optional ``GITHUB_TOKEN`` lifts the unauthenticated 60 req/hr limit. Modeled on
:class:`backend.web.client.WebClient`: env-driven config, one reused client, async context manager,
capped-backoff retry on transient failures.

The sync is **tolerant**: a per-skill failure is collected into the returned summary, never raised,
so one bad skill (or a rate-limit hiccup) can't abort the whole sync.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

from backend.db import repository as repo
from backend.db.arcade_db import ArcadeClient

load_dotenv()

logger = logging.getLogger("agent_graph.marketplace")

DEFAULT_REPO = "anthropics/skills"
DEFAULT_REF = "main"
# The repo subdirectory that holds the skill folders (each is one Agent Skill).
_SKILLS_PREFIX = "skills/"

# Caps so a pathological skill can't blow up memory/storage. Mirror the sandbox's per-file cap.
MAX_FILES_PER_SKILL = 50
MAX_FILE_BYTES = 5_000_000
MAX_TOTAL_BYTES_PER_SKILL = 25_000_000

_USER_AGENT = (
    "agent-graph/1.0 (+https://github.com/agent-graph) skills-sync"
)

# In-process cache for the catalog (name + description) so reopening the marketplace dialog doesn't
# re-fetch ~17 SKILL.md files every time. Keyed by ``repo@ref``; skills change rarely, so a 15-minute
# TTL is plenty. NB: only the catalog metadata is cached — the per-user ``installed`` flag is merged
# fresh on each API request (see api.skills_catalog), so an install shows up immediately.
_CATALOG_TTL_SECONDS = 900
_catalog_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into ``(frontmatter_dict, body)``.

    Recognizes the standard ``---``-delimited YAML block at the very top of the file. When there is
    no frontmatter (or it doesn't parse), returns ``({}, full_text)`` so the caller still gets the
    body. Pure and network-free, so it is unit-testable in isolation.
    """
    stripped = text.lstrip("﻿")  # tolerate a leading BOM
    if not stripped.startswith("---"):
        return {}, text
    # Find the closing fence on its own line after the opening one.
    rest = stripped[3:]
    # The opening fence may be "---\n"; the closing fence is "\n---" at a line start.
    end = rest.find("\n---")
    if end == -1:
        return {}, text
    block = rest[:end]
    body = rest[end + 4 :]
    body = body.lstrip("\n")
    try:
        meta = yaml.safe_load(block)
    except yaml.YAMLError:
        logger.warning("failed to parse SKILL.md frontmatter; treating as bodyless", exc_info=True)
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, body


class MarketplaceClient:
    """Fetches the skills catalog and individual skills from a GitHub repo."""

    def __init__(
        self,
        repo_slug: str | None = None,
        ref: str | None = None,
        token: str | None = None,
        *,
        timeout: float = 20.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.2,
        retry_max_delay: float = 2.0,
    ) -> None:
        self.repo_slug = repo_slug or os.getenv("SKILLS_REPO", DEFAULT_REPO)
        self.ref = ref or os.getenv("SKILLS_REF", DEFAULT_REF)
        self.token = token if token is not None else os.getenv("GITHUB_TOKEN")
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers)

    async def __aenter__(self) -> "MarketplaceClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _backoff(self, attempt: int) -> float:
        return min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)

    async def _get_with_retry(self, url: str) -> httpx.Response:
        """GET ``url``, retrying transport errors / 5xx with capped backoff; raises a final 4xx/5xx."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.get(url)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "marketplace transport error on %s (attempt %d/%d): %s; retrying",
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

    def _raw_url(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repo_slug}/{self.ref}/{path}"

    async def list_catalog(self) -> dict[str, list[str]]:
        """Return ``{skill_name: [repo file paths]}`` for every skill folder in the repo.

        One git-trees API call lists the whole tree (recursive); we filter to ``skills/<name>/...``
        blobs and group by skill name. Cheap on the rate limit (a single API request).
        """
        url = f"https://api.github.com/repos/{self.repo_slug}/git/trees/{self.ref}?recursive=1"
        resp = await self._get_with_retry(url)
        tree = resp.json().get("tree", []) or []
        catalog: dict[str, list[str]] = {}
        for entry in tree:
            if entry.get("type") != "blob":
                continue
            path = entry.get("path") or ""
            if not path.startswith(_SKILLS_PREFIX):
                continue
            rest = path[len(_SKILLS_PREFIX) :]
            name, _, sub = rest.partition("/")
            if not name or not sub:  # need a file under a skill folder, not the folder itself
                continue
            catalog.setdefault(name, []).append(path)
        return catalog

    async def fetch_skill_meta(self, name: str) -> dict[str, str]:
        """Fetch ONLY a skill's frontmatter (name + description) — cheap, for the catalog listing.

        Downloads `skills/<name>/SKILL.md` and parses its frontmatter; does not fetch the bundled
        files (that's `fetch_skill`, used at install time). Raises on a missing/unreadable SKILL.md
        (the caller degrades that to an empty description).
        """
        resp = await self._get_with_retry(self._raw_url(f"{_SKILLS_PREFIX}{name}/SKILL.md"))
        meta, _ = _parse_frontmatter(resp.text)
        return {"name": name, "description": str(meta.get("description") or "").strip()}

    async def fetch_skill(self, name: str, paths: list[str]) -> dict[str, Any]:
        """Fetch one skill: parse its SKILL.md and download its bundled files (capped).

        Returns ``{name, description, body, files}`` where ``files`` is a
        ``relpath -> {content, encoding}`` map (encoding ``"text"`` or ``"base64"``). Raises if the
        skill has no readable SKILL.md (the caller treats that as a per-skill error).
        """
        skill_md_path = f"{_SKILLS_PREFIX}{name}/SKILL.md"
        resp = await self._get_with_retry(self._raw_url(skill_md_path))
        meta, body = _parse_frontmatter(resp.text)
        description = str(meta.get("description") or "").strip()

        files: dict[str, dict[str, str]] = {}
        total = 0
        # Bundled files = everything under the skill folder except SKILL.md itself (that's the body).
        bundled = sorted(p for p in paths if p != skill_md_path)
        for path in bundled:
            if len(files) >= MAX_FILES_PER_SKILL:
                logger.info("skill %r: capped at %d files; rest skipped", name, MAX_FILES_PER_SKILL)
                break
            relpath = path[len(f"{_SKILLS_PREFIX}{name}/") :]
            try:
                file_resp = await self._get_with_retry(self._raw_url(path))
            except httpx.HTTPError:
                logger.warning("skill %r: could not fetch %s; skipping", name, path, exc_info=True)
                continue
            data = file_resp.content
            if len(data) > MAX_FILE_BYTES or total + len(data) > MAX_TOTAL_BYTES_PER_SKILL:
                logger.info("skill %r: file %s exceeds size caps; skipped", name, relpath)
                continue
            total += len(data)
            files[relpath] = _encode_file(data)
        return {"name": name, "description": description, "body": body, "files": files}


def _encode_file(data: bytes) -> dict[str, str]:
    """Encode raw bytes as ``{content, encoding}`` — literal text when it decodes as UTF-8, else base64."""
    try:
        return {"content": data.decode("utf-8"), "encoding": "text"}
    except UnicodeDecodeError:
        return {"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"}


async def sync(
    db: ArcadeClient,
    user_id: str,
    names: list[str] | None = None,
    client: MarketplaceClient | None = None,
) -> dict[str, Any]:
    """Sync skills from the marketplace into ``user_id``'s database. Tolerant.

    ``names`` restricts the sync to specific skills (default: the whole catalog). Returns
    ``{"synced": [...names], "errors": [{"name", "error"}], "source": "<repo>@<ref>"}``. A failure to
    list the catalog returns an error summary; a per-skill failure is collected and the rest proceed.
    ``client`` is a test seam — production builds one from env.
    """
    own_client = client is None
    mc = client or MarketplaceClient()
    source = f"{mc.repo_slug}@{mc.ref}"
    synced: list[str] = []
    errors: list[dict[str, str]] = []
    try:
        try:
            catalog = await mc.list_catalog()
        except Exception as exc:  # noqa: BLE001 — surface as a summary error, never raise.
            logger.warning("marketplace catalog listing failed", exc_info=True)
            return {"synced": [], "errors": [{"name": "*", "error": str(exc)}], "source": source}

        wanted = catalog if names is None else {n: catalog.get(n, []) for n in names}
        for name, paths in wanted.items():
            if not paths:
                errors.append({"name": name, "error": "not found in catalog"})
                continue
            try:
                skill = await mc.fetch_skill(name, paths)
                await repo.upsert_skill(
                    db,
                    user_id,
                    name=skill["name"],
                    description=skill["description"],
                    body=skill["body"],
                    files=skill["files"],
                    source=source,
                )
                synced.append(name)
            except Exception as exc:  # noqa: BLE001 — one bad skill must not abort the sync.
                logger.warning("failed to sync skill %r", name, exc_info=True)
                errors.append({"name": name, "error": str(exc)})
    finally:
        if own_client:
            await mc.aclose()
    return {"synced": synced, "errors": errors, "source": source}


async def catalog(client: MarketplaceClient | None = None) -> list[dict[str, str]]:
    """Return the full marketplace catalog as ``[{"name", "description"}]``, sorted by name.

    Lists the skill names with one git-trees API call, then fetches each skill's frontmatter
    concurrently for its description. Cached in-process with a short TTL so reopening the dialog is
    instant. **Tolerant per skill**: a skill whose meta fetch fails gets an empty description rather
    than dropping out. A failure to list the catalog raises — the API handler maps it to ``[]``.
    ``client`` is a test seam; production builds one from env.
    """
    own_client = client is None
    mc = client or MarketplaceClient()
    key = f"{mc.repo_slug}@{mc.ref}"
    cached = _catalog_cache.get(key)
    if cached is not None and (time.time() - cached[0]) < _CATALOG_TTL_SECONDS:
        if own_client:
            await mc.aclose()
        return cached[1]
    try:
        names = sorted((await mc.list_catalog()).keys())

        async def _meta(name: str) -> dict[str, str]:
            try:
                return await mc.fetch_skill_meta(name)
            except Exception:  # noqa: BLE001 — a bad skill gets an empty description, not dropped.
                logger.warning("failed to fetch catalog meta for skill %r", name, exc_info=True)
                return {"name": name, "description": ""}

        items = list(await asyncio.gather(*(_meta(n) for n in names)))
        _catalog_cache[key] = (time.time(), items)
        return items
    finally:
        if own_client:
            await mc.aclose()


__all__ = ["MarketplaceClient", "sync", "catalog", "_parse_frontmatter"]

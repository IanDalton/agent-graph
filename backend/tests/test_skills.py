"""Tests for the marketplace-skills feature.

All unit tests run without a DB, network, or Docker: the GitHub marketplace client is faked, repo
functions are monkeypatched or hit a duck-typed FakeDb, and the sandbox is a duck-typed stand-in.
They cover the four moving parts — frontmatter parsing, the tolerant sync, the load_skill tool +
per-turn description block, the sandbox /skills mount — plus the capability wiring.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from backend import marketplace
from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.main import _capabilities_for_mode
from backend.marketplace import _parse_frontmatter
from backend.sandbox.runner import DEFAULT_IMAGE, PythonSandbox, SandboxResult
from backend.schemas.sandbox_schemas import RunPythonArgs
from backend.schemas.skill_schemas import LoadSkillArgs
from backend.skills.sandbox_capability import run_python
from backend.skills.skill_capability import build_skills, load_skill, skill_use_frame
from backend.skills.system_prompt import available_skills_block, enabled_skills_block


class FakeDb:
    """Duck-typed ArcadeClient that records commands and returns no rows."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, dict[str, Any]]] = []

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.commands.append((sql, params or {}))
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return []


def _ctx(deps: GraphDependencies) -> RunContext[GraphDependencies]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


# --------------------------------------------------------------------------- #
# Frontmatter parsing (pure)
# --------------------------------------------------------------------------- #
def test_parse_frontmatter_extracts_meta_and_body() -> None:
    text = "---\nname: pdf\ndescription: Make and read PDFs\n---\n# How to\nDo the thing.\n"
    meta, body = _parse_frontmatter(text)
    assert meta["name"] == "pdf"
    assert meta["description"] == "Make and read PDFs"
    assert body.startswith("# How to")


def test_parse_frontmatter_handles_no_frontmatter() -> None:
    meta, body = _parse_frontmatter("# Just a body, no fence")
    assert meta == {}
    assert body == "# Just a body, no fence"


def test_parse_frontmatter_tolerates_bad_yaml() -> None:
    meta, body = _parse_frontmatter("---\nname: : : bad\n: x\n---\nbody")
    # Malformed YAML degrades to bodyless rather than raising.
    assert meta == {}
    assert "body" in body


# --------------------------------------------------------------------------- #
# repo helpers
# --------------------------------------------------------------------------- #
def test_parse_skill_names_normalizes() -> None:
    assert repo._parse_skill_names('["a", "b"]') == ["a", "b"]
    assert repo._parse_skill_names(["a", "b"]) == ["a", "b"]
    assert repo._parse_skill_names(None) == []
    assert repo._parse_skill_names("") == []
    assert repo._parse_skill_names("not json") == []


# --------------------------------------------------------------------------- #
# marketplace.sync (faked GitHub client; tolerant)
# --------------------------------------------------------------------------- #
class FakeMarketplaceClient:
    """Duck-typed MarketplaceClient: canned catalog/skills, optional per-skill failures."""

    repo_slug = "anthropics/skills"
    ref = "main"

    def __init__(
        self,
        catalog: dict[str, list[str]],
        skills: dict[str, dict[str, Any]],
        fail: tuple[str, ...] = (),
        catalog_error: Exception | None = None,
        meta_fail: tuple[str, ...] = (),
    ) -> None:
        self._catalog = catalog
        self._skills = skills
        self._fail = set(fail)
        self._catalog_error = catalog_error
        self._meta_fail = set(meta_fail)
        self.list_calls = 0

    async def list_catalog(self) -> dict[str, list[str]]:
        self.list_calls += 1
        if self._catalog_error:
            raise self._catalog_error
        return self._catalog

    async def fetch_skill_meta(self, name: str) -> dict[str, str]:
        if name in self._meta_fail:
            raise RuntimeError("meta boom")
        return {"name": name, "description": str(self._skills.get(name, {}).get("description") or "")}

    async def fetch_skill(self, name: str, paths: list[str]) -> dict[str, Any]:
        if name in self._fail:
            raise RuntimeError("fetch boom")
        return self._skills[name]

    async def aclose(self) -> None:  # pragma: no cover - never called (client is injected)
        pass


def test_sync_upserts_each_skill_and_collects_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    async def fake_upsert(db: Any, user_id: str, **kw: Any) -> str:
        calls.append((user_id, kw["name"]))
        return "skill-id"

    monkeypatch.setattr(marketplace.repo, "upsert_skill", fake_upsert)
    client = FakeMarketplaceClient(
        catalog={"pdf": ["skills/pdf/SKILL.md"], "bad": ["skills/bad/SKILL.md"]},
        skills={"pdf": {"name": "pdf", "description": "d", "body": "b", "files": {}}},
        fail=("bad",),
    )
    res = asyncio.run(marketplace.sync(FakeDb(), "u", client=client))
    assert res["synced"] == ["pdf"]
    assert [e["name"] for e in res["errors"]] == ["bad"]
    assert res["source"] == "anthropics/skills@main"
    assert calls == [("u", "pdf")]


def test_sync_catalog_failure_returns_summary_error() -> None:
    client = FakeMarketplaceClient(catalog={}, skills={}, catalog_error=RuntimeError("github down"))
    res = asyncio.run(marketplace.sync(FakeDb(), "u", client=client))
    assert res["synced"] == []
    assert res["errors"][0]["name"] == "*"


def test_sync_unknown_requested_name_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_upsert(db: Any, user_id: str, **kw: Any) -> str:
        return "id"

    monkeypatch.setattr(marketplace.repo, "upsert_skill", fake_upsert)
    client = FakeMarketplaceClient(
        catalog={"pdf": ["skills/pdf/SKILL.md"]},
        skills={"pdf": {"name": "pdf", "description": "d", "body": "b", "files": {}}},
    )
    res = asyncio.run(marketplace.sync(FakeDb(), "u", names=["missing"], client=client))
    assert res["synced"] == []
    assert res["errors"][0]["name"] == "missing"


# --------------------------------------------------------------------------- #
# marketplace.catalog + fetch_skill_meta (live-browse path)
# --------------------------------------------------------------------------- #
def test_fetch_skill_meta_parses_name_and_description() -> None:
    async def main() -> None:
        mc = marketplace.MarketplaceClient(token=None)
        captured: dict[str, str] = {}

        async def fake_get(url: str) -> Any:
            captured["url"] = url

            class _Resp:
                text = "---\nname: pdf\ndescription: Make PDFs\n---\n# body"

            return _Resp()

        mc._get_with_retry = fake_get  # type: ignore[assignment]
        meta = await mc.fetch_skill_meta("pdf")
        await mc.aclose()
        assert meta == {"name": "pdf", "description": "Make PDFs"}
        assert captured["url"].endswith("skills/pdf/SKILL.md")

    asyncio.run(main())


def test_catalog_lists_names_and_descriptions() -> None:
    marketplace._catalog_cache.clear()
    client = FakeMarketplaceClient(
        catalog={"pdf": ["skills/pdf/SKILL.md"], "docx": ["skills/docx/SKILL.md"]},
        skills={
            "pdf": {"name": "pdf", "description": "Make PDFs"},
            "docx": {"name": "docx", "description": "Make Word docs"},
        },
    )
    items = asyncio.run(marketplace.catalog(client=client))
    # Sorted by name; each carries its description.
    assert items == [
        {"name": "docx", "description": "Make Word docs"},
        {"name": "pdf", "description": "Make PDFs"},
    ]


def test_catalog_tolerates_per_skill_meta_failure() -> None:
    marketplace._catalog_cache.clear()
    client = FakeMarketplaceClient(
        catalog={"pdf": ["x"], "bad": ["y"]},
        skills={"pdf": {"name": "pdf", "description": "Make PDFs"}},
        meta_fail=("bad",),
    )
    items = asyncio.run(marketplace.catalog(client=client))
    by_name = {i["name"]: i["description"] for i in items}
    assert by_name["pdf"] == "Make PDFs"
    assert by_name["bad"] == ""  # degraded to empty description, not dropped


def test_catalog_is_cached() -> None:
    marketplace._catalog_cache.clear()
    client = FakeMarketplaceClient(
        catalog={"pdf": ["x"]},
        skills={"pdf": {"name": "pdf", "description": "d"}},
    )
    asyncio.run(marketplace.catalog(client=client))
    asyncio.run(marketplace.catalog(client=client))
    assert client.list_calls == 1  # the second call is served from the in-process cache


# --------------------------------------------------------------------------- #
# load_skill tool
# --------------------------------------------------------------------------- #
def test_skills_tool_is_registered() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, deps_type=GraphDependencies, capabilities=[*build_skills()])
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", enabled_skills=["pdf"])
    asyncio.run(agent.run("hi", deps=deps))
    names = {t.name for t in model.last_model_request_parameters.function_tools}
    assert "load_skill" in names


def test_load_skill_returns_body_and_files(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_skill(db: Any, uid: str, ref: str) -> dict[str, Any]:
        return {
            "name": "pdf",
            "description": "Make PDFs",
            "body": "# How to make a PDF",
            "files": {"scripts/make.py": {"content": "print(1)", "encoding": "text"}},
        }

    monkeypatch.setattr("backend.skills.skill_capability.repo.get_skill", fake_get_skill)
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", enabled_skills=["pdf"])
    res = asyncio.run(load_skill(_ctx(deps), LoadSkillArgs(name="pdf")))
    assert res.instructions == "# How to make a PDF"
    assert res.files == ["scripts/make.py"]
    assert res.sandbox_path == "$SKILLS_DIR/pdf"


def test_load_skill_rejects_disabled_name() -> None:
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", enabled_skills=["pdf"])
    with pytest.raises(ModelRetry):
        asyncio.run(load_skill(_ctx(deps), LoadSkillArgs(name="docx")))


def test_load_skill_missing_content_raises_model_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_skill(db: Any, uid: str, ref: str) -> None:
        return None

    monkeypatch.setattr("backend.skills.skill_capability.repo.get_skill", fake_get_skill)
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", enabled_skills=["pdf"])
    with pytest.raises(ModelRetry):
        asyncio.run(load_skill(_ctx(deps), LoadSkillArgs(name="pdf")))


# --------------------------------------------------------------------------- #
# enabled_skills_block (system prompt injection)
# --------------------------------------------------------------------------- #
def test_enabled_skills_block_lists_enabled_only(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_list_skills(db: Any, uid: str, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {"name": "pdf", "description": "Make PDFs"},
            {"name": "docx", "description": "Make Word docs"},
        ]

    monkeypatch.setattr("backend.skills.system_prompt.repo.list_skills", fake_list_skills)
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", enabled_skills=["pdf"])
    block = asyncio.run(enabled_skills_block(deps))
    assert "pdf: Make PDFs" in block
    assert "docx" not in block


def test_enabled_skills_block_empty_when_none_enabled() -> None:
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", enabled_skills=[])
    assert asyncio.run(enabled_skills_block(deps)) == ""


def test_enabled_skills_block_tolerant_on_db_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(db: Any, uid: str, limit: int = 100) -> list[dict[str, Any]]:
        raise RuntimeError("db down")

    monkeypatch.setattr("backend.skills.system_prompt.repo.list_skills", boom)
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", enabled_skills=["pdf"])
    assert asyncio.run(enabled_skills_block(deps)) == ""


# --------------------------------------------------------------------------- #
# available_skills_block (swarm orchestrator: the whole library to assign from)
# --------------------------------------------------------------------------- #
def test_available_skills_block_lists_whole_library(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_list_skills(db: Any, uid: str, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {"name": "pdf", "description": "Make PDFs"},
            {"name": "docx", "description": "Make Word docs"},
        ]

    monkeypatch.setattr("backend.skills.system_prompt.repo.list_skills", fake_list_skills)
    # The orchestrator has no enabled_skills, but the block lists the library regardless.
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", enabled_skills=[])
    block = asyncio.run(available_skills_block(deps))
    assert "pdf: Make PDFs" in block and "docx: Make Word docs" in block


def test_available_skills_block_empty_library_is_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_list_skills(db: Any, uid: str, limit: int = 100) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr("backend.skills.system_prompt.repo.list_skills", fake_list_skills)
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c")
    assert asyncio.run(available_skills_block(deps)) == ""


# --------------------------------------------------------------------------- #
# skill_use_frame (the "Using skill X" notification)
# --------------------------------------------------------------------------- #
def test_skill_use_frame_flat_and_nested_args() -> None:
    flat = skill_use_frame("load_skill", {"name": "pdf"})
    assert flat == {"type": "skill", "action": "used", "skill_name": "pdf", "title": "pdf"}
    nested = skill_use_frame("load_skill", {"args": {"name": "docx"}})
    assert nested is not None and nested["skill_name"] == "docx"


def test_skill_use_frame_none_for_other_tools_or_empty_name() -> None:
    assert skill_use_frame("run_python", {"code": "x"}) is None
    assert skill_use_frame("load_skill", {"name": ""}) is None
    assert skill_use_frame("load_skill", "not-a-dict") is None


# --------------------------------------------------------------------------- #
# _capabilities_for_mode wiring
# --------------------------------------------------------------------------- #
def _capability_ids(caps: list[Any]) -> set[str | None]:
    return {getattr(c, "id", None) for c in caps}


def test_skills_capability_added_only_when_enabled() -> None:
    assert "Skills" in _capability_ids(_capabilities_for_mode("regular", "minimal", ["pdf"]))
    assert "Skills" not in _capability_ids(_capabilities_for_mode("regular", "minimal", []))
    assert "Skills" not in _capability_ids(_capabilities_for_mode("regular", "minimal", None))


def test_skills_capability_absent_in_swarm_mode() -> None:
    # Swarm is a pure router — it never gets the doing/skills tools, even with skills enabled.
    assert "Skills" not in _capability_ids(_capabilities_for_mode("swarm", "minimal", ["pdf"]))


# --------------------------------------------------------------------------- #
# Sandbox /skills mount
# --------------------------------------------------------------------------- #
def test_docker_args_mount_skills_read_only(tmp_path: Path) -> None:
    out = tmp_path / "out"
    skills = tmp_path / "skills"
    out.mkdir()
    skills.mkdir()
    args = PythonSandbox()._docker_args("n", DEFAULT_IMAGE, str(out), str(skills))
    joined = " ".join(args)
    assert f"--volume {skills}:/skills:ro" in joined
    assert "--env SKILLS_DIR=/skills" in joined
    # Hardening must be untouched by the new mount.
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--user 65534:65534"):
        assert flag in joined


def test_docker_args_omit_skills_mount_by_default(tmp_path: Path) -> None:
    joined = " ".join(PythonSandbox()._docker_args("n", DEFAULT_IMAGE, str(tmp_path)))
    assert "/skills" not in joined
    assert "SKILLS_DIR" not in joined


class SkillsSandbox:
    """Duck-typed sandbox that records the skills_dir it was given and snapshots its contents."""

    def __init__(self) -> None:
        self.skills_dir: str | None = None
        self.snapshot: dict[str, str] = {}

    async def run(
        self, code: str, timeout_seconds: float | None = None, skills_dir: str | None = None
    ) -> SandboxResult:
        self.skills_dir = skills_dir
        if skills_dir:
            for dirpath, _dirs, filenames in os.walk(skills_dir):
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, skills_dir).replace(os.sep, "/")
                    with open(full, encoding="utf-8") as fh:
                        self.snapshot[rel] = fh.read()
        return SandboxResult(stdout="ok\n", exit_code=0)


def test_run_python_materializes_enabled_skill_files(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_skill(db: Any, uid: str, ref: str) -> dict[str, Any]:
        return {
            "name": "pdf",
            "body": "# How to make a PDF",
            "files": {"scripts/make.py": {"content": "print('hi')", "encoding": "text"}},
        }

    monkeypatch.setattr("backend.skills.sandbox_capability.repo.get_skill", fake_get_skill)
    sandbox = SkillsSandbox()
    deps = GraphDependencies(
        db=FakeDb(), user_id="u", conversation_id="c", sandbox=sandbox, enabled_skills=["pdf"]
    )
    result = asyncio.run(run_python(_ctx(deps), RunPythonArgs(code="x")))
    assert result.error is None
    assert sandbox.skills_dir is not None
    assert sandbox.snapshot["pdf/SKILL.md"] == "# How to make a PDF"
    assert sandbox.snapshot["pdf/scripts/make.py"] == "print('hi')"
    # The host temp dir is cleaned up after the run.
    assert not os.path.exists(sandbox.skills_dir)


def test_run_python_without_skills_does_not_pass_skills_dir() -> None:
    """A sandbox whose run() lacks skills_dir (existing fakes) must still work when none enabled."""

    class PlainSandbox:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float | None]] = []

        async def run(self, code: str, timeout_seconds: float | None = None) -> SandboxResult:
            self.calls.append((code, timeout_seconds))
            return SandboxResult(stdout="ok\n", exit_code=0)

    sandbox = PlainSandbox()
    deps = GraphDependencies(db=FakeDb(), user_id="u", conversation_id="c", sandbox=sandbox)
    result = asyncio.run(run_python(_ctx(deps), RunPythonArgs(code="print(1)")))
    assert result.error is None
    assert sandbox.calls == [("print(1)", None)]

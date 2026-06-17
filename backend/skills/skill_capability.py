"""Skills capability: let the agent use marketplace skills enabled for the conversation.

Exposed via :func:`build_skills`, dropped into ``Agent(capabilities=...)`` only when the
conversation has skills enabled. Skills are Anthropic Agent Skills the user synced into their
database (see :mod:`backend.marketplace`) and turned on for this conversation.

**Progressive disclosure** (the Agent Skills design): the *descriptions* of the enabled skills are
injected into the system prompt every turn (cheap — see ``enabled_skills_block`` in
:mod:`backend.skills.system_prompt`); the full instructions body of a skill is loaded only when the
agent calls ``load_skill`` because the skill is relevant. Skills that ship scripts/assets have those
files mounted read-only in the ``run_python`` sandbox under ``$SKILLS_DIR/<name>/``.

The enabled skill names ride on ``ctx.deps.enabled_skills`` (set per-run by ``stream_run`` from the
conversation's stored selection). ``load_skill`` raises ``ModelRetry`` for a name that isn't enabled
(the same convention as ``update_document``/``update_fact``), so the model self-corrects.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.schemas.skill_schemas import (
    LoadSkillArgs,
    SaveSkillArgs,
    SaveSkillResult,
    SkillContent,
)

logger = logging.getLogger("agent_graph.skills")

INSTRUCTIONS = (
    "SKILLS are available to you — focused procedures (with optional bundled scripts) for specific "
    "tasks. Their names and one-line descriptions are listed in your instructions under 'Skills "
    "available'.\n"
    "- When a task matches an available skill, call `load_skill(name)` FIRST to read its full "
    "instructions, then follow them. Don't guess a skill's steps from its description alone.\n"
    "- A skill may ship files (scripts/templates/references). When it does, they are available "
    "READ-ONLY inside the `run_python` sandbox under `$SKILLS_DIR/<name>/`. The sandbox has NO "
    "network, so a skill cannot `pip install`; rely on what the sandbox image already provides.\n"
    "- Only the skills listed as available can be loaded.\n"
    "- You can AUTHOR a new skill for yourself with `save_skill(name, description, instructions)` "
    "when you've worked out a reusable procedure (e.g. researched how to design a kind of agent, "
    "then captured the method as an 'agent-architect' skill). Re-using an existing name edits it. "
    "A skill you save becomes available to you on your NEXT turn (not the current one)."
)


def skill_use_frame(tool_name: str | None, args: Any) -> dict[str, Any] | None:
    """A ``skill`` stream frame for a ``load_skill`` / ``save_skill`` call, or ``None`` otherwise.

    Lets the UI surface "Using skill X" / "Saved skill X" the moment the agent invokes a skill
    (parallel to the ``document`` frames emitted after create_document). ``args`` is the tool call's
    arguments; handles both the flat ``{"name": ...}`` and nested ``{"args": {"name": ...}}`` shapes
    Pydantic AI may produce (same defensive read as ``main._document_events`` does for
    update_document).
    """
    action = {"load_skill": "used", "save_skill": "created"}.get(tool_name or "")
    if action is None:
        return None
    data = args if isinstance(args, dict) else {}
    inner = data.get("args")
    if isinstance(inner, dict):
        data = inner
    name = str(data.get("name") or "").strip()
    if not name:
        return None
    return {"type": "skill", "action": action, "skill_name": name, "title": name}

skill_capability = Capability(id="Skills", instructions=INSTRUCTIONS)


@skill_capability.tool
async def load_skill(ctx: RunContext[GraphDependencies], args: LoadSkillArgs) -> SkillContent:
    """Load the full instructions (and file manifest) of a skill enabled for this conversation.

    Call this before acting on a skill: the description in your prompt only says *when* to use it;
    this returns *how*. Files the skill ships are available read-only in run_python under
    ``$SKILLS_DIR/<name>/``.
    """
    deps = ctx.deps
    name = args.name.strip()
    enabled = set(deps.enabled_skills or [])
    if name not in enabled:
        available = ", ".join(sorted(enabled)) or "(none)"
        raise ModelRetry(
            f"Skill {name!r} is not available to you. Available skills: {available}."
        )
    skill = await repo.get_skill(deps.db, deps.user_id, name)
    if skill is None:
        raise ModelRetry(
            f"Skill {name!r} is listed as available but its content could not be found. It may "
            "need to be re-synced/re-saved."
        )
    files = sorted((skill.get("files") or {}).keys())
    return SkillContent(
        name=str(skill.get("name") or name),
        description=str(skill.get("description") or ""),
        instructions=str(skill.get("body") or ""),
        files=files,
        sandbox_path=f"$SKILLS_DIR/{name}",
    )


@skill_capability.tool
async def save_skill(ctx: RunContext[GraphDependencies], args: SaveSkillArgs) -> SaveSkillResult:
    """Author a new skill (or edit one by name) and store it in your library for future turns.

    Use this once you've worked out a reusable procedure worth keeping — for example, after
    researching how to design a kind of specialist agent, save the method as an ``agent-architect``
    skill so it's there next time. The skill is persisted exactly like a user-authored one
    (``source="user"``): it auto-enables across the account and shows up in your "Skills available"
    list. NOTE: a skill you save here only becomes loadable on your NEXT turn (the active set is
    fixed at the start of each turn), so don't try to ``load_skill`` it immediately.
    """
    deps = ctx.deps
    existing = await repo.get_skill(deps.db, deps.user_id, args.name)
    await repo.upsert_skill(
        deps.db,
        deps.user_id,
        name=args.name,
        description=args.description,
        body=args.instructions,
        files=(existing.get("files") if existing else None) or {},
        source="user",
    )
    return SaveSkillResult(
        name=args.name, description=args.description, created=existing is None
    )


def build_skills() -> list[Capability]:
    """Return the Skills capability to add to ``Agent(capabilities=...)``.

    Added by :func:`backend.main._capabilities_for_mode` for every non-swarm conversation (it
    provides both ``load_skill`` AND ``save_skill``, so it's present even when the library is empty
    — otherwise the agent could never author its first skill). The enabled skill names are supplied
    per-run via ``GraphDependencies.enabled_skills`` (``load_skill`` ``ModelRetry``s cleanly on an
    empty library), so nothing needs to be wired in here.
    """
    return [skill_capability]

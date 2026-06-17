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

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability

from backend.db import repository as repo
from backend.db.dependencies import GraphDependencies
from backend.schemas.skill_schemas import LoadSkillArgs, SkillContent

logger = logging.getLogger("agent_graph.skills")

INSTRUCTIONS = (
    "SKILLS are enabled for this conversation — focused procedures (with optional bundled scripts) "
    "for specific tasks. Their names and one-line descriptions are listed in your instructions "
    "under 'Skills enabled for this conversation'.\n"
    "- When a task matches an enabled skill, call `load_skill(name)` FIRST to read its full "
    "instructions, then follow them. Don't guess a skill's steps from its description alone.\n"
    "- A skill may ship files (scripts/templates/references). When it does, they are available "
    "READ-ONLY inside the `run_python` sandbox under `$SKILLS_DIR/<name>/`. The sandbox has NO "
    "network, so a skill cannot `pip install`; rely on what the sandbox image already provides.\n"
    "- Only skills listed as enabled can be loaded. If a skill you need isn't enabled, tell the "
    "user they can enable it from the Configuration card."
)

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
            f"Skill {name!r} is not enabled for this conversation. Enabled skills: {available}."
        )
    skill = await repo.get_skill(deps.db, deps.user_id, name)
    if skill is None:
        raise ModelRetry(
            f"Skill {name!r} is enabled but its content could not be found. It may need to be "
            "re-synced from the marketplace."
        )
    files = sorted((skill.get("files") or {}).keys())
    return SkillContent(
        name=str(skill.get("name") or name),
        description=str(skill.get("description") or ""),
        instructions=str(skill.get("body") or ""),
        files=files,
        sandbox_path=f"$SKILLS_DIR/{name}",
    )


def build_skills() -> list[Capability]:
    """Return the Skills capability to add to ``Agent(capabilities=...)``.

    Added (by :func:`backend.main._capabilities_for_mode`) only when the conversation has skills
    enabled. The enabled skill names are supplied per-run via ``GraphDependencies.enabled_skills``,
    so nothing needs to be wired in here.
    """
    return [skill_capability]

"""Pydantic models for the Skills capability's tool inputs/outputs."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

# Kebab-case skill names, mirroring backend.api._SKILL_NAME_RE so an agent-authored skill and a
# UI-authored one (POST /api/skills) validate identically.
_SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


class LoadSkillArgs(BaseModel):
    """Which enabled skill to load the full instructions for."""

    name: str = Field(
        ...,
        min_length=1,
        description="The name of a skill enabled for this conversation (as listed in your instructions).",
    )


class SkillContent(BaseModel):
    """The full content of a loaded skill: its instructions body + the files it ships."""

    name: str = Field(..., description="The skill's name.")
    description: str = Field("", description="The skill's one-line description.")
    instructions: str = Field(
        "", description="The skill's full instructions (the SKILL.md body). Follow them."
    )
    files: list[str] = Field(
        default_factory=list,
        description=(
            "Relative paths of the files this skill ships (scripts/references/assets). When you run "
            "code with run_python, they are available read-only under $SKILLS_DIR/<name>/<path>."
        ),
    )
    sandbox_path: str = Field(
        "",
        description="Where the skill's files are mounted in the run_python sandbox (e.g. $SKILLS_DIR/pdf).",
    )


class SaveSkillArgs(BaseModel):
    """A skill to author and store for yourself: name + one-line description + instructions body."""

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "Kebab-case name (lowercase letters, digits, hyphens), e.g. 'agent-architect'. "
            "Re-using an existing name edits that skill in place."
        ),
    )
    description: str = Field(
        "",
        description=(
            "One line describing WHEN to use this skill. This is shown to you every turn (progressive "
            "disclosure) so you know the skill exists; keep it specific."
        ),
    )
    instructions: str = Field(
        "",
        description=(
            "The full SKILL.md body — the step-by-step procedure to follow when the skill applies. "
            "This is loaded on demand via load_skill, so it can be detailed."
        ),
    )

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        low = v.strip().lower()
        if not _SKILL_NAME_RE.match(low):
            raise ValueError(
                "name must be kebab-case (lowercase letters, digits, hyphens), e.g. 'my-skill'."
            )
        return low


class SaveSkillResult(BaseModel):
    """Confirmation that a skill was authored/updated and stored in your library."""

    name: str = Field(..., description="The saved skill's name.")
    description: str = Field("", description="The saved skill's one-line description.")
    created: bool = Field(
        ..., description="True if a new skill was created; False if an existing one was edited."
    )

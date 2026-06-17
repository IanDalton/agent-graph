"""Pydantic models for the Skills capability's tool inputs/outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


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

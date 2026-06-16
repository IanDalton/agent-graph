"""Tests for the best-practices base system prompt and auto-loaded user facts.

All unit tests use recording fakes and need no database, network, or embedder.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel

from backend.db.dependencies import GraphDependencies
from backend.skills.system_prompt import (
    BASE_SYSTEM_PROMPT,
    _latest_user_prompt,
    register_system_prompt,
    relevant_facts_block,
)


class FactsClient:
    """Duck-typed ArcadeClient stand-in returning canned Fact rows.

    The hybrid fact block issues two reads per turn: ``list_facts`` (the always-included important
    facts) and ``search_facts`` (the semantic LIKE path). With ``relevant`` unset both reads see the
    same ``facts`` list; pass ``relevant`` to route the LIKE path separately and exercise the merge.
    """

    def __init__(
        self,
        facts: list[dict[str, Any]] | None = None,
        *,
        relevant: list[dict[str, Any]] | None = None,
    ) -> None:
        self._facts = facts if facts is not None else []
        self._relevant = relevant

    async def command(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._relevant is not None and "LIKE" in sql.upper():
            return list(self._relevant)
        return list(self._facts)


class RaisingClient(FactsClient):
    """A db whose query() raises, to prove fact recall is tolerant."""

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise RuntimeError("db down")


def _deps(db: Any) -> GraphDependencies:
    # No embedder ⇒ search_facts uses its substring (LIKE) path against our fake query().
    return GraphDependencies(db=db, user_id="u", conversation_id="c")


def test_base_prompt_mentions_memory_and_honesty() -> None:
    assert BASE_SYSTEM_PROMPT
    lowered = BASE_SYSTEM_PROMPT.lower()
    assert "memory" in lowered
    assert "fabricate" in lowered  # the honesty guidance


def test_latest_user_prompt_picks_the_most_recent() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart(content="first question")]),
        ModelResponse(parts=[TextPart(content="an answer")]),
        ModelRequest(parts=[UserPromptPart(content="second question")]),
    ]
    assert _latest_user_prompt(messages) == "second question"


def test_latest_user_prompt_empty_when_none() -> None:
    assert _latest_user_prompt([ModelResponse(parts=[TextPart(content="hi")])]) == ""


def test_relevant_facts_block_formats_hits() -> None:
    db = FactsClient([{"fact_id": "f1", "text": "likes Recoleta apartments"}, {"fact_id": "f2", "text": "works in finance"}])
    block = asyncio.run(relevant_facts_block(_deps(db), "where should I live?"))
    assert "likes Recoleta apartments" in block
    assert "works in finance" in block
    assert block.startswith("Known facts about the user")


def test_relevant_facts_block_empty_when_no_facts() -> None:
    assert asyncio.run(relevant_facts_block(_deps(FactsClient([])), "anything")) == ""


def test_relevant_facts_block_includes_important_on_blank_query() -> None:
    # Important facts are query-independent: they load even with no current prompt.
    db = FactsClient([{"fact_id": "f1", "text": "lives in Buenos Aires"}])
    block = asyncio.run(relevant_facts_block(_deps(db), ""))
    assert "lives in Buenos Aires" in block


def test_relevant_facts_block_empty_when_nothing_important_and_blank_query() -> None:
    assert asyncio.run(relevant_facts_block(_deps(FactsClient([])), "")) == ""


def test_relevant_facts_block_merges_important_then_relevant_deduped() -> None:
    # Hybrid: important facts come first; a semantic hit already shown (same fact_id) isn't repeated.
    important = [{"fact_id": "f1", "text": "important one"}]
    relevant = [
        {"fact_id": "f1", "text": "important one"},
        {"fact_id": "f2", "text": "relevant two"},
    ]
    block = asyncio.run(relevant_facts_block(_deps(FactsClient(important, relevant=relevant)), "q"))
    assert block.count("important one") == 1  # deduped
    assert block.index("important one") < block.index("relevant two")  # important first


def test_relevant_facts_block_is_tolerant_of_db_errors() -> None:
    # A failing db must degrade to an empty block, never raise.
    assert asyncio.run(relevant_facts_block(_deps(RaisingClient()), "anything")) == ""


def test_register_system_prompt_injects_facts_into_the_request() -> None:
    """End-to-end: the dynamic instructions reach the model's system prompt under TestModel."""
    model = TestModel(call_tools=[])
    agent: Agent = Agent(model, deps_type=GraphDependencies, instructions=BASE_SYSTEM_PROMPT)
    register_system_prompt(agent)
    db = FactsClient([{"fact_id": "f1", "text": "likes Recoleta apartments"}])
    asyncio.run(agent.run("where should I live?", deps=_deps(db)))

    # The combined instructions sent to the model include the base prompt + the injected fact.
    instructions = "\n".join(p.content for p in model.last_model_request_parameters.instruction_parts)
    assert "likes Recoleta apartments" in instructions
    assert "fabricate" in instructions.lower()

"""JSON coercion shared by the streaming layer and the sub-agent runner.

A leaf module (depends only on Pydantic) so both ``backend.main`` and ``backend.skills.subagent``
can use :func:`_jsonable` without an import cycle (main imports the swarm stack, which imports
subagent).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def _jsonable(value: Any) -> Any:
    """Coerce arbitrary tool args/results into something JSON-serializable for an SSE frame.

    Tool results can be any Python value — including Pydantic models or containers OF models
    (e.g. list_documents returns ``list[DocumentInfo]``; passing that list through untouched
    made ``json.dumps`` kill the stream). Models dump to plain dicts, containers recurse,
    JSON scalars pass through, and anything else is rendered as its ``str``.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

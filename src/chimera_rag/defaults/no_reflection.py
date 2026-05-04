"""NoOp reflector — vanilla baseline (always 'perfect')."""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Answer, QualityFeedback, Query
from chimera_rag.interfaces.reflector import BaseReflector


@register("reflector", "defaults.noop")
class NoOpReflector(BaseReflector):
    """Always reports quality=1.0 so the executor never re-loops."""

    async def evaluate(self, query: Query, answer: Answer) -> QualityFeedback:  # noqa: ARG002
        return QualityFeedback(quality=1.0, relevance=1.0, completeness=1.0)


__all__ = ["NoOpReflector"]

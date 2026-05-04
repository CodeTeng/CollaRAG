"""ABC for reflectors (answer quality evaluators)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Answer, QualityFeedback, Query


class BaseReflector(ABC):
    """Evaluate an :class:`Answer` and return :class:`QualityFeedback`.

    Used by IntentAgent's Execution stage to decide whether to re-retrieve
    or stop the feedback loop.
    """

    @abstractmethod
    async def evaluate(self, query: Query, answer: Answer) -> QualityFeedback: ...


__all__ = ["BaseReflector"]

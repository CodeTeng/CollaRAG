"""ABC for answer generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Answer, Query, RetrievalResult


class BaseAnswerGenerator(ABC):
    """Synthesize the final :class:`Answer` given the retrieval output."""

    @abstractmethod
    def generate(self, query: Query, retrieval: RetrievalResult) -> Answer:
        """Return an :class:`Answer` backed by the retrieved evidence."""


__all__ = ["BaseAnswerGenerator"]

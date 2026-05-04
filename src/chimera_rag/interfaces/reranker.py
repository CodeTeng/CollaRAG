"""ABC for rerankers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Query, RetrievalResult


class BaseReranker(ABC):
    """Re-order retrieved chunks using a stronger but slower scorer
    (typically a cross-encoder).
    """

    @abstractmethod
    async def rerank(self, query: Query, retrieval: RetrievalResult) -> RetrievalResult: ...


__all__ = ["BaseReranker"]

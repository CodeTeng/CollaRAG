"""ABC for rerank providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chimera_rag.core.types import Query, RetrievalResult


class BaseRerankProvider(ABC):
    """Re-rank candidate chunks by semantic relevance to the query.

    Concrete providers call an external cross-encoder model (e.g. BGE
    Reranker-v2-v3 served via an OpenAI-compatible ``/v1/rerank`` endpoint).
    """

    @abstractmethod
    async def rerank(
        self, query: Query, result: RetrievalResult, top_n: int | None = None
    ) -> RetrievalResult: ...


__all__ = ["BaseRerankProvider"]

"""NoOp reranker — vanilla baseline (identity)."""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, RetrievalResult
from chimera_rag.interfaces.reranker import BaseReranker


@register("reranker", "defaults.noop")
class NoOpReranker(BaseReranker):
    """Returns the retrieval result unchanged."""

    async def rerank(self, query: Query, retrieval: RetrievalResult) -> RetrievalResult:
        return retrieval


__all__ = ["NoOpReranker"]

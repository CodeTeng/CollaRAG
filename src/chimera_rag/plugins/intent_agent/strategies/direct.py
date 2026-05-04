"""Direct strategy: vector top-k (+ 1-hop graph expansion).

Thin async wrapper around the vanilla :class:`DirectRetriever`.
"""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, ReasoningPlan, RetrievalResult
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.strategy import BaseRetrievalStrategy


@register("strategy", "intent_agent.direct")
class DirectStrategy(BaseRetrievalStrategy):
    def __init__(self, retriever: BaseRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k

    async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult:
        res = self.retriever.retrieve(query, top_k=self.top_k)
        res.metadata.setdefault("strategy", "direct")
        return res


__all__ = ["DirectStrategy"]

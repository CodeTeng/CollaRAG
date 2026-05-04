"""Expansion strategy: broader neighbor expansion, for exploratory queries.

Similar idea to multi_hop but with a larger ``extra_hops`` budget and no
requirement for bridging — the retrieved set is meant to be broad, not
precise.
"""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, ReasoningPlan, RetrievalResult
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.strategy import BaseRetrievalStrategy


@register("strategy", "intent_agent.expansion")
class ExpansionStrategy(BaseRetrievalStrategy):
    def __init__(self, retriever: BaseRetriever, top_k: int = 8, extra_hops: int = 2) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.extra_hops = extra_hops

    async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult:
        original_hops = getattr(self.retriever, "neighbor_hops", None)
        try:
            if original_hops is not None:
                self.retriever.neighbor_hops = original_hops + self.extra_hops
            res = self.retriever.retrieve(query, top_k=self.top_k)
        finally:
            if original_hops is not None:
                self.retriever.neighbor_hops = original_hops

        res.metadata["strategy"] = "expansion"
        res.metadata["extra_hops"] = self.extra_hops
        return res


__all__ = ["ExpansionStrategy"]

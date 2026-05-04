"""Multi-hop strategy: seed with direct retrieval, then follow graph hops.

Used by ``multi_hop`` intents. Runs a direct retrieval, then widens the
neighbor hops on the graph store to collect bridging triples.
"""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, ReasoningPlan, RetrievalResult
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.strategy import BaseRetrievalStrategy


@register("strategy", "intent_agent.multi_hop")
class MultiHopStrategy(BaseRetrievalStrategy):
    def __init__(self, retriever: BaseRetriever, top_k: int = 5, hops: int = 2) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.hops = hops

    async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult:
        # Temporarily widen neighbor_hops if the retriever exposes it.
        original_hops = getattr(self.retriever, "neighbor_hops", None)
        try:
            if original_hops is not None:
                self.retriever.neighbor_hops = max(original_hops, self.hops)
            res = self.retriever.retrieve(query, top_k=self.top_k)
        finally:
            if original_hops is not None:
                self.retriever.neighbor_hops = original_hops

        res.metadata["strategy"] = "multi_hop"
        res.metadata["hops"] = self.hops
        return res


__all__ = ["MultiHopStrategy"]

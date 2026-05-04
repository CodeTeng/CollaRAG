"""Iterative strategy: decompose the query and refine across rounds.

MVP implementation: a single direct-retrieval pass tagged with
``strategy: iterative``. Full decomposition (subquery generation via LLM
+ multi-round refinement) is reserved for a follow-up TDD cycle; the
interface is already in place so callers don't need to change.
"""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, ReasoningPlan, RetrievalResult
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.strategy import BaseRetrievalStrategy


@register("strategy", "intent_agent.iterative")
class IterativeStrategy(BaseRetrievalStrategy):
    def __init__(self, retriever: BaseRetriever, top_k: int = 5, max_iterations: int = 1) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.max_iterations = max_iterations

    async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult:
        # Placeholder single-pass iteration (MVP).
        res = self.retriever.retrieve(query, top_k=self.top_k)
        res.metadata["strategy"] = "iterative"
        res.metadata["iterations"] = 1
        return res


__all__ = ["IterativeStrategy"]

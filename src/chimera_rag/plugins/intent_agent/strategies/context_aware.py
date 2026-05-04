"""Context-aware strategy: prepend session history to the query.

Used by ``follow_up`` intents. Reads ``plan.params['history']`` (or
``plan.params['session_id']`` + external memory lookup done by the
IntentAgent orchestrator) and concatenates the last N turns as context
to the current query.
"""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, ReasoningPlan, RetrievalResult
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.strategy import BaseRetrievalStrategy


@register("strategy", "intent_agent.context_aware")
class ContextAwareStrategy(BaseRetrievalStrategy):
    def __init__(self, retriever: BaseRetriever, top_k: int = 5, history_max: int = 3) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.history_max = history_max

    async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult:
        history: list[str] = plan.params.get("history", [])
        effective_text = query.text
        if history:
            tail = history[-self.history_max :]
            effective_text = " ".join([*tail, query.text])

        # Delegate, but preserve the original session_id on the Query.
        res = self.retriever.retrieve(
            Query(text=effective_text, session_id=query.session_id),
            top_k=self.top_k,
        )
        res.metadata["strategy"] = "context_aware"
        res.metadata["history_turns_used"] = min(len(history), self.history_max)
        return res


__all__ = ["ContextAwareStrategy"]

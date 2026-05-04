"""ABC for retrieval strategies used by IntentAgent."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Query, ReasoningPlan, RetrievalResult


class BaseRetrievalStrategy(ABC):
    """A retrieval strategy consulted by IntentAgent's Decision stage.

    Six default strategies ship with the IntentAgent plugin (direct /
    multi_hop / parallel / iterative / expansion / context_aware).
    """

    @abstractmethod
    async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult: ...


__all__ = ["BaseRetrievalStrategy"]

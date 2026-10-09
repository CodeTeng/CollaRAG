"""ABC for retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Query, RetrievalResult


class BaseRetriever(ABC):
    """Retrieve relevant chunks (and triples) for a :class:`Query`."""

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 5) -> RetrievalResult:
        """Return a :class:`RetrievalResult`; may include chunks, triples, scores."""


__all__ = ["BaseRetriever"]

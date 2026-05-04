"""ABC for knowledge-graph storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Triple


class BaseGraphStore(ABC):
    """Stores and queries the knowledge graph of triples.

    Production backends: :class:`NetworkXGraphStore` (MVP),
    :class:`Neo4jGraphStore` (placeholder, raises NotImplementedError for now).
    """

    @abstractmethod
    def add_triple(self, triple: Triple) -> None: ...

    @abstractmethod
    def get_neighbors(self, entity: str, hops: int = 1) -> list[Triple]:
        """Return triples within ``hops`` hops of ``entity`` (as subject or object)."""

    @abstractmethod
    def all_triples(self) -> list[Triple]: ...

    @abstractmethod
    def persist(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


__all__ = ["BaseGraphStore"]

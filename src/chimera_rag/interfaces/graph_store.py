"""ABC for knowledge-graph storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Triple


class BaseGraphStore(ABC):
    """Stores and queries the knowledge graph of triples.

    Production backends: :class:`NetworkXGraphStore` (MVP),
    :class:`Neo4jGraphStore` (placeholder, raises NotImplementedError for now).

    G-PER (Graph-Grounded Plan-Execute-Reflect) relies on the schema- and
    path-aware query methods below (:meth:`entity_exists`,
    :meth:`relations_between`, :meth:`shortest_path`, :meth:`schema_predicates`)
    to validate plans before execution and to detect structural evidence gaps
    during reflection. Backends that cannot answer these queries should return
    the empty/False defaults so callers degrade gracefully to text-only PER.
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

    # ------------------------------------------------------------------
    # G-PER: schema- and path-aware queries
    # ------------------------------------------------------------------
    def entity_exists(self, entity: str) -> bool:
        """Return True iff ``entity`` is a known node in the graph."""
        return False

    def relations_between(self, subject: str, obj: str) -> list[str]:
        """Return the set of predicates on edges ``subject -> obj``."""
        return []

    def shortest_path(self, a: str, b: str, max_hops: int = 4) -> list[Triple]:
        """Return a shortest evidence path ``a -> ... -> b`` (empty if none)."""
        return []

    def schema_predicates(self) -> set[str]:
        """Return the set of all predicates observed in the graph (the ontology)."""
        return set()


__all__ = ["BaseGraphStore"]

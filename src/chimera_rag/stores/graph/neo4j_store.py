"""Neo4j graph store — placeholder, not yet implemented.

The class satisfies :class:`BaseGraphStore`'s ABC requirements so that
``config.yaml`` referencing ``neo4j`` can still load; every method raises
:class:`NotImplementedError` with a pointer to the real roadmap.

Implementing this is a separate work package (driver management, Cypher
templating, TTL, auth) and is deliberately out of scope for the MVP.
"""

from __future__ import annotations

from chimera_rag.core.types import Triple
from chimera_rag.interfaces.graph_store import BaseGraphStore


class Neo4jGraphStore(BaseGraphStore):
    """Stub class; will be implemented post-MVP."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password

    # All five abstract methods raise NotImplementedError.
    def add_triple(self, triple: Triple) -> None:
        raise NotImplementedError("Neo4jGraphStore is not implemented in the MVP.")

    def get_neighbors(self, entity: str, hops: int = 1) -> list[Triple]:
        raise NotImplementedError("Neo4jGraphStore is not implemented in the MVP.")

    def all_triples(self) -> list[Triple]:
        raise NotImplementedError("Neo4jGraphStore is not implemented in the MVP.")

    def persist(self, path: str) -> None:
        raise NotImplementedError("Neo4jGraphStore is not implemented in the MVP.")

    def load(self, path: str) -> None:
        raise NotImplementedError("Neo4jGraphStore is not implemented in the MVP.")


__all__ = ["Neo4jGraphStore"]

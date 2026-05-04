"""Tests for the Neo4j store placeholder.

The class exists so the ``AppConfig`` can validate a Neo4j-backed config,
but every method raises :class:`NotImplementedError` pending a real impl.
"""

from __future__ import annotations

import pytest


def test_neo4j_store_satisfies_base_graph_store_abc():
    from chimera_rag.interfaces.graph_store import BaseGraphStore
    from chimera_rag.stores.graph.neo4j_store import Neo4jGraphStore

    # Construction must be allowed (it is not itself abstract).
    assert isinstance(Neo4jGraphStore(uri="bolt://x", user="u", password="p"), BaseGraphStore)


@pytest.mark.parametrize(
    "op",
    ["add_triple", "get_neighbors", "all_triples", "persist", "load"],
)
def test_neo4j_store_methods_raise_not_implemented(op: str):
    from chimera_rag.core.types import Triple
    from chimera_rag.stores.graph.neo4j_store import Neo4jGraphStore

    s = Neo4jGraphStore(uri="bolt://x", user="u", password="p")

    with pytest.raises(NotImplementedError, match="Neo4j"):
        if op == "add_triple":
            s.add_triple(Triple(subject="a", predicate="p", object="b"))
        elif op == "get_neighbors":
            s.get_neighbors("a")
        elif op == "all_triples":
            s.all_triples()
        elif op == "persist":
            s.persist("/tmp/x")
        elif op == "load":
            s.load("/tmp/x")

"""In-memory knowledge-graph store backed by :mod:`networkx`.

Implementation notes:

* We use a :class:`networkx.MultiDiGraph` so parallel edges (same subject,
  same object, different predicates) don't collide.
* Each edge stores the full :class:`Triple` under the ``triple`` attribute
  so metadata (confidence, layer, source_chunk_id, ...) survives round-trips.
* :meth:`persist` / :meth:`load` use ``pickle`` — fine for MVP; a real
  production backend (Neo4j) lives in :mod:`neo4j_store`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from chimera_rag.core.types import Triple
from chimera_rag.interfaces.graph_store import BaseGraphStore


class NetworkXGraphStore(BaseGraphStore):
    """MVP graph store: MultiDiGraph + pickle persistence."""

    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_triple(self, triple: Triple) -> None:
        # add_edge on MultiDiGraph allows multiple predicates between the
        # same pair of nodes; the key is the predicate string.
        self._g.add_edge(triple.subject, triple.object, key=triple.predicate, triple=triple)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def all_triples(self) -> list[Triple]:
        return [data["triple"] for _, _, data in self._g.edges(data=True)]

    def get_neighbors(self, entity: str, hops: int = 1) -> list[Triple]:
        """BFS up to ``hops`` hops from ``entity`` (treating edges as undirected)."""
        if entity not in self._g:
            return []

        visited: set[str] = {entity}
        frontier: set[str] = {entity}
        collected: list[Triple] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for _ in range(max(1, hops)):
            next_frontier: set[str] = set()
            for node in frontier:
                # Outbound edges
                for _, tgt, key, data in self._g.out_edges(node, keys=True, data=True):
                    edge_id = (node, tgt, key)
                    if edge_id not in seen_edges:
                        seen_edges.add(edge_id)
                        collected.append(data["triple"])
                    if tgt not in visited:
                        next_frontier.add(tgt)
                # Inbound edges
                for src, _, key, data in self._g.in_edges(node, keys=True, data=True):
                    edge_id = (src, node, key)
                    if edge_id not in seen_edges:
                        seen_edges.add(edge_id)
                        collected.append(data["triple"])
                    if src not in visited:
                        next_frontier.add(src)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return collected

    # ------------------------------------------------------------------
    # G-PER: schema- and path-aware queries
    # ------------------------------------------------------------------
    def entity_exists(self, entity: str) -> bool:
        return entity in self._g

    def relations_between(self, subject: str, obj: str) -> list[str]:
        """Predicates on directed edges ``subject -> obj`` (deduplicated)."""
        if subject not in self._g or obj not in self._g:
            return []
        preds: list[str] = []
        seen: set[str] = set()
        for _, _, key in self._g.out_edges(subject, keys=True):
            if self._g.has_edge(subject, obj, key):
                if key not in seen:
                    seen.add(key)
                    preds.append(key)
        return preds

    def shortest_path(self, a: str, b: str, max_hops: int = 4) -> list[Triple]:
        """Shortest directed path ``a -> ... -> b`` as a list of Triples.

        Treats the graph as directed (evidence flows subject -> object). Falls
        back to undirected BFS if no directed path exists, since a supporting
        fact may be stored with either endpoint as subject.
        """
        if a not in self._g or b not in self._g or a == b:
            return []
        # 1. Directed path.
        try:
            nodes = nx.shortest_path(self._g, source=a, target=b)
        except nx.NetworkXNoPath:
            nodes = None
        except nx.NodeNotFound:
            nodes = None
        # 2. Undirected fallback.
        if not nodes:
            ug = self._g.to_undirected()
            try:
                nodes = nx.shortest_path(ug, source=a, target=b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
        if not nodes or len(nodes) - 1 > max_hops:
            return []
        # Materialize the triples along the node path.
        from itertools import pairwise

        triples: list[Triple] = []
        for u, v in pairwise(nodes):
            data = self._g.get_edge_data(u, v)
            if not data:
                # undirected fallback: edge may be stored as v -> u
                data = self._g.get_edge_data(v, u)
            if not data:
                continue
            # MultiDiGraph: data is {key: {"triple": ...}}; pick any one edge.
            any_key = next(iter(data))
            triples.append(data[any_key]["triple"])
        return triples

    def schema_predicates(self) -> set[str]:
        """The set of all predicates in the graph — the only ontology we have."""
        return {key for _, _, key in self._g.edges(keys=True)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def persist(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self._g, f)

    def load(self, path: str) -> None:
        p = Path(path)
        with p.open("rb") as f:
            g = pickle.load(f)
        if not isinstance(g, nx.MultiDiGraph):
            raise TypeError(f"expected MultiDiGraph, got {type(g).__name__}")
        self._g = g


__all__ = ["NetworkXGraphStore"]

"""Retrieval tools: vector, BM25, hybrid, graph neighbors, graph path, graph community."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.core.types import Chunk, RetrievalResult, Triple

_PREVIEW_CHARS = 160


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"


class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi for chunk retrieval."""

    def __init__(self, chunk_ids: list[str], bm25: Any) -> None:
        self._chunk_ids = chunk_ids
        self._bm25 = bm25

    @classmethod
    def from_chunks(cls, chunk_lookup: dict[str, Chunk]) -> BM25Index:
        if not chunk_lookup:
            return cls([], None)
        from rank_bm25 import BM25Okapi

        ids = list(chunk_lookup.keys())
        tokenized = [re.findall(r"\w+", chunk_lookup[cid].text.lower()) for cid in ids]
        bm25 = BM25Okapi(tokenized)
        return cls(ids, bm25)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._chunk_ids[i], float(s)) for i, s in ranked if s > 0]


@dataclass
class ToolAccumulator:
    """Mutable per-query evidence accumulator shared across tool calls."""

    _chunks: dict[str, Chunk] = field(default_factory=dict)
    _scores: dict[str, float] = field(default_factory=dict)
    _triples: list[Triple] = field(default_factory=list)
    _seen_triple_keys: set[tuple[str, str, str]] = field(default_factory=set)

    def add_chunks(self, chunks: list[Chunk], scores: list[float]) -> None:
        for chunk, score in zip(chunks, scores):
            self._chunks.setdefault(chunk.chunk_id, chunk)
            prev = self._scores.get(chunk.chunk_id)
            if prev is None or score > prev:
                self._scores[chunk.chunk_id] = float(score)

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            key = (t.subject, t.predicate, t.object)
            if key not in self._seen_triple_keys:
                self._seen_triple_keys.add(key)
                self._triples.append(t)

    def merged_result(self, top_k: int | None = None) -> RetrievalResult:
        ordered = sorted(self._scores.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            ordered = ordered[:top_k]
        chunks = [self._chunks[cid] for cid, _ in ordered]
        scores = [s for _, s in ordered]
        return RetrievalResult(chunks=chunks, triples=list(self._triples), scores=scores)


class _QueryTopKArgs(BaseModel):
    query: str = Field(description="Natural-language query.")
    top_k: int = Field(default=5, ge=1, le=50)


class _HybridArgs(BaseModel):
    query: str = Field(description="Natural-language query.")
    top_k: int = Field(default=5, ge=1, le=50)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class _GraphNeighborsArgs(BaseModel):
    entity: str = Field(description="Entity name to look up.")
    hops: int = Field(default=1, ge=1, le=4)


class _GraphPathArgs(BaseModel):
    entity_a: str = Field(description="Start entity.")
    entity_b: str = Field(description="End entity.")
    max_depth: int = Field(default=4, ge=1, le=8)


class _GraphCommunityArgs(BaseModel):
    query: str = Field(description="Query for community search.")
    top_k: int = Field(default=3, ge=1, le=20)


def _summarise(chunks: list[Chunk], scores: list[float], label: str) -> dict:
    return {
        "tool": label,
        "num_chunks": len(chunks),
        "chunks": [
            {"chunk_id": c.chunk_id, "score": round(float(s), 4), "preview": _preview(c.text)}
            for c, s in zip(chunks, scores)
        ],
    }


def build_retrieval_tools(
    *,
    vector_store,
    graph_store,
    embedder,
    chunk_lookup: dict[str, Chunk],
    bm25_index: BM25Index,
    accumulator: ToolAccumulator,
) -> list[StructuredTool]:

    def _vector_search(query: str, top_k: int = 5) -> dict:
        qvec = embedder.encode([query])[0]
        hits = vector_store.search(qvec, top_k=top_k)
        chunks, scores = [], []
        for cid, score in hits:
            ch = chunk_lookup.get(cid)
            if ch:
                chunks.append(ch)
                scores.append(score)
        accumulator.add_chunks(chunks, scores)
        return _summarise(chunks, scores, "vector_search")

    def _bm25_search(query: str, top_k: int = 5) -> dict:
        hits = bm25_index.search(query, top_k=top_k)
        chunks, scores = [], []
        for cid, score in hits:
            ch = chunk_lookup.get(cid)
            if ch:
                chunks.append(ch)
                scores.append(score)
        accumulator.add_chunks(chunks, scores)
        return _summarise(chunks, scores, "bm25_search")

    def _hybrid_search(query: str, top_k: int = 5, vector_weight: float = 0.5) -> dict:
        qvec = embedder.encode([query])[0]
        vec_hits = vector_store.search(qvec, top_k=top_k * 2)
        bm25_hits = bm25_index.search(query, top_k=top_k * 2)
        rrf: dict[str, float] = {}
        k = 60
        for rank, (cid, _) in enumerate(vec_hits):
            rrf[cid] = rrf.get(cid, 0.0) + vector_weight / (k + rank + 1)
        for rank, (cid, _) in enumerate(bm25_hits):
            rrf[cid] = rrf.get(cid, 0.0) + (1 - vector_weight) / (k + rank + 1)
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        chunks, scores = [], []
        for cid, score in ranked:
            ch = chunk_lookup.get(cid)
            if ch:
                chunks.append(ch)
                scores.append(score)
        accumulator.add_chunks(chunks, scores)
        return _summarise(chunks, scores, "hybrid_search")

    def _graph_neighbors(entity: str, hops: int = 1) -> dict:
        triples = graph_store.get_neighbors(entity, hops=hops)
        accumulator.add_triples(triples)
        return {
            "tool": "graph_neighbors",
            "entity": entity,
            "hops": hops,
            "num_triples": len(triples),
            "triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in triples[:20]
            ],
        }

    def _graph_path_search(entity_a: str, entity_b: str, max_depth: int = 4) -> dict:
        import networkx as nx

        g = getattr(graph_store, "_graph", None)
        if g is None:
            return {"tool": "graph_path_search", "paths": [], "num_paths": 0}
        ug = g.to_undirected()
        paths: list[list[str]] = []
        try:
            for path in nx.all_simple_paths(ug, entity_a, entity_b, cutoff=max_depth):
                paths.append(list(path))
                if len(paths) >= 5:
                    break
        except (nx.NodeNotFound, nx.NetworkXError):
            pass
        return {"tool": "graph_path_search", "entity_a": entity_a, "entity_b": entity_b,
                "num_paths": len(paths), "paths": paths}

    def _graph_community_search(query: str, top_k: int = 3) -> dict:
        all_triples = graph_store.all_triples()
        query_lower = query.lower()
        q_tokens = set(re.findall(r"\w+", query_lower))
        scored = []
        for t in all_triples:
            text = f"{t.subject} {t.predicate} {t.object}".lower()
            overlap = len(q_tokens & set(re.findall(r"\w+", text)))
            if overlap > 0:
                scored.append((t, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_triples = [t for t, _ in scored[:top_k * 3]]
        accumulator.add_triples(top_triples)
        return {
            "tool": "graph_community_search",
            "num_triples": len(top_triples),
            "triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in top_triples[:20]
            ],
        }

    return [
        StructuredTool.from_function(_vector_search, name="vector_search",
            description="Semantic vector search. Use for direct factual lookups.",
            args_schema=_QueryTopKArgs),
        StructuredTool.from_function(_bm25_search, name="bm25_search",
            description="BM25 keyword search. Use when exact term matching matters.",
            args_schema=_QueryTopKArgs),
        StructuredTool.from_function(_hybrid_search, name="hybrid_search",
            description="RRF fusion of vector + BM25. Best general-purpose retrieval.",
            args_schema=_HybridArgs),
        StructuredTool.from_function(_graph_neighbors, name="graph_neighbors",
            description="Expand knowledge graph around an entity by N hops.",
            args_schema=_GraphNeighborsArgs),
        StructuredTool.from_function(_graph_path_search, name="graph_path_search",
            description="Find paths between two entities in the knowledge graph.",
            args_schema=_GraphPathArgs),
        StructuredTool.from_function(_graph_community_search, name="graph_community_search",
            description="Find triples related to a query topic in the knowledge graph.",
            args_schema=_GraphCommunityArgs),
    ]


__all__ = ["BM25Index", "ToolAccumulator", "build_retrieval_tools"]

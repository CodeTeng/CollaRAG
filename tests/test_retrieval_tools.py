"""Tests for retrieval tools (vector, bm25, hybrid, graph)."""
import numpy as np

from chimera_rag.core.types import Chunk, Triple
from chimera_rag.plugins.colla_rag.tools.retrieval_tools import (
    BM25Index,
    ToolAccumulator,
    build_retrieval_tools,
)


class MockVectorStore:
    def search(self, query_vector, top_k=5):
        return [("c1", 0.95), ("c2", 0.80)]

    def add(self, ids, vectors):
        pass


class MockGraphStore:
    def get_neighbors(self, entity, hops=1):
        if entity == "Einstein":
            return [Triple(subject="Einstein", predicate="born_in", object="Ulm")]
        return []

    def all_triples(self):
        return [
            Triple(subject="Einstein", predicate="born_in", object="Ulm"),
            Triple(subject="Bohr", predicate="developed", object="atomic model"),
        ]


class MockEmbedder:
    def encode(self, texts):
        return np.random.randn(len(texts), 64).astype(np.float32)


def _setup():
    chunks = {
        "c1": Chunk(chunk_id="c1", doc_id="d1", index=0, text="Einstein was born in Ulm"),
        "c2": Chunk(chunk_id="c2", doc_id="d1", index=1, text="Bohr developed atomic model"),
    }
    bm25 = BM25Index.from_chunks(chunks)
    acc = ToolAccumulator()
    return chunks, bm25, acc


def test_build_retrieval_tools_returns_six():
    chunks, bm25, acc = _setup()
    tools = build_retrieval_tools(
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        bm25_index=bm25,
        accumulator=acc,
    )
    names = {t.name for t in tools}
    assert "vector_search" in names
    assert "bm25_search" in names
    assert "hybrid_search" in names
    assert "graph_neighbors" in names
    assert "graph_path_search" in names
    assert "graph_community_search" in names


def test_vector_search_tool_invocation():
    chunks, bm25, acc = _setup()
    tools = build_retrieval_tools(
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        bm25_index=bm25,
        accumulator=acc,
    )
    vs_tool = next(t for t in tools if t.name == "vector_search")
    result = vs_tool.invoke({"query": "Einstein", "top_k": 2})
    assert result["tool"] == "vector_search"
    assert result["num_chunks"] >= 0


def test_bm25_search_tool_invocation():
    chunks, bm25, acc = _setup()
    tools = build_retrieval_tools(
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        bm25_index=bm25,
        accumulator=acc,
    )
    bm_tool = next(t for t in tools if t.name == "bm25_search")
    result = bm_tool.invoke({"query": "Einstein born", "top_k": 2})
    assert result["tool"] == "bm25_search"


def test_graph_neighbors_tool():
    chunks, bm25, acc = _setup()
    tools = build_retrieval_tools(
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        bm25_index=bm25,
        accumulator=acc,
    )
    gn_tool = next(t for t in tools if t.name == "graph_neighbors")
    result = gn_tool.invoke({"entity": "Einstein", "hops": 1})
    assert result["tool"] == "graph_neighbors"
    assert result["num_triples"] >= 1

"""Tests for BM25Index wrapper and ToolAccumulator."""
from chimera_rag.core.types import Chunk
from chimera_rag.plugins.colla_rag.tools.retrieval_tools import (
    BM25Index,
    ToolAccumulator,
)


def _make_chunks() -> dict[str, Chunk]:
    return {
        "c1": Chunk(chunk_id="c1", doc_id="d1", index=0, text="Einstein was born in Ulm Germany"),
        "c2": Chunk(chunk_id="c2", doc_id="d1", index=1, text="Bohr developed the atomic model"),
        "c3": Chunk(chunk_id="c3", doc_id="d1", index=2, text="Quantum mechanics revolutionized physics"),
    }


def test_bm25_index_build_and_search():
    chunks = _make_chunks()
    idx = BM25Index.from_chunks(chunks)
    results = idx.search("Einstein born", top_k=2)
    assert len(results) <= 2
    assert results[0][0] == "c1"
    assert results[0][1] > 0.0


def test_bm25_index_empty():
    idx = BM25Index.from_chunks({})
    results = idx.search("anything", top_k=5)
    assert results == []


def test_tool_accumulator_add_and_merge():
    acc = ToolAccumulator()
    chunks = _make_chunks()
    acc.add_chunks([chunks["c1"], chunks["c2"]], [0.9, 0.7])
    acc.add_chunks([chunks["c1"], chunks["c3"]], [0.8, 0.6])
    merged = acc.merged_result(top_k=3)
    assert len(merged.chunks) == 3
    c1_idx = next(i for i, c in enumerate(merged.chunks) if c.chunk_id == "c1")
    assert merged.scores[c1_idx] == 0.9


def test_tool_accumulator_top_k_limit():
    acc = ToolAccumulator()
    chunks = _make_chunks()
    acc.add_chunks(list(chunks.values()), [0.9, 0.7, 0.5])
    merged = acc.merged_result(top_k=2)
    assert len(merged.chunks) == 2

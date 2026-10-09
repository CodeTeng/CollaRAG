"""Tests for :mod:`chimera_rag.defaults.direct_retriever`."""

from __future__ import annotations


def test_direct_retriever_returns_top_k_chunks_by_vector_similarity():

    from _fakes import FakeEmbeddingProvider
    from chimera_rag.core.types import Chunk, Query
    from chimera_rag.defaults.retriever import DirectRetriever
    from chimera_rag.stores.graph import NetworkXGraphStore
    from chimera_rag.stores.vector import FAISSVectorStore

    # Build a tiny corpus
    chunks = [
        Chunk(chunk_id="c0", doc_id="d", index=0, text="Nobel invented dynamite"),
        Chunk(chunk_id="c1", doc_id="d", index=1, text="Einstein formulated relativity"),
        Chunk(chunk_id="c2", doc_id="d", index=2, text="Curie discovered radium"),
    ]
    embedder = FakeEmbeddingProvider(dim=16, normalize=True)
    texts = [c.text for c in chunks]
    vecs = embedder.encode(texts)
    vs = FAISSVectorStore(dim=16, index_type="flat_ip")
    vs.add([c.chunk_id for c in chunks], vecs)

    gs = NetworkXGraphStore()

    r = DirectRetriever(
        vector_store=vs,
        graph_store=gs,
        embedder=embedder,
        chunk_lookup={c.chunk_id: c for c in chunks},
        neighbor_hops=0,
    )
    # Querying with the first chunk's exact text should rank it first.
    out = r.retrieve(Query(text="Nobel invented dynamite"), top_k=2)
    assert len(out.chunks) == 2
    assert out.chunks[0].chunk_id == "c0"
    assert len(out.scores) == 2
    assert out.scores[0] >= out.scores[1]


def test_direct_retriever_pulls_graph_neighbors_when_hops_positive():

    from _fakes import FakeEmbeddingProvider
    from chimera_rag.core.types import Chunk, Query, Triple
    from chimera_rag.defaults.retriever import DirectRetriever
    from chimera_rag.stores.graph import NetworkXGraphStore
    from chimera_rag.stores.vector import FAISSVectorStore

    chunks = [
        Chunk(chunk_id="c0", doc_id="d", index=0, text="Alfred Nobel invented dynamite"),
    ]
    embedder = FakeEmbeddingProvider(dim=16, normalize=True)
    vs = FAISSVectorStore(dim=16, index_type="flat_ip")
    vs.add(["c0"], embedder.encode([chunks[0].text]))

    gs = NetworkXGraphStore()
    # Use single-word node names so the capitalised-token heuristic inside
    # DirectRetriever can match them.
    gs.add_triple(
        Triple(
            subject="Nobel",
            predicate="invented",
            object="Dynamite",
            source_chunk_id="c0",
        )
    )
    gs.add_triple(
        Triple(
            subject="Nobel",
            predicate="founded",
            object="NobelPrize",
            source_chunk_id="c0",
        )
    )

    r = DirectRetriever(
        vector_store=vs,
        graph_store=gs,
        embedder=embedder,
        chunk_lookup={c.chunk_id: c for c in chunks},
        neighbor_hops=1,
    )
    # Query mentions "Nobel" (capitalised), which matches a graph node.
    out = r.retrieve(Query(text="What did Nobel invent?"), top_k=1)
    assert len(out.triples) >= 1


def test_direct_retriever_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.retriever import DirectRetriever

    assert get_registry().get("retriever", "defaults.direct") is DirectRetriever


def test_direct_retriever_shares_chunk_lookup_reference_when_empty():
    """Regression: empty-dict falsy trap used to break pipeline integration.

    Callers (the top-level Pipeline) pass an initially-empty dict and later
    fill it during ingest. The retriever must hold the same reference.
    """
    from _fakes import FakeEmbeddingProvider
    from chimera_rag.defaults.retriever import DirectRetriever
    from chimera_rag.stores.graph import NetworkXGraphStore
    from chimera_rag.stores.vector import FAISSVectorStore

    shared: dict = {}
    r = DirectRetriever(
        vector_store=FAISSVectorStore(dim=8),
        graph_store=NetworkXGraphStore(),
        embedder=FakeEmbeddingProvider(dim=8, normalize=True),
        chunk_lookup=shared,
    )
    assert r.chunk_lookup is shared

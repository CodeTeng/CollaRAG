"""Tests for :mod:`chimera_rag.defaults.fixed_chunker`."""

from __future__ import annotations


def test_fixed_chunker_splits_long_text_into_multiple_chunks():
    from chimera_rag.core.types import Document
    from chimera_rag.defaults.chunker import FixedSizeChunker

    text = "word " * 200  # ~1000 chars
    c = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = c.chunk(Document(doc_id="d1", content=text))

    assert len(chunks) > 1
    # Chunk ids follow "<doc_id>::<index>" convention; indices start at 0.
    assert chunks[0].chunk_id == "d1::0"
    assert chunks[0].doc_id == "d1"
    assert chunks[0].index == 0
    # No chunk exceeds chunk_size.
    assert all(len(ch.text) <= 100 for ch in chunks)


def test_fixed_chunker_overlap_is_respected():
    """Consecutive chunks must share the configured overlap in their tails/heads."""
    from chimera_rag.core.types import Document
    from chimera_rag.defaults.chunker import FixedSizeChunker

    text = "a" * 250
    c = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = c.chunk(Document(doc_id="d", content=text))

    assert len(chunks) >= 2
    # Chunk 0 ends at 100, chunk 1 starts at 80 -> chunk_1[:20] == chunk_0[-20:]
    assert chunks[1].text[:20] == chunks[0].text[-20:]


def test_fixed_chunker_shorter_than_chunk_size_produces_single_chunk():
    from chimera_rag.core.types import Document
    from chimera_rag.defaults.chunker import FixedSizeChunker

    c = FixedSizeChunker(chunk_size=100, overlap=20)
    chunks = c.chunk(Document(doc_id="d", content="short text"))
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_fixed_chunker_empty_document_returns_empty_list():
    from chimera_rag.core.types import Document
    from chimera_rag.defaults.chunker import FixedSizeChunker

    c = FixedSizeChunker(chunk_size=100, overlap=20)
    assert c.chunk(Document(doc_id="d", content="")) == []


def test_fixed_chunker_overlap_must_be_less_than_chunk_size():
    import pytest

    from chimera_rag.defaults.chunker import FixedSizeChunker

    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, overlap=60)


def test_fixed_chunker_registers_under_defaults_fixed_slot():
    import chimera_rag.defaults  # noqa: F401 - trigger registration
    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.chunker import FixedSizeChunker

    cls = get_registry().get("chunker", "defaults.fixed")
    assert cls is FixedSizeChunker

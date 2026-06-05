"""Tests for :mod:`chimera_rag.defaults.langchain_chunker`.

These tests require ``langchain-text-splitters`` (the ``chunk`` extra). When
the optional dependency is absent the whole module is skipped rather than
failing the suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "langchain_text_splitters",
    reason="install the 'chunk' extra: uv sync --extra chunk",
)


def test_langchain_chunker_splits_long_text_into_multiple_chunks():
    from chimera_rag.core.types import Document
    from chimera_rag.defaults.langchain_chunker import LangChainChunker

    text = "word " * 200  # ~1000 chars
    c = LangChainChunker(chunk_size=100, overlap=20)
    chunks = c.chunk(Document(doc_id="d1", content=text))

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "d1::0"
    assert chunks[0].doc_id == "d1"
    assert chunks[0].index == 0
    # RecursiveCharacterTextSplitter keeps chunks within chunk_size.
    assert all(len(ch.text) <= 100 for ch in chunks)


def test_langchain_chunker_indices_are_sequential():
    from chimera_rag.core.types import Document
    from chimera_rag.defaults.langchain_chunker import LangChainChunker

    text = "Sentence one. Sentence two. Sentence three. " * 20
    c = LangChainChunker(chunk_size=80, overlap=10)
    chunks = c.chunk(Document(doc_id="doc", content=text))

    assert [ch.index for ch in chunks] == list(range(len(chunks)))
    assert [ch.chunk_id for ch in chunks] == [f"doc::{i}" for i in range(len(chunks))]


def test_langchain_chunker_short_text_single_chunk():
    from chimera_rag.core.types import Document
    from chimera_rag.defaults.langchain_chunker import LangChainChunker

    c = LangChainChunker(chunk_size=100, overlap=20)
    chunks = c.chunk(Document(doc_id="d", content="short text"))
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_langchain_chunker_empty_document_returns_empty_list():
    from chimera_rag.core.types import Document
    from chimera_rag.defaults.langchain_chunker import LangChainChunker

    c = LangChainChunker(chunk_size=100, overlap=20)
    assert c.chunk(Document(doc_id="d", content="")) == []


def test_langchain_chunker_overlap_must_be_less_than_chunk_size():
    from chimera_rag.defaults.langchain_chunker import LangChainChunker

    with pytest.raises(ValueError):
        LangChainChunker(chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        LangChainChunker(chunk_size=50, overlap=60)


def test_langchain_chunker_registers_under_defaults_langchain_slot():
    import chimera_rag.defaults  # noqa: F401 - trigger registration
    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.langchain_chunker import LangChainChunker

    cls = get_registry().get("chunker", "defaults.langchain")
    assert cls is LangChainChunker

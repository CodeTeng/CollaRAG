"""LangChain-backed length chunker.

A thin wrapper around LangChain's
:class:`~langchain_text_splitters.RecursiveCharacterTextSplitter` so users
who prefer a battle-tested splitting algorithm can switch to it via config
without changing any pipeline code::

    ingestion:
      chunker:
        active: defaults.langchain
        params: { chunk_size: 300, overlap: 50 }

The recursive splitter tries a descending list of separators
(paragraph -> line -> sentence -> word -> char), keeping semantically
coherent boundaries while still honouring ``chunk_size`` / ``overlap``.

``chunk_id`` follows the same ``{doc_id}::{index}`` convention as
:class:`chimera_rag.defaults.fixed_chunker.FixedSizeChunker` so downstream
components (vector store ids, evidence lookup) behave identically.

``langchain_text_splitters`` is imported lazily so the optional dependency
is only required when this chunker is actually selected.
"""

from __future__ import annotations

from chimera_rag.core.exceptions import ProviderError
from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Document
from chimera_rag.interfaces.chunker import BaseChunker


@register("chunker", "defaults.langchain")
class LangChainChunker(BaseChunker):
    """Length-based chunker delegating to RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly smaller than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = self._build_splitter()

    def _build_splitter(self):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as e:  # pragma: no cover - optional dep path
            raise ProviderError(
                "langchain-text-splitters not installed; "
                "run `uv sync --extra chunk` to use the 'defaults.langchain' chunker "
                f"({e})"
            ) from e

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []

        pieces = self._splitter.split_text(text)
        return [
            Chunk(
                chunk_id=f"{document.doc_id}::{idx}",
                doc_id=document.doc_id,
                index=idx,
                text=piece,
            )
            for idx, piece in enumerate(pieces)
        ]


__all__ = ["LangChainChunker"]

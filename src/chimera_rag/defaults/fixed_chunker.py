"""Fixed-size character window chunker — the vanilla baseline.

Split strategy: slide a window of ``chunk_size`` characters with step
``chunk_size - overlap``; the last partial window is kept as-is.

Character-based (not token-based) for zero-dep determinism. Real
production systems should swap in a tokeniser-aware version, but for
MVP / research comparison this keeps tests deterministic.
"""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Document
from chimera_rag.interfaces.chunker import BaseChunker


@register("chunker", "defaults.fixed")
class FixedSizeChunker(BaseChunker):
    """Sliding-window chunker parameterised by ``chunk_size`` and ``overlap``."""

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

    # ------------------------------------------------------------------
    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        i = 0
        idx = 0
        while i < len(text):
            piece = text[i : i + self.chunk_size]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{idx}",
                    doc_id=document.doc_id,
                    index=idx,
                    text=piece,
                )
            )
            if i + self.chunk_size >= len(text):
                break
            i += step
            idx += 1
        return chunks


__all__ = ["FixedSizeChunker"]

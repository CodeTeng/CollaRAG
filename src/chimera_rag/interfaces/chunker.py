"""ABC for chunkers: Document -> list[Chunk]."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Chunk, Document


class BaseChunker(ABC):
    """Split a :class:`Document` into :class:`Chunk` objects.

    Implementations include:

    * :class:`chimera_rag.defaults.fixed_chunker.FixedSizeChunker` — fixed
      token window with overlap (vanilla baseline).
    * :class:`chimera_rag.plugins.adagraph.dynamic_chunker.DynamicChunker` —
      AdaGraph's complexity/density-driven adaptive size.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Return an ordered list of chunks with 0-based indices."""


__all__ = ["BaseChunker"]

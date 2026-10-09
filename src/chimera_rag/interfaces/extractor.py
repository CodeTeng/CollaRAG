"""ABC for triple extractors: Chunk -> list[Triple]."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Chunk, Triple


class BaseTripleExtractor(ABC):
    """Extract (subject, predicate, object) triples from a single chunk."""

    @abstractmethod
    def extract(self, chunk: Chunk) -> list[Triple]:
        """Return triples discovered in ``chunk`` (may be empty)."""


__all__ = ["BaseTripleExtractor"]

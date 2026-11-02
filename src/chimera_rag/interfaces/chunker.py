"""ABCs for the chunker slot: ``Document -> list[Chunk]``, plus the optional
audit-and-repair capability that PrismRAG adds on top of it.

Two ABCs live here rather than one because they are different *kinds* of
contract:

* :class:`BaseChunker` is the slot every configuration must fill. Its signature
  never changes, which is what keeps the registry swappable.
* :class:`BaseChunkAuditor` is an **optional capability**. ``IngestionPipeline``
  discovers it with ``isinstance`` and only then runs the
  ``extract -> audit -> repair`` loop, so a chunker that does not implement it
  costs nothing and ``core`` still imports nothing from ``plugins``.

Keeping the capability on the chunker (rather than opening a new registry slot)
means PrismRAG occupies exactly one slot; see ``docs/innovation-1/design.md``
§9.1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from chimera_rag.core.types import Chunk, ChunkDefect, Document, RepairOutcome
from chimera_rag.interfaces.graph_store import BaseGraphStore


class BaseChunker(ABC):
    """Split a :class:`Document` into :class:`Chunk` objects.

    Implementations include:

    * :class:`chimera_rag.defaults.chunker.FixedSizeChunker` — fixed character
      window with overlap (vanilla baseline). Emits every chunk at
      ``granularity="unified"``, i.e. both extracted and indexed.
    * :class:`chimera_rag.plugins.prism.chunker.PrismChunker` — PrismRAG. May
      emit a *mix* of granularities for one document: ``fine`` chunks feed only
      the extractor, ``coarse`` chunks only the vector store, bound by
      ``parent_chunk_id``. See ``core.types.ChunkGranularity``.

    The contract is deliberately granularity-agnostic: the returned list may
    mix layers, and the pipeline routes each chunk by its ``granularity``.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Return an ordered list of chunks with 0-based indices.

        Indices are numbered *within a granularity layer*, so under dual
        granularity a document has a ``fine`` chunk 0 and a ``coarse`` chunk 0.
        Chunk ids must therefore be layer-namespaced to stay unique.
        """


class BaseChunkAuditor(ABC):
    """Optional capability: judge extraction output and re-cut the boundaries.

    This is the hook that makes chunking a closed loop instead of a one-shot
    a-priori decision. It needs the extractor's output, which ``chunk()``
    cannot see — hence a second method pair driven by the pipeline.

    Implementations must make :meth:`repair` a **boundary change**, never an
    extraction retry. Risk R1 in the design doc: if repairing just means
    extracting again and unioning the results, the whole contribution collapses
    into the ``extract-twice-union`` counter-baseline.
    """

    @abstractmethod
    def audit(
        self,
        chunks: Sequence[Chunk],
        triples_by_chunk: Mapping[str, Sequence],
        graph: BaseGraphStore | None = None,
    ) -> list[ChunkDefect]:
        """Return every defect found in *chunks*, including low-confidence ones.

        Called once per round with the full batch, so cross-chunk detectors can
        see their siblings. Reporting a defect does not commit to repairing it:
        the pipeline filters on confidence before calling :meth:`repair`.
        """

    @abstractmethod
    def repair(
        self, defects: Sequence[ChunkDefect], chunks: Sequence[Chunk]
    ) -> RepairOutcome:
        """Re-segment *chunks* to fix *defects*, returning the new chunk list.

        Only repair-driving defects are passed in. Implementations should apply
        at most one action per chunk per round and defer the rest, so that
        conflicting edits (split this / merge that) cannot corrupt a round.
        """


__all__ = ["BaseChunkAuditor", "BaseChunker"]

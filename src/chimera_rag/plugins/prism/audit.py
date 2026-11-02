"""The PrismRAG auditor: run D1–D6 over a batch, then delegate the re-cut.

This is the class that turns six independent detectors into the single
``audit`` / ``repair`` capability :class:`chimera_rag.interfaces.chunker.BaseChunkAuditor`
declares, and therefore the only thing ``IngestionPipeline`` has to know about.

One non-obvious job lives here: building the graph view D1 falsifies against.
The pipeline writes triples to the real store only *after* the loop settles, so
passing that store through would hand D1 an empty graph on every round and
silently promote all its low-confidence hub-entity false positives to
repair-driving. The view is rebuilt per round from the current triple set plus
whatever previous ingests already committed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from chimera_rag.core.types import Chunk, ChunkDefect, RepairOutcome, Triple
from chimera_rag.interfaces.chunker import BaseChunkAuditor
from chimera_rag.interfaces.graph_store import BaseGraphStore
from chimera_rag.plugins.prism.defects import (
    AuditContext,
    D1CollisionDetector,
    D2AntecedentDetector,
    D3StraddleDetector,
    D4StarvationDetector,
    D5BudgetDetector,
    D6ContradictionDetector,
    DefectDetector,
)
from chimera_rag.plugins.prism.repair import apply_repairs


def default_detectors(*, triple_budget: int = 30) -> list[DefectDetector]:
    """The full D1–D6 set.

    ``triple_budget`` must mirror the extractor's ``max_triples_per_chunk``;
    D5 reads that cap as its signal, so a mismatch makes D5 fire early or never.
    """
    return [
        D1CollisionDetector(),
        D2AntecedentDetector(),
        D3StraddleDetector(),
        D4StarvationDetector(),
        D5BudgetDetector(budget=triple_budget),
        D6ContradictionDetector(),
    ]


class PrismAuditor(BaseChunkAuditor):
    """Aggregates the detectors and applies :func:`apply_repairs`.

    ``repair_enabled=False`` gives experiment E2 its "audit but do not repair"
    row: every defect is still reported and recorded, and :meth:`repair`
    returns the segmentation untouched so the loop stops after one round.
    """

    def __init__(
        self,
        detectors: Sequence[DefectDetector] | None = None,
        *,
        triple_budget: int = 30,
        repair_enabled: bool = True,
    ) -> None:
        self.detectors: list[DefectDetector] = (
            list(detectors)
            if detectors is not None
            else default_detectors(triple_budget=triple_budget)
        )
        self.repair_enabled = repair_enabled

    # ------------------------------------------------------------------
    def audit(
        self,
        chunks: Sequence[Chunk],
        triples_by_chunk: Mapping[str, Sequence[Triple]],
        graph: BaseGraphStore | None = None,
    ) -> list[ChunkDefect]:
        ctx = AuditContext(
            graph=self._audit_view(triples_by_chunk, graph),
            chunks=tuple(chunks),
            triples_by_chunk=triples_by_chunk,
        )
        found: list[ChunkDefect] = []
        for ch in chunks:
            triples = triples_by_chunk.get(ch.chunk_id, [])
            for detector in self.detectors:
                found.extend(detector.detect(ch, triples, ctx))
        return found

    # ------------------------------------------------------------------
    def repair(
        self, defects: Sequence[ChunkDefect], chunks: Sequence[Chunk]
    ) -> RepairOutcome:
        if not self.repair_enabled:
            return RepairOutcome(chunks=list(chunks), defects_deferred=len(defects))
        return apply_repairs(defects, chunks)

    # ------------------------------------------------------------------
    @staticmethod
    def _audit_view(
        triples_by_chunk: Mapping[str, Sequence[Triple]], graph: BaseGraphStore | None
    ) -> BaseGraphStore:
        """This round's triples plus everything already committed."""
        from chimera_rag.stores.graph import NetworkXGraphStore

        view = NetworkXGraphStore()
        if graph is not None:
            for t in graph.all_triples():
                view.add_triple(t)
        for triples in triples_by_chunk.values():
            for t in triples:
                view.add_triple(t)
        return view


__all__ = ["PrismAuditor", "default_detectors"]

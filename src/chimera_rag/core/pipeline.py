"""Ingestion and Query pipelines — the dependency-injection heart of
Chimera-RAG.

Pipelines hold references to the pluggable slots resolved from the
Registry at construction time; they never import concrete plugin
classes themselves. This keeps ``core`` decoupled from ``defaults`` and
``plugins`` as required by the project design.
"""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from chimera_rag.core.types import (
    EXTRACTED_GRANULARITIES,
    INDEXED_GRANULARITIES,
    Answer,
    Chunk,
    ChunkDefect,
    Document,
    Query,
    Triple,
)
from chimera_rag.interfaces.chunker import BaseChunkAuditor, BaseChunker
from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider
from chimera_rag.interfaces.extractor import BaseTripleExtractor
from chimera_rag.interfaces.generator import BaseAnswerGenerator
from chimera_rag.interfaces.graph_store import BaseGraphStore
from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier
from chimera_rag.interfaces.pruner import BasePruner
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SharedState: ingest fills it, query reads from it
# ---------------------------------------------------------------------------
@dataclass
class PipelineState:
    """Mutable state shared by ingestion and query pipelines."""

    graph_store: BaseGraphStore
    vector_store: BaseVectorStore
    chunk_lookup: dict[str, Chunk] = field(default_factory=dict)

    def stats(self) -> dict[str, int]:
        return {
            "chunks": len(self.chunk_lookup),
            "triples": len(self.graph_store.all_triples()),
        }


# ---------------------------------------------------------------------------
# IngestionPipeline
# ---------------------------------------------------------------------------
def _reattribute(triples: list[Triple], chunk_id: str) -> list[Triple]:
    """Re-stamp cached triples so they cite the chunk they are handed to.

    A cache hit means the text was extracted under a *different* chunk id (an
    overlapping duplicate, or a segment a repair left untouched while
    renumbering its neighbours). Without re-stamping, the graph would carry
    provenance pointing at a chunk that may no longer exist, and retrieval-time
    citation would silently resolve to the wrong passage.
    """
    return [
        t if t.source_chunk_id == chunk_id else t.model_copy(update={"source_chunk_id": chunk_id})
        for t in triples
    ]


class IngestionPipeline:
    """documents -> chunks -> triples -> (graph_store, vector_store).

    When the resolved chunker also implements :class:`BaseChunkAuditor`,
    extraction becomes a loop rather than a single pass: the extractor's output
    is audited for typed defects (D1–D6), the repair-driving ones re-cut the
    chunk boundaries, and only the re-cut segments are extracted again. This
    runs at most ``audit_rounds`` times and stops early as soon as a round
    yields no actionable defect. Chunkers that do not opt in keep the legacy
    single-pass behaviour bit for bit.
    """

    def __init__(
        self,
        *,
        chunker: BaseChunker,
        extractor: BaseTripleExtractor,
        pruner: BasePruner,
        embedder: BaseEmbeddingProvider,
        state: PipelineState,
        extract_workers: int = 1,
        show_progress: bool = True,
        audit_rounds: int = 2,
    ) -> None:
        self.chunker = chunker
        self.extractor = extractor
        self.pruner = pruner
        self.embedder = embedder
        self.state = state
        # Concurrency for the per-chunk LLM extraction step. With slow local
        # models (e.g. ollama 8B) the ingest bottleneck is many sequential
        # LLM calls; running them in a thread pool cuts wall-clock roughly
        # linearly until the LLM backend itself saturates. 1 == legacy serial.
        self.extract_workers = max(1, int(extract_workers))
        # Render a tqdm progress bar over the per-chunk LLM extraction loop —
        # the slowest, most opaque part of ingestion. Disabled automatically
        # when tqdm is unavailable so the core stays dependency-soft.
        self.show_progress = show_progress
        # k_max for the audit/repair loop. 0 disables it, which is also the
        # E2 ablation switch for "PrismRAG without the closed loop".
        self.audit_rounds = max(0, int(audit_rounds))

    # ------------------------------------------------------------------
    def _progress(self, iterable: Any, *, total: int, desc: str) -> Any:
        """Wrap *iterable* in a tqdm bar when progress is enabled.

        Falls back to the bare iterable if progress is disabled or tqdm is not
        installed, keeping ``core`` usable without the optional dependency.
        """
        if not self.show_progress:
            return iterable
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover - tqdm is a declared dependency
            return iterable
        return tqdm(iterable, total=total, desc=desc, unit="chunk", dynamic_ncols=True)

    def run(self, documents: list[Document]) -> dict[str, Any]:
        all_chunks: list[Chunk] = []
        for doc in self._progress(documents, total=len(documents), desc="切分文档"):
            all_chunks.extend(self.chunker.chunk(doc))

        all_chunks, triples_by_chunk, defects, rounds, repairs = self._extract_with_audit(
            all_chunks
        )

        # Populated after the loop: a repair renumbers chunk ids, so binding
        # lookups earlier would leave dangling entries for pre-repair ids.
        for ch in all_chunks:
            self.state.chunk_lookup[ch.chunk_id] = ch

        logger.info(
            "ingested %d documents -> %d chunks (%d audit rounds, %d repairs)",
            len(documents),
            len(all_chunks),
            rounds,
            repairs,
        )

        # Granularity routing: `fine` chunks never reach the vector index and
        # `coarse` chunks never reach the extractor. `unified` — the default
        # every baseline chunker emits — reaches both, so for them this is a
        # no-op and the ablation stays fair.
        to_index = [c for c in all_chunks if c.granularity in INDEXED_GRANULARITIES]

        if to_index:
            vecs = self.embedder.encode([c.text for c in to_index])
            self.state.vector_store.add([c.chunk_id for c in to_index], vecs)

        raw_triples = [
            t for ch in all_chunks for t in triples_by_chunk.get(ch.chunk_id, [])
        ]

        pruned_triples = self.pruner.prune(raw_triples)
        for t in pruned_triples:
            self.state.graph_store.add_triple(t)

        logger.info(
            "extracted %d raw triples -> %d after pruning", len(raw_triples), len(pruned_triples)
        )

        return {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "triples_raw": len(raw_triples),
            "triples_pruned": len(pruned_triples),
            "audit_rounds": rounds,
            "repairs_applied": repairs,
            "defects": len(defects),
            "defects_by_code": dict(Counter(d.code for d in defects)),
        }

    # ------------------------------------------------------------------
    def _extract_with_audit(
        self, chunks: list[Chunk]
    ) -> tuple[list[Chunk], dict[str, list[Triple]], list[ChunkDefect], int, int]:
        """Run extraction, optionally looping through audit + boundary repair.

        Returns the final chunk list (post-repair), the triples keyed by the
        final chunk ids, every defect seen across all rounds, the number of
        repair rounds actually performed, and the number of repairs applied.
        """
        auditor = self.chunker if isinstance(self.chunker, BaseChunkAuditor) else None
        by_text: dict[str, list[Triple]] = {}
        triples_by_chunk: dict[str, list[Triple]] = {}
        all_defects: list[ChunkDefect] = []
        rounds = repairs = 0

        while True:
            triples_by_chunk = self._extract_new(chunks, by_text, reuse=rounds > 0)
            if auditor is None or rounds >= self.audit_rounds:
                break

            found = auditor.audit(chunks, triples_by_chunk, self.state.graph_store)
            all_defects.extend(found)
            # Risk R3: a repair loop amplifies detector noise, so defects the
            # detector itself could not falsify are recorded but never acted on.
            driving = [d for d in found if d.repair_driving]
            if not driving:
                break

            outcome = auditor.repair(driving, chunks)
            if outcome.defects_applied == 0 or not outcome.chunks:
                break

            chunks = outcome.chunks
            rounds += 1
            repairs += outcome.defects_applied

        return chunks, triples_by_chunk, all_defects, rounds, repairs

    def _extract_new(
        self, chunks: list[Chunk], by_text: dict[str, list[Triple]], *, reuse: bool
    ) -> dict[str, list[Triple]]:
        """Extract *chunks*, reusing earlier rounds' output where the text matches.

        The cache is keyed by text, not id, because a repair renumbers ids: text
        identity is the only stable way to tell a re-cut segment from an
        untouched one, which is what keeps round 2's cost near the marginal
        segments instead of the whole corpus (risk R2).

        Reuse is off on the first round so that a chunker emitting two chunks
        with identical text — routine once overlap is involved — still pays for
        two extractions, exactly as the single-pass pipeline always did.
        """
        pending = [
            ch
            for ch in chunks
            if ch.granularity in EXTRACTED_GRANULARITIES
            and not (reuse and ch.text in by_text)
        ]
        if pending:
            fresh = self._extract(pending)
            for ch in pending:
                by_text.setdefault(ch.text, fresh.get(ch.chunk_id, []))

        return {
            ch.chunk_id: _reattribute(by_text.get(ch.text, []), ch.chunk_id)
            for ch in chunks
            if ch.granularity in EXTRACTED_GRANULARITIES
        }

    def _extract(self, chunks: list[Chunk]) -> dict[str, list[Triple]]:
        """Extract triples for *chunks*, keyed by chunk id in chunk order.

        Each chunk triggers one (slow) LLM call, so we fan out across a
        thread pool when ``extract_workers`` > 1. ``ThreadPoolExecutor.map``
        preserves input order, so the dict — and therefore any triple sequence
        derived from it — follows chunk order exactly as the legacy serial
        path did.
        """
        if not chunks:
            return {}

        per_chunk: dict[str, list[Triple]] = {}
        workers = min(self.extract_workers, len(chunks))
        if workers <= 1:
            for ch in self._progress(chunks, total=len(chunks), desc="抽取三元组"):
                per_chunk[ch.chunk_id] = self.extractor.extract(ch)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                mapped = pool.map(self.extractor.extract, chunks)
                for ch, triples in zip(
                    chunks,
                    self._progress(mapped, total=len(chunks), desc="抽取三元组"),
                    strict=True,
                ):
                    per_chunk[ch.chunk_id] = triples

        return per_chunk



# ---------------------------------------------------------------------------
# QueryPipeline
# ---------------------------------------------------------------------------
class QueryPipeline:
    """query -> (intent) -> retrieve -> generate."""

    def __init__(
        self,
        *,
        intent_classifier: BaseIntentClassifier,
        retriever: BaseRetriever,
        generator: BaseAnswerGenerator,
        top_k: int = 5,
    ) -> None:
        self.intent_classifier = intent_classifier
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    # ------------------------------------------------------------------
    def run(self, query: Query) -> Answer:
        intent = self.intent_classifier.classify(query)
        logger.debug("query %r classified as %s (conf=%.2f)", query.text, intent.label, intent.confidence)

        retrieval = self.retriever.retrieve(query, top_k=self.top_k)
        answer = self.generator.generate(query, retrieval)

        # Enrich trace with intent + retrieval provenance so downstream
        # UIs and examples can inspect the decision without re-running.
        answer.intent_label = intent.label
        answer.trace.setdefault("intent_confidence", intent.confidence)
        answer.trace["retrieval_metadata"] = dict(retrieval.metadata)
        answer.strategy_name = retrieval.metadata.get("strategy")
        return answer


__all__ = ["IngestionPipeline", "PipelineState", "QueryPipeline"]

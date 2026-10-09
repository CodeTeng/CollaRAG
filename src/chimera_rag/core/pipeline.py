"""Ingestion and Query pipelines — the dependency-injection heart of
Chimera-RAG.

Pipelines hold references to the pluggable slots resolved from the
Registry at construction time; they never import concrete plugin
classes themselves. This keeps ``core`` decoupled from ``defaults`` and
``plugins`` as required by the project design.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from chimera_rag.core.types import Answer, Chunk, Document, Query, Triple
from chimera_rag.interfaces.chunker import BaseChunker
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
class IngestionPipeline:
    """documents -> chunks -> triples -> (graph_store, vector_store)."""

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
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
            for ch in chunks:
                self.state.chunk_lookup[ch.chunk_id] = ch

        logger.info("ingested %d documents -> %d chunks", len(documents), len(all_chunks))

        # Embed chunks and push to vector store.
        if all_chunks:
            vecs = self.embedder.encode([c.text for c in all_chunks])
            self.state.vector_store.add([c.chunk_id for c in all_chunks], vecs)

        # Extract + prune triples. Each chunk triggers one (slow) LLM call,
        # so we fan out across a thread pool when extract_workers > 1.
        # ThreadPoolExecutor.map preserves input order, so the resulting
        # triple order is identical to the legacy serial path.
        raw_triples: list[Triple] = []
        if all_chunks:
            workers = min(self.extract_workers, len(all_chunks))
            if workers <= 1:
                for ch in self._progress(
                    all_chunks, total=len(all_chunks), desc="抽取三元组"
                ):
                    raw_triples.extend(self.extractor.extract(ch))
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    mapped = pool.map(self.extractor.extract, all_chunks)
                    for triples in self._progress(
                        mapped, total=len(all_chunks), desc="抽取三元组"
                    ):
                        raw_triples.extend(triples)

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
        }


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

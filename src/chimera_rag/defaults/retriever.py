"""Direct retriever — vanilla baseline.

Pipeline:

1. embed the query,
2. FAISS top-k nearest chunks,
3. optionally expand by ``neighbor_hops`` on the graph around entities
   mentioned in the top chunks.
"""

from __future__ import annotations

import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Query, RetrievalResult, Triple
from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider
from chimera_rag.interfaces.graph_store import BaseGraphStore
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


@register("retriever", "defaults.direct")
class DirectRetriever(BaseRetriever):
    """Vector-similarity retrieval + light graph expansion."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
        embedder: BaseEmbeddingProvider,
        chunk_lookup: dict[str, Chunk] | None = None,
        neighbor_hops: int = 1,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.chunk_lookup: dict[str, Chunk] = (
            chunk_lookup if chunk_lookup is not None else {}
        )
        self.neighbor_hops = neighbor_hops

    # ------------------------------------------------------------------
    def retrieve(self, query: Query, top_k: int = 5) -> RetrievalResult:
        # 1. embed
        qvec = self.embedder.encode([query.text])[0]
        # 2. search
        hits = self.vector_store.search(qvec, top_k=top_k)
        chunks: list[Chunk] = []
        scores: list[float] = []
        for chunk_id, score in hits:
            ch = self.chunk_lookup.get(chunk_id)
            if ch is None:
                logger.debug("retriever: vector hit id %r not in chunk_lookup", chunk_id)
                continue
            chunks.append(ch)
            scores.append(score)

        # 3. optional graph expansion
        triples: list[Triple] = []
        if self.neighbor_hops > 0 and chunks:
            entities = _extract_candidate_entities(chunks, query.text)
            seen_triples: set[tuple[str, str, str]] = set()
            for e in entities:
                for t in self.graph_store.get_neighbors(e, hops=self.neighbor_hops):
                    key = (t.subject, t.predicate, t.object)
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    triples.append(t)

        return RetrievalResult(
            chunks=chunks,
            triples=triples,
            scores=scores,
            metadata={"neighbor_hops": self.neighbor_hops, "entity_seeds": []},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CAPITALISED_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def _extract_candidate_entities(chunks: list[Chunk], query_text: str) -> list[str]:
    """Pick capitalised tokens from the query + top chunks as seed entities.

    This is a dumb heuristic suitable for the vanilla baseline. AdaGraph /
    CollaRAG use smarter entity detection via the triple store itself.
    """
    pool = query_text + "\n" + "\n".join(c.text for c in chunks)
    seen: set[str] = set()
    ordered: list[str] = []
    for m in _CAPITALISED_RE.finditer(pool):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return ordered


__all__ = ["DirectRetriever"]

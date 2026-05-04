"""Parallel strategy: split on 'and', retrieve each sub-query concurrently.

Used by ``comparative`` intents. Splits the query string at ``' and '``
(case-insensitive), fires retrievals in asyncio.gather, then merges
chunks by de-duplicating on ``chunk_id`` while keeping the higher score.
"""

from __future__ import annotations

import asyncio
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Query, ReasoningPlan, RetrievalResult
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.strategy import BaseRetrievalStrategy

_SPLIT_RE = re.compile(r"\s+(?:and|vs\.?|versus|compared to)\s+", re.I)


@register("strategy", "intent_agent.parallel")
class ParallelStrategy(BaseRetrievalStrategy):
    def __init__(self, retriever: BaseRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k

    async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult:
        parts = [p.strip() for p in _SPLIT_RE.split(query.text) if p.strip()]
        if len(parts) < 2:
            # No split possible — behave like direct.
            res = self.retriever.retrieve(query, top_k=self.top_k)
            res.metadata["strategy"] = "parallel"
            res.metadata["parallel_subqueries"] = 1
            return res

        async def _retrieve_one(sub: str) -> RetrievalResult:
            return self.retriever.retrieve(Query(text=sub), top_k=self.top_k)

        sub_results = await asyncio.gather(*[_retrieve_one(p) for p in parts])

        merged_chunks: dict[str, tuple[Chunk, float]] = {}
        for sub_res in sub_results:
            for ch, sc in zip(sub_res.chunks, sub_res.scores + [0.0] * len(sub_res.chunks)):
                existing = merged_chunks.get(ch.chunk_id)
                if existing is None or sc > existing[1]:
                    merged_chunks[ch.chunk_id] = (ch, sc)

        chunks = [c for c, _ in merged_chunks.values()]
        scores = [s for _, s in merged_chunks.values()]
        # Sort descending by score for consistent output.
        paired = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
        chunks = [c for c, _ in paired]
        scores = [s for _, s in paired]

        triples = []
        seen = set()
        for sub_res in sub_results:
            for t in sub_res.triples:
                key = (t.subject, t.predicate, t.object)
                if key not in seen:
                    seen.add(key)
                    triples.append(t)

        return RetrievalResult(
            chunks=chunks,
            triples=triples,
            scores=scores,
            metadata={"strategy": "parallel", "parallel_subqueries": len(parts)},
        )


__all__ = ["ParallelStrategy"]

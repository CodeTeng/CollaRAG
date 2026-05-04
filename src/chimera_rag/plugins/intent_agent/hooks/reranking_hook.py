"""Token-overlap reranking hook (no cross-encoder required).

Reorders :class:`RetrievalResult` chunks by simple token Jaccard against
the query. This is a dependency-free stand-in for a real cross-encoder
reranker; the interface is the same so swapping in sentence-transformers
later is one class change.
"""

from __future__ import annotations

import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, RetrievalResult
from chimera_rag.interfaces.reranker import BaseReranker

_WORD_RE = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


@register("reranker", "intent_agent.token_overlap")
class TokenOverlapRerankerHook(BaseReranker):
    """Reorder chunks by Jaccard overlap with the query."""

    def __init__(self, top_n: int | None = None) -> None:
        self.top_n = top_n

    async def rerank(self, query: Query, retrieval: RetrievalResult) -> RetrievalResult:
        q_toks = _tokens(query.text)
        if not q_toks or not retrieval.chunks:
            return retrieval

        scored = []
        for ch in retrieval.chunks:
            c_toks = _tokens(ch.text)
            union = q_toks | c_toks
            jac = len(q_toks & c_toks) / len(union) if union else 0.0
            scored.append((ch, jac))

        scored.sort(key=lambda p: p[1], reverse=True)
        if self.top_n is not None:
            scored = scored[: self.top_n]

        chunks = [c for c, _ in scored]
        scores = [s for _, s in scored]
        return RetrievalResult(
            chunks=chunks,
            triples=retrieval.triples,
            scores=scores,
            metadata={**retrieval.metadata, "reranked_by": "token_overlap"},
        )


__all__ = ["TokenOverlapRerankerHook"]

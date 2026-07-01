"""Cross-encoder reranker backed by ``sentence-transformers``.

Loads a :class:`sentence_transformers.CrossEncoder` model (e.g.
``cross-encoder/ms-marco-MiniLM-L-6-v2``) and scores each
``(query, chunk)`` pair to produce a relevance-ranked
:class:`RetrievalResult`.

Requires the ``reranker`` optional extra (``uv sync --extra reranker``).
If ``sentence-transformers`` is not installed, the constructor raises
:class:`ImportError` with an actionable install hint.

Registered as ``('reranker', 'intent_agent.cross_encoder')``.
"""

from __future__ import annotations

import logging

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, RetrievalResult
from chimera_rag.interfaces.reranker import BaseReranker

logger = logging.getLogger(__name__)


def _require_cross_encoder() -> type:
    """Lazy import that raises a helpful error when the extra is missing."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "CrossEncoder reranker requires the 'reranker' extra. "
            "Install it with: uv sync --extra reranker"
        ) from exc


@register("reranker", "intent_agent.cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """Cross-encoder model reranker for retrieval results."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int | None = None,
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        CrossEncoder = _require_cross_encoder()
        logger.info("cross_encoder_reranker: loading model %s", model_name)
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.top_n = top_n
        self._model_name = model_name

    async def rerank(self, query: Query, retrieval: RetrievalResult) -> RetrievalResult:
        if not retrieval.chunks:
            return retrieval

        pairs = [(query.text, chunk.text) for chunk in retrieval.chunks]
        scores = self.model.predict(pairs)

        # Pair chunks with their cross-encoder scores, then sort descending.
        scored = list(zip(retrieval.chunks, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        if self.top_n is not None:
            scored = scored[: self.top_n]

        chunks = [c for c, _ in scored]
        rerank_scores = [float(s) for _, s in scored]
        return RetrievalResult(
            chunks=chunks,
            triples=retrieval.triples,
            scores=rerank_scores,
            metadata={**retrieval.metadata, "reranked_by": f"cross_encoder/{self._model_name}"},
        )


__all__ = ["CrossEncoderReranker"]

"""Heuristic quality evaluator — lightweight, no LLM required.

Scores an answer by: length, refusal detection, query token overlap,
evidence presence. Used as the default reflector in MultiAgentRetriever
when no explicit reflector is provided.
"""
from __future__ import annotations

import re

from chimera_rag.core.types import Answer, QualityFeedback, Query

_REFUSAL_RE = re.compile(
    r"\b(i (?:do not|don'?t) know|not enough (?:context|information)|unknown)\b", re.I
)
_WORD_RE = re.compile(r"\w+")


class HeuristicQualityEvaluator:
    """Non-LLM, deterministic quality estimator."""

    def __init__(self, min_length: int = 10) -> None:
        self.min_length = min_length

    async def evaluate(self, query: Query, answer: Answer) -> QualityFeedback:
        text = answer.text.strip()
        if not text:
            return QualityFeedback(quality=0.0, relevance=0.0, completeness=0.0,
                                   suggestion="empty answer")

        if _REFUSAL_RE.search(text):
            return QualityFeedback(
                quality=0.2, relevance=0.3, completeness=0.1,
                suggestion="answer refuses; re-retrieve or widen strategy",
            )

        length_score = min(1.0, len(text) / max(self.min_length, 1) / 4)

        q_tokens = {t.lower() for t in _WORD_RE.findall(query.text)}
        a_tokens = {t.lower() for t in _WORD_RE.findall(text)}
        overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))

        has_evidence = 1.0 if answer.evidence_chunk_ids else 0.5

        relevance = float(0.5 * overlap + 0.5 * has_evidence)
        completeness = float(length_score)
        quality = float(0.5 * relevance + 0.5 * completeness)

        return QualityFeedback(
            quality=max(0.0, min(1.0, quality)),
            relevance=max(0.0, min(1.0, relevance)),
            completeness=max(0.0, min(1.0, completeness)),
            suggestion=None,
        )


__all__ = ["HeuristicQualityEvaluator"]

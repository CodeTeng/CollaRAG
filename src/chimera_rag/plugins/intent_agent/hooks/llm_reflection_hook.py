"""LLM-backed reflection hook — Self-Reflection mechanism (innovation #2d).

Produces a verbal (language) self-critique alongside the numeric quality
score, enabling Reflexion-style self-correction: the agent reads its own
reflection text and adjusts its next planning round.

Compared to :class:`HeuristicReflectionHook` (rule-based, no language
output), this hook:

1. Calls the shared ``BaseLLMProvider`` to generate a structured critique.
2. Returns ``QualityFeedback.reflection_text`` — a natural-language
   diagnosis of what went wrong and how to fix it.
3. Still computes a numeric ``quality`` score from simple signals (length,
   token overlap, refusal phrases) so the loop's conditional edge can
   decide whether to stop or re-plan.

The LLM prompt is deliberately terse (single paragraph, < 200 words) to
keep latency and token cost low. Under ``MockProvider`` the hook falls back
to the heuristic scorer so tests stay deterministic and zero-cost.

Registered as ``('reflector', 'intent_agent.llm_reflection')``.
"""

from __future__ import annotations

import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Answer, QualityFeedback, Query
from chimera_rag.interfaces.llm_provider import BaseLLMProvider
from chimera_rag.interfaces.reflector import BaseReflector

logger = logging.getLogger(__name__)

_REFUSAL_RE = re.compile(
    r"\b(i (?:do not|don'?t) know|not enough (?:context|information)|unknown)\b", re.I
)
_WORD_RE = re.compile(r"\w+")

_REFLECTION_PROMPT_TEMPLATE = """\
You are a retrieval quality evaluator. Given the question and the draft answer,
write a **single paragraph** (< 200 words) critique that:

1. Identifies what is missing or wrong in the answer.
2. Suggests which retrieval strategy or tool would fix it.

Question: {query}
Draft answer: {answer}
Evidence chunks used: {evidence_count}

Critique:"""


@register("reflector", "intent_agent.llm_reflection")
class LLMReflectionHook(BaseReflector):
    """LLM-driven self-reflection: numeric score + verbal critique."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        min_length: int = 10,
        max_reflection_words: int = 200,
    ) -> None:
        self.llm = llm
        self.min_length = min_length
        self.max_reflection_words = max_reflection_words

    # ------------------------------------------------------------------
    # Numeric scoring (same signals as HeuristicReflectionHook, kept fast)
    # ------------------------------------------------------------------
    def _numeric_score(self, query: Query, answer: Answer) -> tuple[float, float, float]:
        text = answer.text.strip()
        if not text:
            return 0.0, 0.0, 0.0

        # Refusal signal.
        if _REFUSAL_RE.search(text):
            return 0.2, 0.3, 0.1

        # Length signal.
        length_score = min(1.0, len(text) / max(self.min_length, 1) / 4)

        # Token overlap signal.
        q_tokens = {t.lower() for t in _WORD_RE.findall(query.text)}
        a_tokens = {t.lower() for t in _WORD_RE.findall(text)}
        overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))

        # Evidence signal.
        has_evidence = 1.0 if answer.evidence_chunk_ids else 0.5

        relevance = float(0.5 * overlap + 0.5 * has_evidence)
        completeness = float(length_score)
        quality = float(0.5 * relevance + 0.5 * completeness)
        return quality, relevance, completeness

    # ------------------------------------------------------------------
    # Verbal reflection via LLM
    # ------------------------------------------------------------------
    async def _generate_reflection_text(self, query: Query, answer: Answer) -> str | None:
        """Call the LLM to produce a verbal critique. Returns None on error."""
        prompt = _REFLECTION_PROMPT_TEMPLATE.format(
            query=query.text,
            answer=answer.text[:500],  # truncate to keep prompt short
            evidence_count=len(answer.evidence_chunk_ids),
        )
        try:
            raw = await self.llm.complete(prompt)
        except Exception as exc:
            logger.warning("LLMReflectionHook: LLM call failed (%s); skipping verbal reflection", exc)
            return None

        # Truncate to max words to keep reflection concise.
        words = raw.strip().split()
        if len(words) > self.max_reflection_words:
            raw = " ".join(words[:self.max_reflection_words])
        return raw.strip() or None

    # ------------------------------------------------------------------
    # BaseReflector interface
    # ------------------------------------------------------------------
    async def evaluate(self, query: Query, answer: Answer) -> QualityFeedback:
        quality, relevance, completeness = self._numeric_score(query, answer)

        # Only ask the LLM for a verbal critique when quality is below
        # threshold — high-quality answers don't need reflection.
        reflection_text: str | None = None
        suggestion: str | None = None

        if quality < 0.7:
            reflection_text = await self._generate_reflection_text(query, answer)
            if reflection_text:
                # Extract first sentence as a short suggestion for metadata.
                suggestion = reflection_text.split(".")[0].strip() + "."
            elif _REFUSAL_RE.search(answer.text):
                suggestion = "answer refuses; re-retrieve or widen strategy"

        return QualityFeedback(
            quality=max(0.0, min(1.0, quality)),
            relevance=max(0.0, min(1.0, relevance)),
            completeness=max(0.0, min(1.0, completeness)),
            suggestion=suggestion,
            reflection_text=reflection_text,
        )


__all__ = ["LLMReflectionHook"]

"""IntentAgent's 6-way intent classifier with LLM + rule fallback.

Intent taxonomy (from the design doc):

================ =====================================================
Label            Typical surface forms
================ =====================================================
factual          "who/what/when/where invented/discovered …"
analytical       "why/how did X happen", cause-effect queries
comparative      "compare A and B", "difference between X and Y"
multi_hop        "impact / influence / relation between X and Y"
exploratory      "tell me about …", "explain …", very open-ended
follow_up        anaphoric queries referring to prior turns
                 ("what about her?", "and then?")
================ =====================================================

Strategy:
1. Ask the LLM for a JSON blob ``{"label": ..., "confidence": ...}``.
2. If the response parses *and* confidence >= threshold *and* the label
   is one of the six canonical strings, return it.
3. Otherwise fall back to :func:`rule_based_classify` which applies a
   hand-tuned regex cascade.
"""

from __future__ import annotations

import json
import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Intent, Query
from chimera_rag.defaults.llm_extractor import _run_async
from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

CANONICAL_LABELS = {
    "factual",
    "analytical",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
}


CLASSIFY_PROMPT = """Classify the user question into exactly one of six intents:
factual, analytical, comparative, multi_hop, exploratory, follow_up.

Reply with a single JSON object with keys "label" (one of the six) and
"confidence" (a number between 0 and 1). Do not add any other text.

Question: {text}
"""


# ---------------------------------------------------------------------------
# Rule-based fallback: order matters (more specific rules first).
# ---------------------------------------------------------------------------
# "compare", "difference between", "vs" -> comparative
_RULE_COMPARATIVE = re.compile(r"\b(compare|comparison|vs\.?|difference between)\b", re.I)
# "impact", "effect", "influence", "relation" -> multi_hop (often requires bridging entities)
_RULE_MULTI_HOP = re.compile(
    r"\b(impact|effect|influence|relation|between\s+\w+\s+and\s+\w+)\b", re.I
)
# "why", "how come" -> analytical (cause-effect)
_RULE_ANALYTICAL = re.compile(r"\b(why|how come|explain why)\b", re.I)
# "tell me about", "explain", open-ended "describe" -> exploratory
_RULE_EXPLORATORY = re.compile(r"\b(tell me about|explain|describe|overview)\b", re.I)
# pronouns at start or alone -> follow_up
_RULE_FOLLOW_UP = re.compile(
    r"^(and|then|what about|how about)\b"
    r"|\b(her|him|them|it|this|that|these|those)\b\s*\??$",
    re.I,
)
# factual wh-questions
_RULE_FACTUAL = re.compile(r"\b(who|what|when|where|which)\b", re.I)


def rule_based_classify(query: Query) -> Intent:
    """Cascade of regexes; fall through to 'factual' as the safest default."""
    q = query.text.strip()
    # Check specific-first.
    if _RULE_COMPARATIVE.search(q):
        return Intent(label="comparative", confidence=0.6, rationale="rule: compare/vs")
    if _RULE_MULTI_HOP.search(q):
        return Intent(label="multi_hop", confidence=0.6, rationale="rule: impact/relation")
    if _RULE_ANALYTICAL.search(q):
        return Intent(label="analytical", confidence=0.6, rationale="rule: why/how")
    if _RULE_FOLLOW_UP.search(q):
        return Intent(label="follow_up", confidence=0.55, rationale="rule: anaphora")
    if _RULE_EXPLORATORY.search(q):
        return Intent(label="exploratory", confidence=0.55, rationale="rule: open-ended")
    if _RULE_FACTUAL.search(q):
        return Intent(label="factual", confidence=0.6, rationale="rule: wh-question")
    return Intent(label="factual", confidence=0.4, rationale="rule: default")


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _safe_parse_label(raw: str) -> tuple[str, float] | None:
    """Try to recover (label, confidence) from the LLM output."""
    fence_match = _FENCE_RE.search(raw)
    text = fence_match.group(1).strip() if fence_match else raw

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # fallback: greedy first-brace slice
        l, r = text.find("{"), text.rfind("}")
        if l == -1 or r == -1 or r <= l:
            return None
        try:
            obj = json.loads(text[l : r + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None
    label = obj.get("label")
    conf = obj.get("confidence", 0.0)
    if not isinstance(label, str):
        return None
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    return label, max(0.0, min(1.0, conf))


@register("intent_classifier", "intent_agent.llm")
class LLMIntentClassifier(BaseIntentClassifier):
    """LLM-driven 6-way classifier with rule fallback."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        confidence_threshold: float = 0.6,
        prompt_template: str = CLASSIFY_PROMPT,
    ) -> None:
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.prompt_template = prompt_template

    # ------------------------------------------------------------------
    def classify(self, query: Query) -> Intent:
        prompt = self.prompt_template.format(text=query.text)
        try:
            raw = _run_async(self.llm.complete(prompt))
        except Exception as e:  # pragma: no cover - provider errors
            logger.warning("intent_classifier: LLM call failed: %s", e)
            return rule_based_classify(query)

        parsed = _safe_parse_label(raw)
        if parsed is None:
            logger.debug("intent_classifier: unparseable LLM output -> rules")
            return rule_based_classify(query)

        label, confidence = parsed
        if label not in CANONICAL_LABELS:
            logger.debug("intent_classifier: unknown label %r -> rules", label)
            return rule_based_classify(query)
        if confidence < self.confidence_threshold:
            logger.debug(
                "intent_classifier: confidence %.2f below threshold %.2f -> rules",
                confidence,
                self.confidence_threshold,
            )
            return rule_based_classify(query)

        return Intent(label=label, confidence=confidence, rationale="llm")


__all__ = [
    "CANONICAL_LABELS",
    "CLASSIFY_PROMPT",
    "LLMIntentClassifier",
    "rule_based_classify",
]

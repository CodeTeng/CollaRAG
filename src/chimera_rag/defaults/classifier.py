"""Rule-based intent classifier — vanilla baseline (always factual)."""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Intent, Query
from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier


@register("intent_classifier", "defaults.rule")
class RuleBasedIntentClassifier(BaseIntentClassifier):
    """Dumbest useful classifier: every query is 'factual'.

    CollaRAG replaces this with a genuine 6-way classifier.
    """

    def classify(self, query: Query) -> Intent:
        return Intent(label="factual", confidence=0.5, rationale="rule-based default")


__all__ = ["RuleBasedIntentClassifier"]

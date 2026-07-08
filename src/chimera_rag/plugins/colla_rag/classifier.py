"""Tree-shaped intent classifier: Level-1 greeting filter + Level-2 LLM classification."""
from __future__ import annotations

import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Intent, Query
from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier
from chimera_rag.plugins.colla_rag.prompts import CLASSIFY_PROMPT

logger = logging.getLogger(__name__)

_VALID_LABELS = {"single_hop", "multi_hop", "summarization", "other"}

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|你好|嗨|哈喽|good\s*(morning|afternoon|evening)|howdy|greetings)"
    r"[!?.,;：！？。，\s]*$",
    re.IGNORECASE,
)
_GREETING_CONTAINS = re.compile(
    r"^(hi|hello|hey|你好|嗨)\b.*\b(how are you|how's it going|what's up|怎么样|吗)\b",
    re.IGNORECASE,
)

_MULTI_HOP_KW = re.compile(
    r"\b(compar[ei]|contrast|relationship|differ|versus|vs\.?|between .+ and"
    r"|how .+ relate|why .+ and .+|explore|investigat)"
    r"\b",
    re.IGNORECASE,
)
_SUMMARIZE_KW = re.compile(
    r"\b(summariz\w*|总结|概括|归纳|overview|synopsis|recap|key (findings|points|takeaways))\b",
    re.IGNORECASE,
)


@register("intent_classifier", "colla_rag.tree")
class TreeIntentClassifier(BaseIntentClassifier):
    def __init__(self, llm: object, confidence_threshold: float = 0.7,
                 rule_fallback: bool = True) -> None:
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.rule_fallback = rule_fallback

    def classify(self, query: Query) -> Intent:
        text = query.text.strip()

        if self._is_greeting(text):
            return Intent(label="greeting", confidence=0.95, rationale="rule:greeting")

        label = self._llm_classify(text)
        if label in _VALID_LABELS:
            return Intent(label=label, confidence=self.confidence_threshold, rationale="llm")

        if self.rule_fallback:
            label = self._rule_classify(text)
            return Intent(label=label, confidence=0.5, rationale="rule:fallback")

        return Intent(label="single_hop", confidence=0.3, rationale="default")

    @staticmethod
    def _is_greeting(text: str) -> bool:
        if _GREETING_PATTERNS.match(text):
            return True
        return bool(_GREETING_CONTAINS.match(text))

    def _llm_classify(self, text: str) -> str:
        from chimera_rag.defaults.extractor import _run_async

        prompt = CLASSIFY_PROMPT.format(query=text)
        try:
            raw = _run_async(self.llm.complete(prompt, max_tokens=20))
            label = raw.strip().lower().replace('"', "").replace("'", "")
            for candidate in _VALID_LABELS:
                if candidate in label:
                    return candidate
            return label
        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
            return "unknown"

    @staticmethod
    def _rule_classify(text: str) -> str:
        if _SUMMARIZE_KW.search(text):
            return "summarization"
        if _MULTI_HOP_KW.search(text):
            return "multi_hop"
        return "single_hop"


__all__ = ["TreeIntentClassifier"]

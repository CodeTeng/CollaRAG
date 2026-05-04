"""Tests for :mod:`chimera_rag.plugins.intent_agent.intent_classifier`.

The LLM-based 6-way intent classifier must:

* parse JSON ``{"label": ..., "confidence": ...}`` from the LLM
* coerce the label into the 6 canonical intents
  (factual / analytical / comparative / multi_hop / exploratory / follow_up)
* gracefully fall back to a rule-based classifier when the LLM output is
  unparseable or below a configured confidence threshold
"""

from __future__ import annotations


def test_llm_intent_classifier_implements_abc():
    from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    assert isinstance(LLMIntentClassifier(llm=MockLLMProvider()), BaseIntentClassifier)


def test_llm_intent_classifier_returns_parsed_label():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    mock = MockLLMProvider(default_response='{"label": "comparative", "confidence": 0.92}')
    c = LLMIntentClassifier(llm=mock)
    intent = c.classify(Query(text="compare A and B"))
    assert intent.label == "comparative"
    assert intent.confidence == 0.92


def test_llm_intent_classifier_falls_back_to_rules_on_parse_error():
    """LLM returns garbage -> fallback to rule-based classification."""
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    mock = MockLLMProvider(default_response="not json")
    c = LLMIntentClassifier(llm=mock)
    intent = c.classify(Query(text="compare A and B"))
    # rule-based picks comparative for queries containing "compare".
    assert intent.label == "comparative"


def test_llm_intent_classifier_falls_back_below_confidence_threshold():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    mock = MockLLMProvider(
        default_response='{"label": "exploratory", "confidence": 0.1}'
    )
    c = LLMIntentClassifier(llm=mock, confidence_threshold=0.6)
    intent = c.classify(Query(text="who invented dynamite?"))
    # "who ..." matches factual rule.
    assert intent.label == "factual"


def test_llm_intent_classifier_rejects_unknown_label():
    """Unknown labels from the LLM must fall back to rules rather than crash."""
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    mock = MockLLMProvider(default_response='{"label": "weirdo", "confidence": 0.9}')
    c = LLMIntentClassifier(llm=mock)
    intent = c.classify(Query(text="hi"))
    assert intent.label in {
        "factual", "analytical", "comparative",
        "multi_hop", "exploratory", "follow_up",
    }


# ---------------------------------------------------------------------------
# Rule-based fallback rules in isolation
# ---------------------------------------------------------------------------
def test_rule_based_fallback_hits_expected_categories():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.intent_classifier import rule_based_classify

    cases = [
        ("Who invented dynamite?",      "factual"),
        ("Compare A and B",             "comparative"),
        ("Why did the revolution happen?", "analytical"),
        ("What was the impact of X on Y?", "multi_hop"),
        ("Tell me about AI",            "exploratory"),
        ("What about her?",             "follow_up"),
    ]
    for q, expected in cases:
        got = rule_based_classify(Query(text=q))
        assert got.label == expected, f"{q!r} -> {got.label!r} (expected {expected!r})"


def test_intent_classifier_registers_under_slot():
    import chimera_rag.plugins.intent_agent  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier

    assert (
        get_registry().get("intent_classifier", "intent_agent.llm")
        is LLMIntentClassifier
    )

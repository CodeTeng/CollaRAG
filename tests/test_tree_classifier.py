"""Tests for TreeIntentClassifier."""
from chimera_rag.core.types import Query
from chimera_rag.plugins.colla_rag.classifier import TreeIntentClassifier


class FakeLLM:
    """Mock LLM that returns a predefined label."""

    def __init__(self, response: str = "single_hop"):
        self.response = response

    async def complete(self, prompt: str, **kwargs) -> str:
        return self.response


def _make_classifier(llm_response: str = "single_hop") -> TreeIntentClassifier:
    return TreeIntentClassifier(llm=FakeLLM(llm_response))


def test_greeting_detected_by_rules():
    c = _make_classifier()
    result = c.classify(Query(text="你好"))
    assert result.label == "greeting"


def test_greeting_english():
    c = _make_classifier()
    result = c.classify(Query(text="Hello, how are you?"))
    assert result.label == "greeting"


def test_single_hop_from_llm():
    c = _make_classifier("single_hop")
    result = c.classify(Query(text="What is the capital of France?"))
    assert result.label == "single_hop"


def test_multi_hop_from_llm():
    c = _make_classifier("multi_hop")
    result = c.classify(Query(text="Compare Einstein and Bohr contributions"))
    assert result.label == "multi_hop"


def test_summarization_from_llm():
    c = _make_classifier("summarization")
    result = c.classify(Query(text="Summarize the key findings of this paper"))
    assert result.label == "summarization"


def test_other_from_llm():
    c = _make_classifier("other")
    result = c.classify(Query(text="What's the weather in Tokyo?"))
    assert result.label == "other"


def test_invalid_llm_output_falls_back_to_rule():
    c = _make_classifier("gibberish_not_a_label")
    result = c.classify(Query(text="What is quantum computing?"))
    assert result.label in ("single_hop", "multi_hop", "summarization", "other")


def test_confidence_range():
    c = _make_classifier("single_hop")
    result = c.classify(Query(text="What is X?"))
    assert 0.0 <= result.confidence <= 1.0


def test_rule_fallback_multi_hop_keywords():
    c = _make_classifier("gibberish")
    q = Query(text="Compare A and B and explain the relationship between them")
    result = c.classify(q)
    assert result.label == "multi_hop"


def test_rule_fallback_summarization_keywords():
    c = _make_classifier("gibberish")
    q = Query(text="Please summarize the following document for me")
    result = c.classify(q)
    assert result.label == "summarization"

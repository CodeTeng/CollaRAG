"""Tests for multi-agent type extensions."""
import pytest

from chimera_rag.core.types import (
    AgentExperienceEntry,
    MultiAgentIntentLabel,
    PreprocessResult,
)


def test_preprocess_result_valid():
    r = PreprocessResult(
        rewritten_query="What is Einstein's birthplace?",
        original_query="Where was he born?",
        is_followup=True,
        entities_mentioned=["Einstein"],
        language="en",
    )
    assert r.rewritten_query == "What is Einstein's birthplace?"
    assert r.is_followup is True


def test_preprocess_result_defaults():
    r = PreprocessResult(
        rewritten_query="hello",
        original_query="hello",
    )
    assert r.is_followup is False
    assert r.entities_mentioned == []
    assert r.language == "en"


def test_agent_experience_entry_valid():
    e = AgentExperienceEntry(
        agent_type="react",
        intent="multi_hop",
        query_text="What is X?",
        tool_sequence=["hybrid_search", "graph_neighbors"],
        reflection_count=1,
        final_quality=0.8,
        outcome="success",
    )
    assert e.agent_type == "react"
    assert e.outcome == "success"
    assert e.hit_count == 0


def test_agent_experience_entry_lesson():
    e = AgentExperienceEntry(
        agent_type="react",
        intent="multi_hop",
        query_text="What is X?",
        tool_sequence=["vector_search"],
        reflection_count=2,
        final_quality=0.3,
        outcome="lesson",
        lesson="vector_search alone is insufficient for multi-hop",
    )
    assert e.outcome == "lesson"
    assert e.lesson is not None


def test_agent_experience_entry_quality_bounds():
    with pytest.raises(ValueError):
        AgentExperienceEntry(
            agent_type="react",
            intent="multi_hop",
            query_text="x",
            tool_sequence=[],
            reflection_count=0,
            final_quality=1.5,
            outcome="success",
        )


def test_multi_agent_intent_label_values():
    valid: list[MultiAgentIntentLabel] = [
        "greeting", "single_hop", "multi_hop", "summarization", "other",
    ]
    assert len(valid) == 5

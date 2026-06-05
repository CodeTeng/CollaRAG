"""Tests for experience write-back + hint rendering (innovation #2c).

Covers build_experience_entry's quality-based classification (success/lesson),
deterministic lesson templating, and render_experience_hint output.
"""

from __future__ import annotations

from chimera_rag.core.types import Chunk, RetrievalResult
from chimera_rag.interfaces.experiential_memory import ExperienceEntry
from chimera_rag.plugins.intent_agent.langgraph.experience import (
    build_experience_entry,
    render_experience_hint,
)


def _result_with_chunks():
    return RetrievalResult(
        chunks=[Chunk(chunk_id="c0", doc_id="d", index=0, text="dynamite fact")],
        scores=[0.9],
    )


def test_high_quality_builds_success_with_qa_pair():
    entry = build_experience_entry(
        intent="multi_hop",
        query_text="who?",
        tool_sequence=["graph_neighbors", "multi_hop_retrieve"],
        iterations=2,
        quality=0.82,
        result=_result_with_chunks(),
        quality_threshold=0.6,
    )
    assert entry.outcome == "success"
    assert entry.answer_text == "dynamite fact"
    assert entry.evidence_chunk_ids == ["c0"]
    assert entry.lesson is None
    assert entry.created_at  # timestamp populated


def test_low_quality_builds_lesson_without_answer():
    entry = build_experience_entry(
        intent="multi_hop",
        query_text="who?",
        tool_sequence=["vector_search"],
        iterations=2,
        quality=0.4,
        result=_result_with_chunks(),
        quality_threshold=0.6,
    )
    assert entry.outcome == "lesson"
    assert entry.answer_text is None  # low-quality answers are not reused
    assert entry.evidence_chunk_ids == []
    assert entry.lesson is not None
    assert "vector_search" in entry.lesson
    assert "multi_hop" in entry.lesson


def test_threshold_boundary_is_success():
    entry = build_experience_entry(
        intent="factual",
        query_text="q",
        tool_sequence=["vector_search"],
        iterations=1,
        quality=0.6,
        result=_result_with_chunks(),
        quality_threshold=0.6,
    )
    assert entry.outcome == "success"


def test_render_hint_empty_returns_none():
    assert render_experience_hint([]) is None


def test_render_hint_includes_success_and_lesson():
    entries = [
        ExperienceEntry(
            intent="multi_hop", query_text="q",
            tool_sequence=["graph_neighbors", "multi_hop_retrieve"],
            final_quality=0.82, outcome="success",
        ),
        ExperienceEntry(
            intent="multi_hop", query_text="q",
            tool_sequence=["vector_search"], final_quality=0.4,
            outcome="lesson", lesson="只用 vector_search 质量不足",
        ),
    ]
    hint = render_experience_hint(entries)
    assert hint is not None
    assert "成功路径" in hint
    assert "graph_neighbors → multi_hop_retrieve" in hint
    assert "规避教训" in hint
    assert "只用 vector_search 质量不足" in hint

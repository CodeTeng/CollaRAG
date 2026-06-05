"""Tests for FileExperientialMemory storage layer (innovation #2c).

Covers remember/recall/stats, JSONL round-trip across instances (persistence),
intent bucketing, de-duplication, and capacity eviction.
"""

from __future__ import annotations

from pathlib import Path

from chimera_rag.interfaces.experiential_memory import (
    BaseExperientialMemory,
    ExperienceEntry,
)
from chimera_rag.plugins.intent_agent.experiential_memory import FileExperientialMemory


def _entry(intent="multi_hop", tools=None, quality=0.8, outcome="success", **kw):
    return ExperienceEntry(
        intent=intent,
        query_text=kw.get("query_text", "q"),
        tool_sequence=tools if tools is not None else ["vector_search"],
        iterations=kw.get("iterations", 1),
        final_quality=quality,
        outcome=outcome,
        lesson=kw.get("lesson"),
        answer_text=kw.get("answer_text"),
        evidence_chunk_ids=kw.get("evidence_chunk_ids", []),
        created_at=kw.get("created_at", "2026-01-01T00:00:00+00:00"),
        hit_count=kw.get("hit_count", 0),
    )


def test_implements_abc(tmp_path: Path):
    mem = FileExperientialMemory(path=str(tmp_path / "m"))
    assert isinstance(mem, BaseExperientialMemory)


def test_registered_under_memory_slot():
    import chimera_rag.plugins.intent_agent  # noqa: F401
    from chimera_rag.core.registry import get_registry

    cls = get_registry().get("memory", "intent_agent.experiential")
    assert cls is FileExperientialMemory


def test_remember_then_recall(tmp_path: Path):
    mem = FileExperientialMemory(path=str(tmp_path / "m"))
    mem.remember(_entry(tools=["graph_neighbors", "multi_hop_retrieve"], quality=0.9))
    recalled = mem.recall("multi_hop", "any query", top_n=3)
    assert len(recalled) == 1
    assert recalled[0].tool_sequence == ["graph_neighbors", "multi_hop_retrieve"]
    assert recalled[0].final_quality == 0.9


def test_persists_across_instances(tmp_path: Path):
    root = str(tmp_path / "m")
    mem1 = FileExperientialMemory(path=root)
    mem1.remember(_entry(tools=["vector_search"], quality=0.7))
    # New instance pointing at same dir must see the persisted entry.
    mem2 = FileExperientialMemory(path=root)
    recalled = mem2.recall("multi_hop", "q", top_n=3)
    assert len(recalled) == 1
    assert recalled[0].tool_sequence == ["vector_search"]


def test_bucketing_isolates_intents(tmp_path: Path):
    mem = FileExperientialMemory(path=str(tmp_path / "m"))
    mem.remember(_entry(intent="multi_hop", tools=["graph_neighbors"]))
    mem.remember(_entry(intent="factual", tools=["vector_search"]))
    assert len(mem.recall("multi_hop", "q")) == 1
    assert len(mem.recall("factual", "q")) == 1
    # Each bucket only sees its own intent.
    assert mem.recall("multi_hop", "q")[0].tool_sequence == ["graph_neighbors"]


def test_recall_ranks_by_quality_then_hits(tmp_path: Path):
    mem = FileExperientialMemory(path=str(tmp_path / "m"))
    mem.remember(_entry(tools=["a"], quality=0.5))
    mem.remember(_entry(tools=["b"], quality=0.9))
    mem.remember(_entry(tools=["c"], quality=0.7))
    recalled = mem.recall("multi_hop", "q", top_n=2)
    assert [e.tool_sequence[0] for e in recalled] == ["b", "c"]


def test_dedup_merges_same_path_and_outcome(tmp_path: Path):
    mem = FileExperientialMemory(path=str(tmp_path / "m"))
    mem.remember(_entry(tools=["vector_search"], quality=0.6))
    mem.remember(_entry(tools=["vector_search"], quality=0.8))  # same path+outcome
    recalled = mem.recall("multi_hop", "q", top_n=5)
    assert len(recalled) == 1  # merged, not appended
    assert recalled[0].final_quality == 0.8  # sliding max
    assert recalled[0].hit_count == 1  # bumped


def test_eviction_keeps_hot_and_recent(tmp_path: Path):
    mem = FileExperientialMemory(path=str(tmp_path / "m"), max_per_bucket=2)
    # 3 distinct paths -> exceeds capacity 2; coldest/oldest must be evicted.
    mem.remember(_entry(tools=["cold"], quality=0.9, hit_count=0, created_at="2020-01-01T00:00:00+00:00"))
    mem.remember(_entry(tools=["hot"], quality=0.5, hit_count=99, created_at="2021-01-01T00:00:00+00:00"))
    mem.remember(_entry(tools=["recent"], quality=0.5, hit_count=0, created_at="2026-01-01T00:00:00+00:00"))
    recalled = mem.recall("multi_hop", "q", top_n=10)
    paths = {e.tool_sequence[0] for e in recalled}
    assert len(recalled) == 2
    assert "cold" not in paths  # evicted: lowest hit_count + oldest
    assert "hot" in paths


def test_stats_reports_bucket_counts(tmp_path: Path):
    mem = FileExperientialMemory(path=str(tmp_path / "m"))
    mem.remember(_entry(intent="multi_hop", tools=["a"]))
    mem.remember(_entry(intent="factual", tools=["b"]))
    stats = mem.stats()
    assert stats["total"] == 2
    assert stats["buckets"]["multi_hop"] == 1
    assert stats["buckets"]["factual"] == 1


def test_recall_empty_bucket_returns_empty(tmp_path: Path):
    mem = FileExperientialMemory(path=str(tmp_path / "m"))
    assert mem.recall("multi_hop", "q") == []


def test_corrupt_line_is_skipped(tmp_path: Path):
    root = tmp_path / "m"
    mem = FileExperientialMemory(path=str(root))
    mem.remember(_entry(tools=["vector_search"], quality=0.7))
    # Append a garbage line to the bucket file.
    bucket = root / "experience" / "multi_hop.jsonl"
    with bucket.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    # Load must skip the bad line, not crash.
    recalled = FileExperientialMemory(path=str(root)).recall("multi_hop", "q")
    assert len(recalled) == 1

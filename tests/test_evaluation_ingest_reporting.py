"""Tests for the 构图消耗（ingest）信息传递链路."""

from __future__ import annotations

from pathlib import Path


def test_scorer_preserves_ingest_metadata():
    from chimera_rag.evaluation.scorer import Scorer

    result = Scorer(metrics=["em"]).score(
        per_example=[
            {"qid": "1", "prediction": "x", "reference": "x",
             "latency_s": 1.0, "total_tokens": 50},
        ],
        config_name="cfg",
        usage={"total_tokens": 50, "calls": 1},
        elapsed_s=1.0,
        ingest={
            "elapsed_s": 30.0,
            "usage": {"total_tokens": 2000, "calls": 25,
                      "prompt_tokens": 1500, "completion_tokens": 500},
            "documents": 1, "chunks": 25,
            "triples_raw": 40, "triples_pruned": 26,
        },
    )
    assert result.ingest is not None
    assert result.ingest["chunks"] == 25
    assert result.ingest["usage"]["calls"] == 25


def test_eval_reporter_renders_ingest_section_when_present(tmp_path: Path):
    from chimera_rag.core.types import EvalResult
    from chimera_rag.evaluation.reporter import EvalReporter

    r = EvalResult(
        config_name="demo",
        metrics={"em": 0.7, "f1": 0.7, "avg_latency_s": 1.5},
        per_example=[
            {"qid": "1", "prediction": "x", "reference": "x",
             "em": 1.0, "f1": 1.0, "latency_s": 1.5, "total_tokens": 100},
        ],
        usage={"total_tokens": 100, "calls": 2},
        elapsed_s=1.5,
        ingest={
            "elapsed_s": 42.3,
            "usage": {"total_tokens": 3000, "calls": 26,
                      "prompt_tokens": 2500, "completion_tokens": 500},
            "documents": 1, "chunks": 25,
            "triples_raw": 40, "triples_pruned": 26,
        },
    )
    md = EvalReporter(r).render_markdown()
    assert "构图阶段" in md
    assert "Chunk 数" in md
    # 显示 triples 抽取→保留
    assert "40" in md and "26" in md
    # 构图耗时与 token
    assert "42.3 秒" in md or "42 分" in md  # 只要在就行
    assert "3,000" in md
    # 合计表
    assert "总计（构图 + 推理）" in md


def test_eval_reporter_skips_ingest_section_when_absent():
    from chimera_rag.core.types import EvalResult
    from chimera_rag.evaluation.reporter import EvalReporter

    r = EvalResult(
        config_name="demo",
        metrics={"em": 0.7, "f1": 0.7, "avg_latency_s": 1.5},
        per_example=[{"qid": "1", "prediction": "x", "reference": "x",
                      "em": 1.0, "f1": 1.0, "latency_s": 1.5, "total_tokens": 100}],
        usage={"total_tokens": 100, "calls": 2},
        elapsed_s=1.5,
        ingest=None,
    )
    md = EvalReporter(r).render_markdown()
    assert "构图阶段" not in md
    assert "总计（构图 + 推理）" not in md


def test_compare_reporter_shows_ingest_columns(tmp_path: Path):
    from chimera_rag.core.types import EvalResult
    from chimera_rag.evaluation.reporter import CompareReporter

    r1 = EvalResult(
        config_name="cfg_a",
        metrics={"em": 0.8, "f1": 0.85, "avg_latency_s": 1.0},
        per_example=[{"qid": "1", "latency_s": 1.0, "total_tokens": 100}],
        usage={"total_tokens": 100, "calls": 1},
        elapsed_s=1.0,
        ingest={"elapsed_s": 30.0, "usage": {"total_tokens": 2000, "calls": 25},
                "documents": 1, "chunks": 25, "triples_pruned": 26},
    )
    r2 = EvalResult(
        config_name="cfg_b",
        metrics={"em": 0.5, "f1": 0.6, "avg_latency_s": 2.0},
        per_example=[{"qid": "1", "latency_s": 2.0, "total_tokens": 200}],
        usage={"total_tokens": 200, "calls": 1},
        elapsed_s=2.0,
        ingest=None,  # 这个配置复用了现成的图
    )
    md = CompareReporter([r1, r2]).write(tmp_path)["md"].read_text(encoding="utf-8")
    assert "构图耗时" in md
    assert "构图 Token" in md
    # cfg_a 有构图数据
    assert "2,000" in md
    # cfg_b 没有 → 应该是 "-"
    assert "| `cfg_b` |" in md


def test_scorer_ingest_roundtrip_via_eval_result_json():
    """确保 EvalResult 的 JSON 序列化能保留 ingest 字段。"""
    import json

    from chimera_rag.evaluation.scorer import Scorer

    result = Scorer(metrics=["em"]).score(
        per_example=[{"qid": "1", "prediction": "x", "reference": "x",
                      "latency_s": 1.0, "total_tokens": 10}],
        config_name="cfg",
        usage={"total_tokens": 10, "calls": 1},
        elapsed_s=1.0,
        ingest={"chunks": 5, "triples_pruned": 7,
                "elapsed_s": 12.5,
                "usage": {"total_tokens": 500, "calls": 5}},
    )
    dumped = json.loads(json.dumps(result.model_dump()))
    assert dumped["ingest"]["chunks"] == 5
    assert dumped["ingest"]["usage"]["total_tokens"] == 500

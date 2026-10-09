"""Tests for EvalReporter Markdown / CSV output."""

from __future__ import annotations

from pathlib import Path

from chimera_rag.core.types import EvalResult


def _sample_result() -> EvalResult:
    return EvalResult(
        config_name="colla_rag_only",
        metrics={"em": 0.8, "f1": 0.85, "rouge_l": 0.82, "avg_latency_s": 4.3},
        usage={"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500, "calls": 2},
        elapsed_s=9.0,
        per_example=[
            {
                "qid": "q1",
                "question": "Where does Ada work?",
                "reference": "Lumentech",
                "prediction": "Lumentech Corp",
                "em": 0.0,
                "f1": 0.67,
                "rouge_l": 0.67,
                "latency_s": 3.8,
                "intent_label": "factual",
                "strategy_name": "direct",
                "confidence": 1.0,
                "evidence_chunk_ids": ["c0", "c1"],
                "prompt_tokens": 600,
                "completion_tokens": 150,
                "total_tokens": 750,
                "llm_calls": 1,
            },
            {
                "qid": "q2",
                "question": "Which company acquired Lumentech?",
                "reference": "Helios Systems",
                "prediction": "Helios Systems",
                "em": 1.0,
                "f1": 1.0,
                "rouge_l": 1.0,
                "latency_s": 4.5,
                "intent_label": "multi_hop",
                "strategy_name": "multi_hop",
                "confidence": 1.0,
                "evidence_chunk_ids": ["c0"],
                "prompt_tokens": 600,
                "completion_tokens": 150,
                "total_tokens": 750,
                "llm_calls": 1,
            },
        ],
    )


def test_reporter_writes_json_md_csv(tmp_path: Path):
    from chimera_rag.evaluation.reporter import EvalReporter

    paths = EvalReporter(_sample_result()).write(tmp_path, "test_report")

    assert paths["json"].is_file()
    assert paths["md"].is_file()
    assert paths["csv"].is_file()


def test_markdown_contains_summary_and_tables(tmp_path: Path):
    from chimera_rag.evaluation.reporter import EvalReporter

    md = EvalReporter(_sample_result()).render_markdown()

    # 总览
    assert "评测报告" in md
    assert "一、总览" in md
    assert "综合指标" in md

    # 按意图分组（当标签存在时）
    assert "按意图分组" in md
    assert "事实型" in md  # factual
    assert "多跳型" in md  # multi_hop

    # 策略分布
    assert "策略分布" in md
    assert "direct" in md

    # 最差 / 最好样本
    assert "最差样本 Top-10" in md
    assert "最好样本 Top-10" in md


def test_markdown_intent_section_skipped_when_no_labels(tmp_path: Path):
    from chimera_rag.evaluation.reporter import EvalReporter

    r = _sample_result()
    for row in r.per_example:
        row.pop("intent_label", None)
        row.pop("strategy_name", None)

    md = EvalReporter(r).render_markdown()
    assert "按意图分组" not in md
    assert "综合指标" in md


def test_csv_has_header_and_rows(tmp_path: Path):
    from chimera_rag.evaluation.reporter import EvalReporter

    csv = EvalReporter(_sample_result()).render_csv()
    lines = csv.strip().split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert lines[0].startswith("qid,intent_label,strategy_name")
    assert "Helios" in lines[2]


def test_csv_escapes_commas_and_quotes(tmp_path: Path):
    from chimera_rag.evaluation.reporter import EvalReporter

    r = _sample_result()
    r.per_example[0]["prediction"] = 'Alice, Bob, and "Charlie"'

    csv = EvalReporter(r).render_csv()
    # Contains escaped version
    assert '"Alice, Bob, and ""Charlie"""' in csv


def test_markdown_shows_token_and_elapsed(tmp_path: Path):
    """Usage + elapsed_s must surface in the 总览 section in a readable way."""
    from chimera_rag.evaluation.reporter import EvalReporter

    md = EvalReporter(_sample_result()).render_markdown()
    # 合计 Token = 1500
    assert "合计 Token" in md
    assert "1,500" in md
    # 总耗时 9 秒
    assert "总耗时" in md
    assert "9.0 秒" in md
    # LLM 调用次数
    assert "LLM 调用次数" in md


def test_csv_includes_token_columns():
    from chimera_rag.evaluation.reporter import EvalReporter

    csv = EvalReporter(_sample_result()).render_csv()
    lines = csv.strip().split("\n")
    assert "total_tokens" in lines[0]
    assert "prompt_tokens" in lines[0]
    # value cells present
    assert ",750," in lines[1] or lines[1].endswith(",750,1") or "750" in lines[1]

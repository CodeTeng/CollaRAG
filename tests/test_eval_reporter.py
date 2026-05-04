"""Tests for EvalReporter Markdown / CSV output."""

from __future__ import annotations

from pathlib import Path

from chimera_rag.core.types import EvalResult


def _sample_result() -> EvalResult:
    return EvalResult(
        config_name="deepseek_intent_only",
        metrics={"em": 0.8, "f1": 0.85, "rouge_l": 0.82, "avg_latency_s": 4.3},
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

    # Summary section
    assert "# Evaluation report" in md
    assert "Aggregate metrics" in md
    assert "0.8" in md  # EM value

    # Intent breakdown when labels are present
    assert "Breakdown by intent" in md
    assert "factual" in md
    assert "multi_hop" in md

    # Strategy distribution
    assert "Strategy distribution" in md
    assert "direct" in md

    # Sample tables
    assert "worst samples" in md
    assert "best samples" in md


def test_markdown_intent_section_skipped_when_no_labels(tmp_path: Path):
    from chimera_rag.evaluation.reporter import EvalReporter

    # Drop intent labels.
    r = _sample_result()
    for row in r.per_example:
        row.pop("intent_label", None)
        row.pop("strategy_name", None)

    md = EvalReporter(r).render_markdown()
    assert "Breakdown by intent" not in md
    # But aggregate metrics still present
    assert "Aggregate metrics" in md


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

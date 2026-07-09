"""Tests for the experiments/eval.py offline scoring script and CompareReporter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_predictions(
    out_dir: Path,
    *,
    config_name: str,
    rows: list[dict],
    elapsed: float = 1.0,
    total_tokens: int = 0,
    calls: int = 0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    (out_dir / "meta.json").write_text(json.dumps({
        "config_name": config_name,
        "n_examples": len(rows),
        "elapsed_s": elapsed,
        "usage_total": {"total_tokens": total_tokens, "calls": calls},
    }))
    return out_dir


# ----------------------------------------------------------------------
# CLI (subprocess) smoke tests
# ----------------------------------------------------------------------
def _run_eval(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "experiments/eval.py", *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def test_eval_script_single_dir_writes_report(tmp_path: Path):
    preds = _write_predictions(
        tmp_path / "cfg_a",
        config_name="cfg_a",
        rows=[
            {"qid": "1", "prediction": "Alfred Nobel", "reference": "Alfred Nobel",
             "latency_s": 1.0, "total_tokens": 100, "prompt_tokens": 80,
             "completion_tokens": 20, "intent_label": "factual",
             "strategy_name": "direct", "confidence": 0.9},
            {"qid": "2", "prediction": "Einstein", "reference": "Alfred Nobel",
             "latency_s": 2.0, "total_tokens": 200, "prompt_tokens": 160,
             "completion_tokens": 40, "intent_label": "factual",
             "strategy_name": "direct", "confidence": 0.5},
        ],
        elapsed=3.0, total_tokens=300, calls=2,
    )
    r = _run_eval(["--predictions", str(preds), "--metrics", "em,f1"])
    assert r.returncode == 0, r.stderr
    assert (preds / "report.md").is_file()
    assert (preds / "report.json").is_file()
    assert (preds / "report.csv").is_file()

    md = (preds / "report.md").read_text(encoding="utf-8")
    assert "评测报告" in md
    assert "耗时分布" in md
    assert "Token 分布" in md


def test_eval_script_multi_dir_compare_writes_compare_report(tmp_path: Path):
    a = _write_predictions(
        tmp_path / "cfg_a",
        config_name="cfg_a",
        rows=[{"qid": "1", "prediction": "x", "reference": "x",
               "latency_s": 1.0, "total_tokens": 100}],
        elapsed=1.0, total_tokens=100, calls=1,
    )
    b = _write_predictions(
        tmp_path / "cfg_b",
        config_name="cfg_b",
        rows=[{"qid": "1", "prediction": "y", "reference": "x",
               "latency_s": 2.0, "total_tokens": 200}],
        elapsed=2.0, total_tokens=200, calls=1,
    )
    out_dir = tmp_path / "cmp"
    r = _run_eval([
        "--predictions", str(a), str(b),
        "--out", str(out_dir),
        "--metrics", "em,f1,rouge_l",
    ])
    assert r.returncode == 0, r.stderr
    assert (out_dir / "compare.md").is_file()
    assert (out_dir / "compare.json").is_file()
    assert (out_dir / "compare.csv").is_file()

    md = (out_dir / "compare.md").read_text(encoding="utf-8")
    assert "多配置对比报告" in md
    assert "cfg_a" in md and "cfg_b" in md


def test_eval_script_help_exits_zero():
    r = _run_eval(["--help"])
    assert r.returncode == 0
    assert "predictions" in (r.stdout + r.stderr).lower()


# ----------------------------------------------------------------------
# CompareReporter unit tests
# ----------------------------------------------------------------------
def test_compare_reporter_renders_tables(tmp_path: Path):
    from chimera_rag.core.types import EvalResult
    from chimera_rag.evaluation.reporter import CompareReporter

    r1 = EvalResult(
        config_name="cfg_a",
        metrics={"em": 0.8, "f1": 0.85, "rouge_l": 0.82, "avg_latency_s": 1.0},
        per_example=[
            {"qid": "1", "latency_s": 1.0, "total_tokens": 100},
        ],
        usage={"total_tokens": 100, "calls": 1},
        elapsed_s=1.0,
    )
    r2 = EvalResult(
        config_name="cfg_b",
        metrics={"em": 0.5, "f1": 0.6, "rouge_l": 0.55, "avg_latency_s": 2.0},
        per_example=[
            {"qid": "1", "latency_s": 2.0, "total_tokens": 200},
        ],
        usage={"total_tokens": 200, "calls": 1},
        elapsed_s=2.0,
    )
    paths = CompareReporter([r1, r2]).write(tmp_path)
    assert paths["md"].is_file()
    md = paths["md"].read_text(encoding="utf-8")
    assert "多配置对比报告" in md
    assert "cfg_a" in md and "cfg_b" in md
    assert "耗时分布对比" in md
    assert "Token 分布对比" in md

    csv_txt = paths["csv"].read_text(encoding="utf-8")
    assert "config,n_examples" in csv_txt
    assert "cfg_a" in csv_txt and "cfg_b" in csv_txt


def test_compare_reporter_rejects_empty_input():
    import pytest

    from chimera_rag.evaluation.reporter import CompareReporter

    with pytest.raises(ValueError):
        CompareReporter([])

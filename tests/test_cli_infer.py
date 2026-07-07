"""Tests for ``main.py infer`` subcommand."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _inject_eval_dataset(cfg_path: str, ds_path: Path, out_dir: Path) -> None:
    """Append an eval_mini dataset entry into a mock config yaml."""
    raw = Path(cfg_path).read_text()
    lines = raw.splitlines()
    out: list[str] = []
    in_datasets = False
    for line in lines:
        if line.startswith("datasets:"):
            in_datasets = True
            out.append(line)
            out.append("  eval_mini:")
            out.append("    loader: generic_jsonl")
            out.append(f"    path: {ds_path}")
            out.append(f"    output_dir: {out_dir}")
            continue
        if in_datasets and line and not line.startswith(" "):
            in_datasets = False
        out.append(line)
    Path(cfg_path).write_text("\n".join(out))


@pytest.mark.integration
def test_main_infer_writes_predictions_and_meta(
    mock_vanilla_yaml: str, tmp_path: Path
):
    ds_path = tmp_path / "qa.jsonl"
    ds_path.write_text("\n".join(
        json.dumps({"qid": f"q{i}", "question": f"Q{i}?", "answer": f"A{i}"})
        for i in range(3)
    ))
    out_dir = tmp_path / "infer_out"
    _inject_eval_dataset(mock_vanilla_yaml, ds_path, out_dir)

    r = subprocess.run(
        [
            sys.executable, "main.py", "infer",
            "--config", mock_vanilla_yaml,
            "--dataset", "eval_mini",
            "--limit", "3",
        ],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stderr={r.stderr}\nstdout={r.stdout}"

    # 输出目录应该是 <output_dir>/<config_stem>/
    stem = Path(mock_vanilla_yaml).stem
    expected_dir = out_dir / stem
    assert (expected_dir / "predictions.jsonl").is_file()
    assert (expected_dir / "meta.json").is_file()

    rows = [
        json.loads(ln) for ln in
        (expected_dir / "predictions.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert len(rows) == 3
    for row in rows:
        assert "prediction" in row
        assert "reference" in row
        assert "latency_s" in row
        # evidence chunk 原文随推理一并落盘，供 RAGAS 等 context 类指标使用。
        assert "evidence_texts" in row
        assert isinstance(row["evidence_texts"], list)
        # Scorer 还没跑，所以不应该有 em/f1
        assert "em" not in row
        assert "f1" not in row

    meta = json.loads((expected_dir / "meta.json").read_text())
    assert meta["config_name"] == stem
    assert meta["dataset"] == "eval_mini"
    assert meta["n_examples"] == 3
    assert "started_at" in meta and "finished_at" in meta


@pytest.mark.integration
def test_main_infer_explicit_out_dir(mock_vanilla_yaml: str, tmp_path: Path):
    ds_path = tmp_path / "qa.jsonl"
    ds_path.write_text(json.dumps({"qid": "q1", "question": "Q?", "answer": "A"}))
    _inject_eval_dataset(mock_vanilla_yaml, ds_path, tmp_path / "default_out")

    explicit = tmp_path / "custom_out"
    r = subprocess.run(
        [
            sys.executable, "main.py", "infer",
            "--config", mock_vanilla_yaml,
            "--dataset", "eval_mini",
            "--out", str(explicit),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (explicit / "predictions.jsonl").is_file()
    assert (explicit / "meta.json").is_file()

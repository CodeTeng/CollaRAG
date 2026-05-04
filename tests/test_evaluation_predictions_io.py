"""Tests for :mod:`chimera_rag.evaluation.predictions_io`."""

from __future__ import annotations

from pathlib import Path


def test_write_and_read_roundtrip(tmp_path: Path):
    from chimera_rag.evaluation.predictions_io import (
        read_predictions_dir,
        write_predictions_dir,
    )

    per_example = [
        {"qid": "q1", "question": "Q?", "prediction": "A", "reference": "A",
         "latency_s": 0.5, "total_tokens": 100},
        {"qid": "q2", "question": "Q2?", "prediction": "B", "reference": "C",
         "latency_s": 1.5, "total_tokens": 200},
    ]
    meta = {
        "config_name": "cfg", "dataset": "mock",
        "usage_total": {"total_tokens": 300, "calls": 2},
        "elapsed_s": 2.0,
    }
    paths = write_predictions_dir(tmp_path, per_example=per_example, meta=meta)
    assert paths["predictions"].name == "predictions.jsonl"
    assert paths["meta"].name == "meta.json"
    assert paths["predictions"].is_file()
    assert paths["meta"].is_file()

    rows2, meta2 = read_predictions_dir(tmp_path)
    assert rows2 == per_example
    assert meta2 == meta


def test_read_accepts_jsonl_path_directly(tmp_path: Path):
    from chimera_rag.evaluation.predictions_io import (
        read_predictions_dir,
        write_predictions_dir,
    )

    paths = write_predictions_dir(
        tmp_path,
        per_example=[{"qid": "1", "prediction": "x", "reference": "x"}],
        meta={"config_name": "cfg"},
    )
    rows, meta = read_predictions_dir(paths["predictions"])
    assert len(rows) == 1
    assert meta["config_name"] == "cfg"


def test_read_raises_when_missing(tmp_path: Path):
    import pytest

    from chimera_rag.evaluation.predictions_io import read_predictions_dir

    with pytest.raises(FileNotFoundError):
        read_predictions_dir(tmp_path)


def test_read_raises_on_malformed_jsonl(tmp_path: Path):
    import pytest

    from chimera_rag.evaluation.predictions_io import read_predictions_dir

    (tmp_path / "predictions.jsonl").write_text("{not json}\n")
    with pytest.raises(ValueError, match="malformed JSONL"):
        read_predictions_dir(tmp_path)


def test_read_tolerates_blank_lines(tmp_path: Path):
    from chimera_rag.evaluation.predictions_io import read_predictions_dir

    (tmp_path / "predictions.jsonl").write_text(
        '{"qid":"1"}\n\n{"qid":"2"}\n'
    )
    rows, _ = read_predictions_dir(tmp_path)
    assert [r["qid"] for r in rows] == ["1", "2"]

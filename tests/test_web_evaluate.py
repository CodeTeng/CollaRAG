"""Tests for :mod:`chimera_rag.web.routers.evaluate`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client_with_eval_ds(mock_vanilla_yaml: str, tmp_path: Path):
    from fastapi.testclient import TestClient

    from chimera_rag import ChimeraRAG
    from chimera_rag.web.app import create_app

    # Write a small eval jsonl and point the config at it.
    ds_path = tmp_path / "qa.jsonl"
    ds_path.write_text("\n".join(
        json.dumps({"qid": f"q{i}", "question": f"Q{i}?", "answer": f"A{i}"})
        for i in range(3)
    ))

    raw = Path(mock_vanilla_yaml).read_text()
    raw = raw.replace(
        "./storage/vanilla",
        str(tmp_path / "storage").replace("\\", "/"),
    )
    # Append a named eval dataset.
    raw += (
        "\n"
        f"  eval_mini:\n"
        f"    loader: generic_jsonl\n"
        f"    path: {ds_path}\n"
    )
    # Hack: the insertion above is under the datasets: key. Safer is to write a
    # full replacement config.
    cfg_dict_yaml = f"""
datasets:
  eval_mini:
    loader: generic_jsonl
    path: {ds_path}
"""
    # Simpler: inject by replacing datasets section.
    raw = Path(mock_vanilla_yaml).read_text().replace(
        "./storage/vanilla",
        str(tmp_path / "storage").replace("\\", "/"),
    )
    lines = raw.splitlines(keepends=False)
    out: list[str] = []
    in_datasets = False
    for line in lines:
        if line.startswith("datasets:"):
            in_datasets = True
            out.append(line)
            out.append("  eval_mini:")
            out.append("    loader: generic_jsonl")
            out.append(f"    path: {ds_path}")
            continue
        if in_datasets and line and not line.startswith(" "):
            in_datasets = False
        out.append(line)
    raw = "\n".join(out)

    p = tmp_path / "config.yaml"
    p.write_text(raw)
    crag = ChimeraRAG.from_config(p)
    return TestClient(create_app(crag))


def test_evaluate_runs_on_named_dataset_and_returns_metrics(client_with_eval_ds):
    r = client_with_eval_ds.post(
        "/api/evaluate", json={"dataset": "eval_mini", "limit": 3}
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["config_name"]
    assert "em" in data["metrics"]
    assert "f1" in data["metrics"]
    assert len(data["per_example"]) == 3


def test_evaluate_unknown_dataset_returns_404(client_with_eval_ds):
    r = client_with_eval_ds.post("/api/evaluate", json={"dataset": "nope"})
    assert r.status_code == 404


def test_evaluate_limit_bounds_per_example_rows(client_with_eval_ds):
    r = client_with_eval_ds.post(
        "/api/evaluate", json={"dataset": "eval_mini", "limit": 2}
    )
    assert r.status_code == 200
    assert len(r.json()["per_example"]) == 2

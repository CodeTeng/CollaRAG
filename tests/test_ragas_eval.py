"""Tests for :mod:`chimera_rag.evaluation.ragas_eval` and the RAGAS report.

The unit tests here do NOT require the real ``ragas`` package: the pure
data-mapping helper and the degradation logic are tested directly, and the
``RagasEvaluator.evaluate`` flow is exercised with stubbed builders +
a fake ``ragas`` module injected into ``sys.modules``.

A single ``@pytest.mark.integration`` smoke test runs the real RAGAS stack
against DeepSeek; it is skipped unless deps + ``LLM_API_KEY`` are present.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# rows_to_samples — pure mapping, no ragas needed
# ---------------------------------------------------------------------------
def test_rows_to_samples_maps_fields_to_ragas_naming():
    from chimera_rag.evaluation.ragas_eval import rows_to_samples

    rows = [
        {
            "question": "Where is Alice?",
            "prediction": "Paris",
            "reference": "Paris, France",
            "evidence_texts": ["Alice lives in Paris.", "Paris is in France."],
        }
    ]
    samples = rows_to_samples(rows)
    assert samples == [
        {
            "user_input": "Where is Alice?",
            "response": "Paris",
            "retrieved_contexts": ["Alice lives in Paris.", "Paris is in France."],
            "reference": "Paris, France",
        }
    ]


def test_rows_to_samples_tolerates_missing_fields():
    from chimera_rag.evaluation.ragas_eval import rows_to_samples

    samples = rows_to_samples([{"question": "Q?"}])
    assert samples[0]["user_input"] == "Q?"
    assert samples[0]["response"] == ""
    assert samples[0]["retrieved_contexts"] == []
    assert samples[0]["reference"] == ""


# ---------------------------------------------------------------------------
# Fake ragas module + stubbed builders to exercise evaluate() without deps
# ---------------------------------------------------------------------------
class _FakeEvalResult:
    """Mimics ragas EvaluationResult: supports to_pandas()."""

    def __init__(self, records: list[dict]):
        self._records = records

    def to_pandas(self):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(self._records)


def _install_fake_ragas(monkeypatch, captured: dict):
    """Inject a minimal fake `ragas` module that records evaluate() inputs."""

    fake_ragas = types.ModuleType("ragas")

    def fake_evaluate(*, dataset, metrics, llm, embeddings, **kwargs):
        captured["dataset"] = dataset
        captured["metrics"] = metrics
        captured["llm"] = llm
        captured["embeddings"] = embeddings
        # Return one record per sample carrying each metric name.
        records = []
        for s in dataset:
            rec = {"user_input": s["user_input"]}
            for m in metrics:
                rec[m] = 0.5
            records.append(rec)
        return _FakeEvalResult(records)

    fake_ragas.evaluate = fake_evaluate
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)


def _make_evaluator(monkeypatch):
    """Build a RagasEvaluator whose dep-heavy builders are stubbed out."""
    from chimera_rag.evaluation.ragas_eval import RagasEvaluator, rows_to_samples

    ev = RagasEvaluator(config=types.SimpleNamespace())
    # Dataset = plain list of samples (the fake evaluate just iterates it).
    monkeypatch.setattr(ev, "_build_dataset", lambda rows: rows_to_samples(rows))
    monkeypatch.setattr(ev, "_resolve_metrics", lambda names: list(names))
    monkeypatch.setattr(ev, "_build_llm", lambda: "FAKE_LLM")
    monkeypatch.setattr(ev, "_build_embeddings", lambda: "FAKE_EMB")
    return ev


def test_evaluate_passes_mapped_dataset_and_builders(monkeypatch):
    captured: dict = {}
    _install_fake_ragas(monkeypatch, captured)
    ev = _make_evaluator(monkeypatch)

    rows = [
        {
            "question": "Q1?",
            "prediction": "A1",
            "reference": "A1",
            "evidence_texts": ["ctx for q1"],
        }
    ]
    result = ev.evaluate(rows, config_name="unit")

    # Mapping reached evaluate() with ragas-0.2 naming.
    assert captured["dataset"][0]["user_input"] == "Q1?"
    assert captured["dataset"][0]["retrieved_contexts"] == ["ctx for q1"]
    assert captured["llm"] == "FAKE_LLM"
    assert captured["embeddings"] == "FAKE_EMB"
    # All four default metrics requested (context present).
    assert set(captured["metrics"]) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }
    assert result.n_examples == 1
    assert not result.skipped_metrics


def test_evaluate_skips_context_metrics_when_no_evidence(monkeypatch):
    captured: dict = {}
    _install_fake_ragas(monkeypatch, captured)
    ev = _make_evaluator(monkeypatch)

    rows = [{"question": "Q?", "prediction": "A", "reference": "A"}]  # no evidence_texts
    result = ev.evaluate(rows, config_name="unit")

    # Only the non-context metric survives.
    assert captured["metrics"] == ["answer_relevancy"]
    assert set(result.skipped_metrics) == {
        "faithfulness",
        "context_precision",
        "context_recall",
    }


def test_evaluate_raises_when_all_metrics_need_context(monkeypatch):
    from chimera_rag.core.exceptions import ProviderError

    captured: dict = {}
    _install_fake_ragas(monkeypatch, captured)
    ev = _make_evaluator(monkeypatch)

    rows = [{"question": "Q?", "prediction": "A", "reference": "A"}]
    with pytest.raises(ProviderError):
        ev.evaluate(rows, metrics=["faithfulness", "context_recall"])


# ---------------------------------------------------------------------------
# RagasReporter — no ragas needed
# ---------------------------------------------------------------------------
def test_ragas_reporter_writes_three_files(tmp_path: Path):
    from chimera_rag.evaluation.ragas_eval import RagasResult
    from chimera_rag.evaluation.ragas_report import RagasReporter

    result = RagasResult(
        config_name="demo",
        metrics={"faithfulness": 0.9, "answer_relevancy": 0.8},
        per_example=[
            {"user_input": "Q1?", "faithfulness": 0.9, "answer_relevancy": 0.8}
        ],
        skipped_metrics=["context_recall"],
        n_examples=1,
    )
    paths = RagasReporter(result).write(tmp_path)

    assert paths["md"].is_file()
    assert paths["json"].is_file()
    assert paths["csv"].is_file()

    md = paths["md"].read_text(encoding="utf-8")
    assert "RAGAS 评测报告" in md
    assert "忠实度" in md
    assert "context_recall" in md  # skipped note

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "faithfulness" in csv_text
    assert "AVG" in csv_text


# ---------------------------------------------------------------------------
# Integration smoke (skipped unless real deps + key are present)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_ragas_real_smoke():
    import os

    pytest.importorskip("ragas", reason="install: uv sync --extra ragas")
    if not os.getenv("LLM_API_KEY"):
        pytest.skip("LLM_API_KEY not set")

    from chimera_rag.core.config import load_config
    from chimera_rag.evaluation.ragas_eval import RagasEvaluator

    config = load_config("configs/colla_rag_only.yaml")
    rows = [
        {
            "question": "In what year was Yggdrasil AI founded?",
            "prediction": "2019",
            "reference": "2019",
            "evidence_texts": ["Yggdrasil AI was founded in 2019."],
        }
    ]
    result = RagasEvaluator(config).evaluate(rows, config_name="smoke")
    assert result.n_examples == 1
    assert result.metrics  # at least one metric computed

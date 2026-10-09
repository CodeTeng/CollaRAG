"""Tests for :mod:`chimera_rag.evaluation.inference` and :mod:`.scorer`."""

from __future__ import annotations

from chimera_rag.core.types import Answer, QAExample


def test_inference_runner_records_latency_and_tokens_without_scoring():
    from chimera_rag.evaluation.inference import InferenceRunner

    class FakeLLM:
        def __init__(self):
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0, "calls": 0}

        def tick(self, p: int, c: int):
            self.usage["prompt_tokens"] += p
            self.usage["completion_tokens"] += c
            self.usage["total_tokens"] += p + c
            self.usage["calls"] += 1

    class FakeRAG:
        def __init__(self):
            self.llm = FakeLLM()

        def query(self, query):
            self.llm.tick(100, 10)
            return Answer(text="Alfred Nobel", intent_label="factual",
                          strategy_name="direct", confidence=0.9)

    rag = FakeRAG()
    examples = [
        QAExample(qid="1", question="Who invented dynamite?", answer="Alfred Nobel"),
        QAExample(qid="2", question="Capital of France?", answer="Paris"),
    ]
    rows, meta = InferenceRunner().run(
        pipeline=rag, dataset_examples=examples,
        config_name="fake", show_progress=False,
    )
    # 每条都不算分，但带上 prediction + token
    assert len(rows) == 2
    for row in rows:
        assert row["prediction"] == "Alfred Nobel"
        assert "em" not in row and "f1" not in row  # scoring 还没发生
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 10
        assert row["total_tokens"] == 110
        assert row["llm_calls"] == 1
        assert row["latency_s"] >= 0

    assert meta["config_name"] == "fake"
    assert meta["n_examples"] == 2
    assert meta["usage_total"]["calls"] == 2
    assert "started_at" in meta and "finished_at" in meta


def test_scorer_computes_em_f1_rouge_on_existing_rows():
    from chimera_rag.evaluation.scorer import Scorer

    rows = [
        {"qid": "1", "prediction": "Alfred Nobel", "reference": "Alfred Nobel",
         "latency_s": 1.0, "total_tokens": 100},
        {"qid": "2", "prediction": "Einstein", "reference": "Alfred Nobel",
         "latency_s": 2.0, "total_tokens": 200},
    ]
    result = Scorer(metrics=["em", "f1", "rouge_l"]).score(
        per_example=rows, config_name="fake",
        usage={"total_tokens": 300, "calls": 2}, elapsed_s=3.0,
    )
    assert result.config_name == "fake"
    assert result.metrics["em"] == 0.5  # 1 of 2 exact
    # 聚合 avg_latency_s = (1+2)/2 = 1.5
    assert abs(result.metrics["avg_latency_s"] - 1.5) < 1e-6
    # per_example 里被加上 em/f1/rouge_l 字段
    assert result.per_example[0]["em"] == 1.0
    assert result.per_example[1]["em"] == 0.0
    assert result.usage["total_tokens"] == 300


def test_scorer_rejects_unknown_metric():
    import pytest

    from chimera_rag.evaluation.scorer import Scorer

    with pytest.raises(ValueError):
        Scorer(metrics=["bleu"])


def test_scorer_handles_empty_rows_gracefully():
    from chimera_rag.evaluation.scorer import Scorer

    result = Scorer(metrics=["em", "f1"]).score(per_example=[], config_name="empty")
    assert result.metrics["em"] == 0.0
    assert result.metrics["f1"] == 0.0
    assert result.metrics["avg_latency_s"] == 0.0
    assert result.per_example == []

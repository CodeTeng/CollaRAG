"""Tests for :mod:`chimera_rag.evaluation.evaluator`."""

from __future__ import annotations


def test_evaluator_runs_examples_and_aggregates_metrics():
    from chimera_rag.core.types import Answer, QAExample
    from chimera_rag.evaluation.evaluator import Evaluator

    # Fake pipeline: always answers "Alfred Nobel" regardless of question.
    class FakeRAG:
        def query(self, query):
            return Answer(text="Alfred Nobel")

    examples = [
        QAExample(qid="1", question="?", answer="Alfred Nobel"),   # perfect match
        QAExample(qid="2", question="?", answer="Einstein"),       # wrong
        QAExample(qid="3", question="?", answer="alfred nobel"),   # case-insensitive match
    ]
    result = Evaluator(metrics=["em", "f1"]).run(
        pipeline=FakeRAG(), dataset_examples=examples, config_name="fake"
    )
    # EM: 2 out of 3 correct after normalisation = 0.666...
    assert result.config_name == "fake"
    assert result.metrics["em"] > 0.5
    assert "f1" in result.metrics
    assert len(result.per_example) == 3


def test_evaluator_respects_limit_in_dataset_examples():
    from chimera_rag.core.types import Answer, QAExample
    from chimera_rag.evaluation.evaluator import Evaluator

    class FakeRAG:
        def query(self, query):
            return Answer(text="x")

    examples = [QAExample(qid=str(i), question="?", answer="x") for i in range(10)]
    result = Evaluator(metrics=["em"]).run(
        pipeline=FakeRAG(), dataset_examples=examples[:5], config_name="c"
    )
    assert len(result.per_example) == 5

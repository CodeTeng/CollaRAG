"""Evaluator: run a pipeline over a QA dataset and aggregate metrics."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from chimera_rag.core.types import Answer, EvalResult, QAExample, Query
from chimera_rag.evaluation.metrics import exact_match, rouge_l, token_f1

logger = logging.getLogger(__name__)


class SupportsQuery(Protocol):
    """Structural type: anything with a `.query(Query) -> Answer` method."""

    def query(self, query: Query) -> Answer: ...


_METRIC_FNS = {
    "em": exact_match,
    "f1": token_f1,
    "rouge_l": rouge_l,
}


class Evaluator:
    """Run a pipeline over :class:`QAExample` list, return :class:`EvalResult`."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["em", "f1"]
        for m in self.metrics:
            if m not in _METRIC_FNS:
                raise ValueError(f"unknown metric: {m!r}; supported: {sorted(_METRIC_FNS)}")

    def run(
        self,
        *,
        pipeline: SupportsQuery,
        dataset_examples: list[QAExample],
        config_name: str = "default",
    ) -> EvalResult:
        per_example: list[dict[str, Any]] = []
        aggregates: dict[str, list[float]] = {m: [] for m in self.metrics}

        for ex in dataset_examples:
            t0 = time.perf_counter()
            answer = pipeline.query(Query(text=ex.question))
            latency = time.perf_counter() - t0

            row: dict[str, Any] = {
                "qid": ex.qid,
                "question": ex.question,
                "reference": ex.answer,
                "prediction": answer.text,
                "latency_s": round(latency, 4),
                # Extra trace fields — especially useful for IntentAgent runs.
                "intent_label": answer.intent_label,
                "strategy_name": answer.strategy_name,
                "confidence": answer.confidence,
                "evidence_chunk_ids": list(answer.evidence_chunk_ids),
            }
            for m in self.metrics:
                score = _METRIC_FNS[m](answer.text, ex.answer)
                row[m] = score
                aggregates[m].append(score)
            per_example.append(row)

        agg_metrics = {
            m: sum(scores) / len(scores) if scores else 0.0
            for m, scores in aggregates.items()
        }
        # Secondary aggregates for diagnostics.
        agg_metrics["avg_latency_s"] = (
            sum(r["latency_s"] for r in per_example) / len(per_example)
            if per_example
            else 0.0
        )
        logger.info("evaluation complete: config=%s metrics=%s", config_name, agg_metrics)

        return EvalResult(
            config_name=config_name,
            metrics=agg_metrics,
            per_example=per_example,
        )


__all__ = ["Evaluator"]

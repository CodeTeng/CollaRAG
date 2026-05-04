"""Evaluator: run a pipeline over a QA dataset and aggregate metrics."""

from __future__ import annotations

import copy
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


def _snapshot_usage(pipeline: Any) -> dict[str, int]:
    """Best-effort: return LLM provider's cumulative usage as a flat dict."""
    llm = getattr(pipeline, "llm", None)
    usage = getattr(llm, "usage", None)
    if isinstance(usage, dict):
        return dict(usage)
    return {}


def _diff_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Return a - b for each key present in either (missing treated as 0)."""
    keys = set(a) | set(b)
    return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in keys}


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
        show_progress: bool = True,
    ) -> EvalResult:
        per_example: list[dict[str, Any]] = []
        aggregates: dict[str, list[float]] = {m: [] for m in self.metrics}

        # Snapshot LLM usage before the loop so we can record per-example deltas.
        usage_before_run = _snapshot_usage(pipeline)
        usage_prev = copy.deepcopy(usage_before_run)

        t0_run = time.perf_counter()

        # Optional tqdm progress bar — graceful fallback if not installed.
        iterator: Any = dataset_examples
        bar = None
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                bar = tqdm(
                    dataset_examples,
                    desc=f"评测 [{config_name}]",
                    unit="题",
                    total=len(dataset_examples),
                    dynamic_ncols=True,
                )
                iterator = bar
            except ImportError:  # pragma: no cover
                iterator = dataset_examples

        for i, ex in enumerate(iterator):
            t0 = time.perf_counter()
            answer = pipeline.query(Query(text=ex.question))
            latency = time.perf_counter() - t0

            usage_now = _snapshot_usage(pipeline)
            usage_delta = _diff_usage(usage_now, usage_prev)
            usage_prev = usage_now

            row: dict[str, Any] = {
                "qid": ex.qid,
                "question": ex.question,
                "reference": ex.answer,
                "prediction": answer.text,
                "latency_s": round(latency, 4),
                "intent_label": answer.intent_label,
                "strategy_name": answer.strategy_name,
                "confidence": answer.confidence,
                "evidence_chunk_ids": list(answer.evidence_chunk_ids),
                # Per-example token delta from the LLM provider.
                "prompt_tokens": int(usage_delta.get("prompt_tokens", 0)),
                "completion_tokens": int(usage_delta.get("completion_tokens", 0)),
                "total_tokens": int(usage_delta.get("total_tokens", 0)),
                "llm_calls": int(usage_delta.get("calls", 0)),
            }
            for m in self.metrics:
                score = _METRIC_FNS[m](answer.text, ex.answer)
                row[m] = score
                aggregates[m].append(score)
            per_example.append(row)

            # Live log + progress bar update.
            msg = (
                f"[{i + 1:>3}/{len(dataset_examples)}] "
                f"intent={answer.intent_label or '-':<12s} "
                f"f1={row.get('f1', 0.0):.2f} "
                f"em={row.get('em', 0.0):.2f} "
                f"lat={latency:5.2f}s "
                f"tok={row['total_tokens']:4d} "
                f"| Q: {ex.question[:48]!r}"
            )
            logger.info(msg)
            if bar is not None:
                # Short postfix so it fits in one line
                bar.set_postfix_str(
                    f"EM={row.get('em', 0.0):.2f} F1={row.get('f1', 0.0):.2f} "
                    f"tok={row['total_tokens']}"
                )

        if bar is not None:
            bar.close()

        elapsed = time.perf_counter() - t0_run
        usage_after = _snapshot_usage(pipeline)
        usage_total = _diff_usage(usage_after, usage_before_run)

        agg_metrics = {
            m: sum(scores) / len(scores) if scores else 0.0
            for m, scores in aggregates.items()
        }
        agg_metrics["avg_latency_s"] = (
            sum(r["latency_s"] for r in per_example) / len(per_example)
            if per_example
            else 0.0
        )
        logger.info(
            "evaluation complete: config=%s samples=%d elapsed=%.1fs usage=%s metrics=%s",
            config_name, len(per_example), elapsed, usage_total, agg_metrics,
        )

        return EvalResult(
            config_name=config_name,
            metrics=agg_metrics,
            per_example=per_example,
            usage=usage_total,
            elapsed_s=round(elapsed, 3),
        )


__all__ = ["Evaluator"]

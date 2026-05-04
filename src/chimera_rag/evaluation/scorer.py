"""打分阶段 (不调用 LLM)。

:class:`Scorer` 读取上一阶段产生的 ``per_example`` 列表（字典，至少包含
``prediction`` / ``reference``），逐条计算 EM / F1 / ROUGE-L 等指标并聚合。

核心不变量：**Scorer 不依赖 pipeline、也不依赖网络**，纯函数运算，
可以在完全离线的环境里反复跑。
"""

from __future__ import annotations

import logging
from typing import Any

from chimera_rag.core.types import EvalResult
from chimera_rag.evaluation.metrics import exact_match, rouge_l, token_f1

logger = logging.getLogger(__name__)


_METRIC_FNS = {
    "em": exact_match,
    "f1": token_f1,
    "rouge_l": rouge_l,
}


class Scorer:
    """对已产出的 predictions 离线打分。"""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["em", "f1"]
        for m in self.metrics:
            if m not in _METRIC_FNS:
                raise ValueError(
                    f"unknown metric: {m!r}; supported: {sorted(_METRIC_FNS)}"
                )

    def score(
        self,
        *,
        per_example: list[dict[str, Any]],
        config_name: str = "default",
        usage: dict[str, int] | None = None,
        elapsed_s: float = 0.0,
        ingest: dict[str, Any] | None = None,
    ) -> EvalResult:
        aggregates: dict[str, list[float]] = {m: [] for m in self.metrics}
        scored_rows: list[dict[str, Any]] = []
        for row in per_example:
            out = dict(row)
            pred = str(out.get("prediction", ""))
            ref = str(out.get("reference", ""))
            for m in self.metrics:
                s = _METRIC_FNS[m](pred, ref)
                out[m] = s
                aggregates[m].append(s)
            scored_rows.append(out)

        agg_metrics = {
            m: sum(scores) / len(scores) if scores else 0.0
            for m, scores in aggregates.items()
        }
        latencies = [r.get("latency_s", 0.0) for r in scored_rows]
        agg_metrics["avg_latency_s"] = (
            sum(latencies) / len(latencies) if latencies else 0.0
        )

        logger.info(
            "scoring complete: config=%s samples=%d metrics=%s",
            config_name, len(scored_rows), agg_metrics,
        )
        return EvalResult(
            config_name=config_name,
            metrics=agg_metrics,
            per_example=scored_rows,
            usage=dict(usage or {}),
            elapsed_s=round(float(elapsed_s), 3),
            ingest=dict(ingest) if ingest else None,
        )


__all__ = ["Scorer"]

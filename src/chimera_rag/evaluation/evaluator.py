"""Evaluator：InferenceRunner + Scorer 的薄组合。

保留是为了兼容既有的 Web 路由和单元测试。新代码更推荐直接使用
:class:`chimera_rag.evaluation.inference.InferenceRunner` 和
:class:`chimera_rag.evaluation.scorer.Scorer`。
"""

from __future__ import annotations

from typing import Protocol

from chimera_rag.core.types import Answer, EvalResult, QAExample, Query
from chimera_rag.evaluation.inference import InferenceRunner
from chimera_rag.evaluation.scorer import Scorer


class SupportsQuery(Protocol):
    def query(self, query: Query) -> Answer: ...


class Evaluator:
    """保留旧 API：一次调用跑完推理 + 打分。"""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self._scorer = Scorer(metrics=metrics)
        self._runner = InferenceRunner()
        self.metrics = self._scorer.metrics

    def run(
        self,
        *,
        pipeline: SupportsQuery,
        dataset_examples: list[QAExample],
        config_name: str = "default",
        show_progress: bool = True,
    ) -> EvalResult:
        per_example, meta = self._runner.run(
            pipeline=pipeline,
            dataset_examples=dataset_examples,
            config_name=config_name,
            show_progress=show_progress,
        )
        return self._scorer.score(
            per_example=per_example,
            config_name=config_name,
            usage=meta.get("usage_total", {}),
            elapsed_s=float(meta.get("elapsed_s", 0.0)),
        )


_ = EvalResult  # re-exported via package __init__
__all__ = ["Evaluator"]

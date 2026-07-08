"""推理阶段 (不计算指标)。

`InferenceRunner.run()` 对一个 pipeline 跑一批 QAExample，产出：

* ``per_example`` : 每条 QA 的预测 + 耗时 + token 细节
* ``meta``        : 运行级元信息（起止时间、累计 token/调用、配置名等）

产物可被 :class:`chimera_rag.evaluation.scorer.Scorer` 消费，也可以被
:mod:`chimera_rag.evaluation.predictions_io` 落盘为 ``predictions.jsonl`` +
``meta.json`` 供离线打分。
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import UTC, datetime
from typing import Any, Protocol

from chimera_rag.core.types import Answer, QAExample, Query

logger = logging.getLogger(__name__)


class SupportsQuery(Protocol):
    """Structural type: anything with a ``.query(Query) -> Answer`` method."""

    def query(self, query: Query) -> Answer: ...


def _snapshot_usage(pipeline: Any) -> dict[str, int]:
    """尽力而为：读取 pipeline.llm.usage 里的累计 usage（dict 形式）。"""
    llm = getattr(pipeline, "llm", None)
    usage = getattr(llm, "usage", None)
    if isinstance(usage, dict):
        return dict(usage)
    return {}


def _diff_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """返回 a - b （缺失键按 0 处理）。"""
    keys = set(a) | set(b)
    return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in keys}


def _lookup_texts(pipeline: Any, chunk_ids: list[str]) -> list[str]:
    """把 evidence chunk id 反查成原文，供 RAGAS 等需要 context 的离线评测使用。

    chunk 原文存放在 pipeline 的 ``chunk_lookup``（CollaRAG 等检索器在其上
    暴露），回退到 ``pipeline.state.chunk_lookup``。查不到的 id 静默跳过，保证
    推理流程的健壮性——缺失只意味着对应指标拿不到该片段，不应让整批推理失败。
    """
    lookup = getattr(pipeline, "chunk_lookup", None)
    if not lookup:
        state = getattr(pipeline, "state", None)
        lookup = getattr(state, "chunk_lookup", None)
    if not lookup:
        return []

    texts: list[str] = []
    for cid in chunk_ids:
        chunk = lookup.get(cid)
        text = getattr(chunk, "text", None)
        if text:
            texts.append(text)
    return texts


class InferenceRunner:
    """只跑 pipeline、不算指标；保留耗时、token、evidence 等诊断字段。"""

    def run(
        self,
        *,
        pipeline: SupportsQuery,
        dataset_examples: list[QAExample],
        config_name: str = "default",
        show_progress: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        per_example: list[dict[str, Any]] = []

        usage_before_run = _snapshot_usage(pipeline)
        usage_prev = copy.deepcopy(usage_before_run)

        started_at = datetime.now(UTC)
        t0_run = time.perf_counter()

        iterator: Any = dataset_examples
        bar = None
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                bar = tqdm(
                    dataset_examples,
                    desc=f"推理 [{config_name}]",
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
                "evidence_texts": _lookup_texts(pipeline, list(answer.evidence_chunk_ids)),
                "prompt_tokens": int(usage_delta.get("prompt_tokens", 0)),
                "completion_tokens": int(usage_delta.get("completion_tokens", 0)),
                "total_tokens": int(usage_delta.get("total_tokens", 0)),
                "llm_calls": int(usage_delta.get("calls", 0)),
            }
            per_example.append(row)

            logger.info(
                "[%3d/%d] intent=%-12s lat=%5.2fs tok=%4d | Q: %r",
                i + 1, len(dataset_examples),
                (answer.intent_label or "-"),
                latency, row["total_tokens"], ex.question[:48],
            )
            if bar is not None:
                bar.set_postfix_str(
                    f"tok={row['total_tokens']} lat={latency:.1f}s"
                )

        if bar is not None:
            bar.close()

        elapsed = time.perf_counter() - t0_run
        finished_at = datetime.now(UTC)
        usage_total = _diff_usage(_snapshot_usage(pipeline), usage_before_run)

        meta: dict[str, Any] = {
            "config_name": config_name,
            "n_examples": len(per_example),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed, 3),
            "usage_total": usage_total,
        }
        logger.info(
            "inference complete: config=%s samples=%d elapsed=%.1fs usage=%s",
            config_name, len(per_example), elapsed, usage_total,
        )
        return per_example, meta


__all__ = ["InferenceRunner", "SupportsQuery"]

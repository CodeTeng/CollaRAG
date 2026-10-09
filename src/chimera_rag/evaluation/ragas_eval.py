"""RAGAS 语义级评测 (调用 LLM + embedding)。

:class:`RagasEvaluator` 消费 ``predictions.jsonl`` 的行（由 ``main.py infer``
产出，需含 ``evidence_texts`` 字段），用 RAGAS 计算四个语义指标：

* ``faithfulness``      —— 答案是否由检索 context 支撑（衡量幻觉）
* ``answer_relevancy``  —— 答案与问题的相关程度（需 embedding）
* ``context_precision`` —— 检索 context 中有用片段的排序质量
* ``context_recall``    —— 标准答案所需信息是否被 context 覆盖

judge LLM 与 embedding 均从项目 ``config.yaml`` 构建：

* LLM     使用 ``config.llm``（DeepSeek / OpenAI 兼容；连接信息来自
  ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL`` 等环境变量，与
  :class:`chimera_rag.providers.llm.openai_provider.OpenAILLMProvider` 同源）。
* embedding 根据 ``config.embedding.provider`` 选择对应 LangChain embeddings
  （``sentence_transformer`` -> HuggingFace 本地；``openai`` -> OpenAI 接口）。

所有 ``ragas`` / ``langchain*`` 依赖延迟导入，未安装 ``ragas`` extra 时给出
清晰报错，不影响其它评测路径。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from chimera_rag.core.exceptions import ProviderError

if TYPE_CHECKING:
    from chimera_rag.core.config import AppConfig

logger = logging.getLogger(__name__)

# 默认四个语义指标，名称与 ragas.metrics 的导出对象对齐。
DEFAULT_RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# 需要 context 原文 (retrieved_contexts) 才能计算的指标；缺失 evidence_texts 时跳过。
_CONTEXT_METRICS = {"faithfulness", "context_precision", "context_recall"}


def rows_to_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 predictions.jsonl 的行映射为 RAGAS 0.2 的样本字段命名。

    抽成纯函数便于单测（不依赖 ragas 安装）。
    """
    return [
        {
            "user_input": str(r.get("question", "")),
            "response": str(r.get("prediction", "")),
            "retrieved_contexts": list(r.get("evidence_texts", []) or []),
            "reference": str(r.get("reference", "")),
        }
        for r in rows
    ]


@dataclass
class RagasResult:
    """RAGAS 评测的聚合结果 + 逐条明细。"""

    config_name: str
    metrics: dict[str, float]
    per_example: list[dict[str, Any]] = field(default_factory=list)
    skipped_metrics: list[str] = field(default_factory=list)
    n_examples: int = 0


class RagasEvaluator:
    """基于项目 config 构建 judge LLM / embedding，跑 RAGAS 四指标。"""

    def __init__(self, config: AppConfig | Any) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def evaluate(
        self,
        rows: list[dict[str, Any]],
        *,
        metrics: list[str] | None = None,
        config_name: str = "ragas",
    ) -> RagasResult:
        metric_names = metrics or list(DEFAULT_RAGAS_METRICS)

        has_context = any(r.get("evidence_texts") for r in rows)
        active_metrics = list(metric_names)
        skipped: list[str] = []
        if not has_context:
            skipped = [m for m in active_metrics if m in _CONTEXT_METRICS]
            active_metrics = [m for m in active_metrics if m not in _CONTEXT_METRICS]
            if skipped:
                logger.warning(
                    "predictions 缺少 evidence_texts，跳过需要 context 的指标: %s。"
                    "请用更新后的 infer 重跑以获得完整 RAGAS 指标。",
                    skipped,
                )
        if not active_metrics:
            raise ProviderError(
                "没有可计算的 RAGAS 指标：所有所选指标都依赖 context 原文，"
                "但 predictions 中缺少 evidence_texts。"
            )

        dataset = self._build_dataset(rows)
        ragas_metrics = self._resolve_metrics(active_metrics)
        llm = self._build_llm()
        embeddings = self._build_embeddings()

        from ragas import evaluate as ragas_evaluate

        logger.info(
            "running RAGAS: metrics=%s samples=%d", active_metrics, len(rows)
        )
        result = ragas_evaluate(
            dataset=dataset,
            metrics=ragas_metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=True,
        )

        return self._to_result(result, active_metrics, skipped, config_name, len(rows))

    # ------------------------------------------------------------------
    def _build_dataset(self, rows: list[dict[str, Any]]):
        """rows -> ragas EvaluationDataset（0.2 字段命名）。"""
        try:
            from ragas import EvaluationDataset
        except ImportError as e:  # pragma: no cover - optional dep path
            raise ProviderError(
                "ragas not installed; run `uv sync --extra ragas` ({})"f"{e}"
            ) from e

        return EvaluationDataset.from_list(rows_to_samples(rows))

    # ------------------------------------------------------------------
    def _resolve_metrics(self, names: list[str]) -> list[Any]:
        try:
            from ragas import metrics as ragas_metrics
        except ImportError as e:  # pragma: no cover - optional dep path
            raise ProviderError(
                "ragas not installed; run `uv sync --extra ragas` ({})"f"{e}"
            ) from e

        resolved: list[Any] = []
        for name in names:
            metric = getattr(ragas_metrics, name, None)
            if metric is None:
                raise ProviderError(
                    f"unknown RAGAS metric: {name!r}; "
                    f"supported: {sorted(DEFAULT_RAGAS_METRICS)}"
                )
            resolved.append(metric)
        return resolved

    # ------------------------------------------------------------------
    def _build_llm(self):
        """从 config.llm 构造 RAGAS 的 judge LLM（LangChain ChatOpenAI 包装）。"""
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
        except ImportError as e:  # pragma: no cover - optional dep path
            raise ProviderError(
                "ragas/langchain-openai not installed; run `uv sync --extra ragas` "
                f"({e})"
            ) from e

        cfg = getattr(self.config, "llm", None)
        provider = getattr(cfg, "provider", "openai")

        api_key, base_url, model = self._resolve_llm_conn(cfg, provider)
        chat = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
            timeout=float(getattr(cfg, "timeout", 60.0) or 60.0),
        )
        return LangchainLLMWrapper(chat)

    @staticmethod
    def _resolve_llm_conn(cfg: Any, provider: str) -> tuple[str, str | None, str]:
        """复用 OpenAILLMProvider 的解析优先级：统一 LLM_* 环境变量优先。"""
        defaults = {
            "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
            "openai": ("OPENAI_API_KEY", None, "gpt-4o-mini"),
        }
        default_key_env, default_base, default_model = defaults.get(
            provider, ("OPENAI_API_KEY", None, "gpt-4o-mini")
        )

        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            api_key_env = getattr(cfg, "api_key_env", None) or default_key_env
            api_key = os.getenv(api_key_env)
        if not api_key:
            raise ProviderError(
                "API key not found; set LLM_API_KEY in .env for the RAGAS judge LLM."
            )

        base_url = os.getenv("LLM_BASE_URL")
        if not base_url:
            base_url_env = getattr(cfg, "base_url_env", None)
            base_url = os.getenv(base_url_env) if base_url_env else None
        if not base_url:
            base_url = default_base

        model = os.getenv("LLM_MODEL") or getattr(cfg, "model", "") or default_model
        return api_key, base_url, model

    # ------------------------------------------------------------------
    def _build_embeddings(self):
        """从 config.embedding 构造 RAGAS 的 embeddings（answer_relevancy 需要）。"""
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
        except ImportError as e:  # pragma: no cover - optional dep path
            raise ProviderError(
                "ragas not installed; run `uv sync --extra ragas` ({})"f"{e}"
            ) from e

        cfg = getattr(self.config, "embedding", None)
        provider = getattr(cfg, "provider", "sentence_transformer")
        model = getattr(cfg, "model", "sentence-transformers/all-MiniLM-L6-v2")

        if provider in {"sentence_transformer", "huggingface"}:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as e:  # pragma: no cover - optional dep path
                raise ProviderError(
                    "langchain-huggingface not installed; run "
                    f"`uv sync --extra ragas` ({e})"
                ) from e
            return LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=model)
            )

        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError as e:  # pragma: no cover - optional dep path
                raise ProviderError(
                    "langchain-openai not installed; run "
                    f"`uv sync --extra ragas` ({e})"
                ) from e
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            return LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)
            )

        raise ProviderError(
            f"unsupported embedding provider for RAGAS: {provider!r}; "
            "use 'sentence_transformer' or 'openai'."
        )

    # ------------------------------------------------------------------
    def _to_result(
        self,
        ragas_result: Any,
        metric_names: list[str],
        skipped: list[str],
        config_name: str,
        n_examples: int,
    ) -> RagasResult:
        """把 ragas EvaluationResult 转成我们自己的 RagasResult。"""
        # 逐条明细：ragas 0.2 的 EvaluationResult 可转 pandas，再转 records。
        per_example: list[dict[str, Any]] = []
        try:
            df = ragas_result.to_pandas()
            per_example = df.to_dict(orient="records")
        except Exception as e:  # pragma: no cover - 防御性
            logger.warning("failed to expand RAGAS per-example rows: %s", e)

        # 聚合分：对每个指标取所有样本的均值（忽略 NaN）。
        agg: dict[str, float] = {}
        for name in metric_names:
            scores = [
                row[name]
                for row in per_example
                if name in row and _is_number(row[name])
            ]
            if scores:
                agg[name] = sum(scores) / len(scores)

        # 兜底：若 per_example 解析失败，尝试直接从 result 的 dict 形态读取。
        if not agg:
            try:
                raw = dict(ragas_result)
                agg = {k: float(v) for k, v in raw.items() if _is_number(v)}
            except Exception:  # pragma: no cover
                pass

        return RagasResult(
            config_name=config_name,
            metrics=agg,
            per_example=per_example,
            skipped_metrics=skipped,
            n_examples=n_examples,
        )


def _is_number(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f  # NaN != NaN


__all__ = [
    "DEFAULT_RAGAS_METRICS",
    "RagasEvaluator",
    "RagasResult",
    "rows_to_samples",
]

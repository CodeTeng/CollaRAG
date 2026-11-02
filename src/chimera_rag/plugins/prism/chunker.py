"""PrismRAG 分块器 —— 当前为过渡版（单粒度自适应）。

⚠️ **实装状态**：本文件目前只提供 Phase 2（预测式合并）的单粒度版本，
产出的 Chunk 全部是 ``unified`` 粒度（既进抽取也进向量库），
因此对下游管线完全透明。

PrismRAG 的完整设计（双粒度 fine/coarse + 父子 provenance 绑定 +
类型化缺陷 D1–D6 审计修复循环）见 ``docs/innovation-1/design.md``，
按 W1–W6 增量落地。落地时本类会：

1. 产出 ``granularity="fine"`` 的抽取单元与 ``granularity="coarse"`` 的检索单元
2. 用 ``parent_chunk_id`` 把 fine 绑定到其 coarse 父块
3. 由 ``IngestionPipeline`` 按粒度三分支路由（fine→extractor，coarse→vector_store）

尺寸公式（Phase 2 的预测式合并目标）::

    size = base - (score - 0.5) * (max_size - min_size)

score 高（复杂 / 实体密集）→ 块更小，保护三元组纯度；
score 低 → 块更大，控制抽取成本。
"""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Document
from chimera_rag.interfaces.chunker import BaseChunker
from chimera_rag.plugins.prism.features import (
    estimate_complexity,
    estimate_entity_density,
    split_sentences,
)


def compute_adaptive_chunk_size(
    score: float,
    *,
    min_size: int,
    max_size: int,
    base_size: int,
) -> int:
    """给定 0..1 的综合 ``score``，返回目标块大小。

    即使 ``score`` 越界也会截断到 ``[min_size, max_size]``。
    刻意暴露为模块级纯函数，便于在 score = 0 / 0.5 / 1 三个边界上独立单测。
    """
    if min_size > max_size:
        raise ValueError("min_size must be <= max_size")
    span = max_size - min_size
    raw = base_size - (score - 0.5) * span
    if raw < min_size:
        return min_size
    if raw > max_size:
        return max_size
    return round(raw)


@register("chunker", "prism.chunker")
class PrismChunker(BaseChunker):
    """PrismRAG 分块器（过渡版：单粒度自适应）。"""

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        base_chunk_size: int = 300,
        complexity_weight: float = 0.5,
        density_weight: float = 0.5,
        # 兼容 defaults 风格的 "chunk_size: 300" 参数
        chunk_size: int | None = None,
    ) -> None:
        if chunk_size is not None:
            base_chunk_size = chunk_size
        if not (0 < min_chunk_size <= base_chunk_size <= max_chunk_size):
            raise ValueError(
                "require 0 < min_chunk_size <= base_chunk_size <= max_chunk_size; "
                f"got ({min_chunk_size}, {base_chunk_size}, {max_chunk_size})"
            )
        if abs(complexity_weight + density_weight - 1.0) > 1e-6:
            raise ValueError("complexity_weight + density_weight must sum to 1.0")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.base_chunk_size = base_chunk_size
        self.complexity_weight = complexity_weight
        self.density_weight = density_weight

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []

        complexity = estimate_complexity(text)
        density = estimate_entity_density(text)
        score = self.complexity_weight * complexity + self.density_weight * density
        adaptive_size = compute_adaptive_chunk_size(
            score=score,
            min_size=self.min_chunk_size,
            max_size=self.max_chunk_size,
            base_size=self.base_chunk_size,
        )

        # 句子级装箱：比字符硬切少了割裂句子的伪影。
        sentences = split_sentences(text)

        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_len = 0
        idx = 0
        trace_meta = {
            "adaptive_chunk_size": adaptive_size,
            "complexity": round(complexity, 3),
            "density": round(density, 3),
            "score": round(score, 3),
        }

        for sent in sentences:
            sent_len = len(sent) + 1  # +1 是句间空格
            if buf and buf_len + sent_len > adaptive_size:
                chunks.append(self._make_chunk(document, idx, buf, trace_meta))
                idx += 1
                buf, buf_len = [], 0
            buf.append(sent)
            buf_len += sent_len

        if buf:
            chunks.append(self._make_chunk(document, idx, buf, trace_meta))

        return chunks

    @staticmethod
    def _make_chunk(
        document: Document,
        idx: int,
        buf: list[str],
        trace_meta: dict,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{document.doc_id}::{idx}",
            doc_id=document.doc_id,
            index=idx,
            text=" ".join(buf).strip(),
            metadata=dict(trace_meta),
        )


__all__ = ["PrismChunker", "compute_adaptive_chunk_size"]

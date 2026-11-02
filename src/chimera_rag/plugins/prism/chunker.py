"""PrismRAG 分块器 —— 双粒度构建（Phase 1–2）+ 审计修复能力（Phase 3–4）。

「分光」：同一份原文按用途分成两条粒度，各自服务一个消费者。

===============  ====================  ==================  =====================
粒度             目标                  进 extractor        进 vector_store
===============  ====================  ==================  =====================
``fine``         三元组纯度（min D1/D6）  ✅                ❌
``coarse``       证据可读性（min D2/D4）  ❌                ✅
``unified``      两者妥协（= 所有基线）   ✅                ✅
===============  ====================  ==================  =====================

fine 块通过 ``parent_chunk_id`` 绑到覆盖它的 coarse 块，于是检索命中 coarse 之后
可以反查其全部 fine 子块，拿到高纯度三元组——文本证据与图谱证据同时呈现且各自可追溯。

两条粒度都由同一个 0..1 结构分数驱动（:func:`compute_adaptive_chunk_size`），
只是各自套用自己的尺寸区间：复杂/实体密集的文档两层都更小。

``dual_granularity=False`` 保留过渡版单粒度自适应行为（全部 ``unified``），
它是消融实验 E2 的第 2 行，必须始终可达。

尺寸公式::

    size = base - (score - 0.5) * (max_size - min_size)

⚠️ ``min_chunk_size`` / ``max_chunk_size`` / ``base_chunk_size`` / ``chunk_size``
只在 ``dual_granularity=False`` 时生效；双粒度用 ``fine_*`` / ``coarse_*`` 四个键。
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, ChunkDefect, Document, RepairOutcome, Triple
from chimera_rag.interfaces.chunker import BaseChunkAuditor, BaseChunker
from chimera_rag.interfaces.graph_store import BaseGraphStore
from chimera_rag.plugins.prism.audit import PrismAuditor
from chimera_rag.plugins.prism.defects import DefectDetector
from chimera_rag.plugins.prism.features import (
    estimate_complexity,
    estimate_entity_density,
    split_sentences,
)

logger = logging.getLogger(__name__)


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
    双粒度下它被调用两次，一次套 fine 区间、一次套 coarse 区间。
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


def pack_sentences(sentences: Sequence[str], target: int) -> list[list[int]]:
    """把句子索引装箱成不超过 ``target`` 字符的组（Phase 1→2 的合并步）。

    返回索引而非文本，因为父子绑定要按句子归属来算重叠。
    句子永不被切断——这是相比字符硬切滑窗的关键区别。
    """
    if target < 1:
        raise ValueError(f"target must be >= 1, got {target}")
    groups: list[list[int]] = []
    buf: list[int] = []
    buf_len = 0
    for i, sent in enumerate(sentences):
        sent_len = len(sent) + 1  # +1 是句间空格
        if buf and buf_len + sent_len > target:
            groups.append(buf)
            buf, buf_len = [], 0
        buf.append(i)
        buf_len += sent_len
    if buf:
        groups.append(buf)
    return groups


def assign_parents(
    fine_groups: Sequence[Sequence[int]], coarse_groups: Sequence[Sequence[int]]
) -> list[int]:
    """每个 fine 组的 coarse 父块下标（按句子多数归属，平局取靠前者）。"""
    owner: dict[int, int] = {}
    for ci, group in enumerate(coarse_groups):
        for s in group:
            owner[s] = ci
    parents: list[int] = []
    for group in fine_groups:
        tally = Counter(owner[s] for s in group if s in owner)
        if not tally:
            parents.append(0)
            continue
        parents.append(min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0])
    return parents


@register("chunker", "prism.chunker")
class PrismChunker(BaseChunker, BaseChunkAuditor):
    """PrismRAG 分块器：双粒度输出 + D1–D6 审计修复。

    同时实现两个 ABC——``BaseChunker`` 是插槽契约，``BaseChunkAuditor`` 是可选能力。
    能力挂在分块器上而不是新开一个 registry 插槽，是为了让 PrismRAG 只占 ``chunker``
    一个插槽（``docs/innovation-1/design.md`` §9.1），插件开关因此仍是单个布尔值。
    """

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        base_chunk_size: int = 300,
        complexity_weight: float = 0.5,
        density_weight: float = 0.5,
        # 兼容 defaults 风格的 "chunk_size: 300" 参数
        chunk_size: int | None = None,
        *,
        dual_granularity: bool = True,
        fine_min_size: int = 100,
        fine_max_size: int = 250,
        coarse_min_size: int = 400,
        coarse_max_size: int = 800,
        audit_enabled: bool = True,
        repair_enabled: bool = True,
        triple_budget: int = 30,
        detectors: Sequence[DefectDetector] | None = None,
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
        if not (0 < fine_min_size <= fine_max_size):
            raise ValueError(
                f"require 0 < fine_min_size <= fine_max_size; "
                f"got ({fine_min_size}, {fine_max_size})"
            )
        if not (0 < coarse_min_size <= coarse_max_size):
            raise ValueError(
                f"require 0 < coarse_min_size <= coarse_max_size; "
                f"got ({coarse_min_size}, {coarse_max_size})"
            )
        if coarse_min_size < fine_min_size or coarse_max_size < fine_max_size:
            raise ValueError(
                "the coarse layer must not be finer than the fine layer: require "
                "coarse_min_size >= fine_min_size and coarse_max_size >= fine_max_size"
            )
        if triple_budget < 1:
            raise ValueError(f"triple_budget must be >= 1, got {triple_budget}")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.base_chunk_size = base_chunk_size
        self.complexity_weight = complexity_weight
        self.density_weight = density_weight
        self.dual_granularity = dual_granularity
        self.fine_min_size = fine_min_size
        self.fine_max_size = fine_max_size
        self.coarse_min_size = coarse_min_size
        self.coarse_max_size = coarse_max_size
        self.audit_enabled = audit_enabled
        self.repair_enabled = repair_enabled
        self.triple_budget = triple_budget
        self._auditor = PrismAuditor(
            detectors, triple_budget=triple_budget, repair_enabled=repair_enabled
        )

    # ------------------------------------------------------------------
    # Phase 1 + 2
    # ------------------------------------------------------------------
    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []

        complexity = estimate_complexity(text)
        density = estimate_entity_density(text)
        score = self.complexity_weight * complexity + self.density_weight * density
        sentences = split_sentences(text)
        trace = {"complexity": round(complexity, 3), "density": round(density, 3),
                 "score": round(score, 3)}

        if not self.dual_granularity:
            return self._chunk_unified(document, sentences, score, trace)
        return self._chunk_dual(document, sentences, score, trace)

    def _chunk_unified(
        self, document: Document, sentences: list[str], score: float, trace: dict
    ) -> list[Chunk]:
        """过渡版单粒度路径：全部 ``unified``，与 E2 第 2 行等价。"""
        target = compute_adaptive_chunk_size(
            score=score,
            min_size=self.min_chunk_size,
            max_size=self.max_chunk_size,
            base_size=self.base_chunk_size,
        )
        meta = {**trace, "adaptive_chunk_size": target, "layer": "unified"}
        return [
            Chunk(
                chunk_id=f"{document.doc_id}::{i}",
                doc_id=document.doc_id,
                index=i,
                text=" ".join(sentences[i] for i in group).strip(),
                metadata=dict(meta),
            )
            for i, group in enumerate(pack_sentences(sentences, target))
        ]

    def _chunk_dual(
        self, document: Document, sentences: list[str], score: float, trace: dict
    ) -> list[Chunk]:
        """双粒度路径：同一批句子装两次箱，再按多数归属绑父子链。"""
        fine_target = compute_adaptive_chunk_size(
            score=score,
            min_size=self.fine_min_size,
            max_size=self.fine_max_size,
            base_size=(self.fine_min_size + self.fine_max_size) // 2,
        )
        coarse_target = compute_adaptive_chunk_size(
            score=score,
            min_size=self.coarse_min_size,
            max_size=self.coarse_max_size,
            base_size=(self.coarse_min_size + self.coarse_max_size) // 2,
        )

        fine_groups = pack_sentences(sentences, fine_target)
        coarse_groups = pack_sentences(sentences, coarse_target)
        parents = assign_parents(fine_groups, coarse_groups)
        doc = document.doc_id

        chunks = [
            Chunk(
                chunk_id=f"{doc}::coarse::{i}",
                doc_id=doc,
                index=i,
                text=" ".join(sentences[s] for s in group).strip(),
                granularity="coarse",
                metadata={
                    **trace,
                    "adaptive_chunk_size": coarse_target,
                    "layer": "coarse",
                    "fine_children": 0,
                },
            )
            for i, group in enumerate(coarse_groups)
        ]
        chunks.extend(
            Chunk(
                chunk_id=f"{doc}::fine::{i}",
                doc_id=doc,
                index=i,
                text=" ".join(sentences[s] for s in group).strip(),
                granularity="fine",
                parent_chunk_id=f"{doc}::coarse::{parents[i]}",
                metadata={
                    **trace,
                    "adaptive_chunk_size": fine_target,
                    "layer": "fine",
                },
            )
            for i, group in enumerate(fine_groups)
        )

        for ch in chunks:
            if ch.granularity == "coarse":
                ch.metadata["fine_children"] = sum(
                    1 for p in parents if f"{doc}::coarse::{p}" == ch.chunk_id
                )
        logger.debug(
            "prism: %s -> %d coarse + %d fine", doc, len(coarse_groups), len(fine_groups)
        )
        return chunks

    # ------------------------------------------------------------------
    # Phase 3 + 4 — the optional audit capability
    # ------------------------------------------------------------------
    def audit(
        self,
        chunks: Sequence[Chunk],
        triples_by_chunk: Mapping[str, Sequence[Triple]],
        graph: BaseGraphStore | None = None,
    ) -> list[ChunkDefect]:
        if not self.audit_enabled:
            return []
        return self._auditor.audit(chunks, triples_by_chunk, graph)

    def repair(
        self, defects: Sequence[ChunkDefect], chunks: Sequence[Chunk]
    ) -> RepairOutcome:
        return self._auditor.repair(defects, chunks)


__all__ = [
    "PrismChunker",
    "assign_parents",
    "compute_adaptive_chunk_size",
    "pack_sentences",
]

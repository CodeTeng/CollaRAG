"""PrismRAG Phase-2 特征：文本结构信号估计器。

这两个信号是 PrismRAG 的**内部特征**，不是对外宣称的研究贡献。
它们服务于 Phase 2 的预测式合并（决定 fine / coarse 层的初始边界），
真正的贡献在于类型化缺陷 D1–D6 与双粒度 provenance 绑定
（见 ``docs/innovation-1/design.md``）。

刻意保持零依赖：两者都只用正则与计数，不引入 spaCy / NER 模型。
生产环境可无缝替换为 NER 或依存分析版本，公共签名不变。
"""

from __future__ import annotations

import re
from collections import Counter

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b\w+\b")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def estimate_complexity(text: str) -> float:
    """粗略的 0..1 复杂度分数。

    三项信号加权（各自归一化到 0..1）：

    * 词汇多样性 —— type-token ratio，上限 1
    * 句长变异系数 —— std/mean，上限 1
    * 平均词长 —— (avg - 3) / 5，截断到 [0, 1]
    """
    tokens = _WORD_RE.findall(text.lower())
    if len(tokens) < 2:
        return 0.5

    ttr = min(1.0, len(set(tokens)) / len(tokens))
    avg_word_len = sum(len(t) for t in tokens) / len(tokens)
    avg_word_len_norm = max(0.0, min(1.0, (avg_word_len - 3) / 5))

    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) >= 2:
        lengths = [len(_WORD_RE.findall(s)) for s in sentences]
        mean = sum(lengths) / len(lengths)
        if mean > 0:
            var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
            cv = (var**0.5) / mean
            length_signal = max(0.0, min(1.0, cv))
        else:
            length_signal = 0.5
    else:
        length_signal = 0.5

    return float(0.5 * ttr + 0.3 * length_signal + 0.2 * avg_word_len_norm)


def estimate_entity_density(text: str) -> float:
    """看起来像命名实体（首字母大写）的 token 占比。

    刻意粗糙：要的是零依赖信号。句首大写会被扣除以消除假阳性。
    """
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return 0.0
    cap_tokens = _CAP_RE.findall(text)
    sentence_starts: Counter[str] = Counter()
    for sent in _SENTENCE_RE.split(text):
        first = _WORD_RE.search(sent.lstrip())
        if first:
            sentence_starts[first.group(0)] += 1
    adjusted = sum(1 for tok in cap_tokens if sentence_starts.get(tok, 0) == 0)
    return max(0.0, min(1.0, adjusted / len(tokens)))


def split_sentences(text: str) -> list[str]:
    """按句边界切分，供 Phase 2 的装箱与 Phase 4 的边界修复共用。"""
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences or [text]


__all__ = [
    "estimate_complexity",
    "estimate_entity_density",
    "split_sentences",
]

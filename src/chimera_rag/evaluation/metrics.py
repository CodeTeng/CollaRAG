"""Evaluation metrics: EM, token-F1, ROUGE-L (SQuAD + summarisation style).

All use the canonical text normalisation:
1. lowercase
2. strip punctuation
3. remove leading articles (a, an, the)
4. collapse whitespace
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalise(text: str) -> str:
    text = text.lower().translate(_PUNCT_TABLE)
    tokens = [t for t in re.split(r"\s+", text) if t and t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, ref: str) -> float:
    """Return 1.0 iff the two strings match after normalisation."""
    return 1.0 if _normalise(pred) == _normalise(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    """Return token-level F1 between prediction and reference."""
    pred_tokens = _normalise(pred).split()
    ref_tokens = _normalise(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Standard O(len(a) * len(b)) longest-common-subsequence length."""
    if not a or not b:
        return 0
    # Use a 1-D rolling row to keep memory small for long sequences.
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if x == y else max(curr[j - 1], prev[j])
        prev = curr
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    """Return ROUGE-L F1 between prediction and reference.

    Computed as the F1 of the longest common subsequence against the
    normalised, whitespace-tokenised pred/ref. Zero-length inputs give 0.
    """
    pred_tokens = _normalise(pred).split()
    ref_tokens = _normalise(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(pred_tokens)
    r = lcs / len(ref_tokens)
    return 2 * p * r / (p + r)


__all__ = ["exact_match", "rouge_l", "token_f1"]


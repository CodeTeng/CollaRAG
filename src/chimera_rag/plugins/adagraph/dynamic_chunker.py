"""AdaGraph's dynamic, complexity-aware chunker.

Pipeline:

1. Score the document's (complexity, density) on a rough 0..1 scale.
2. Combine into a single ``score`` and call
   :func:`compute_adaptive_chunk_size` to decide the target chunk size.
3. Split on sentence boundaries, packing sentences into chunks up to the
   adaptive target.

The formula (verbatim from the design doc)::

    size = base - (score - 0.5) * (max_size - min_size)

A high score (complex / entity-dense) shrinks chunks so each one carries
finer context; a low score grows chunks so ingestion stays cheap.
"""

from __future__ import annotations

import re
from collections import Counter

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Document
from chimera_rag.interfaces.chunker import BaseChunker


# ---------------------------------------------------------------------------
# Pure formula — easy to unit-test in isolation
# ---------------------------------------------------------------------------
def compute_adaptive_chunk_size(
    score: float,
    *,
    min_size: int,
    max_size: int,
    base_size: int,
) -> int:
    """Return the target chunk size given a combined 0..1 ``score``.

    Clamped so the result is always within ``[min_size, max_size]`` even
    when ``score`` is out of range.
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


# ---------------------------------------------------------------------------
# Complexity / density estimators (lightweight, zero-dep)
# ---------------------------------------------------------------------------
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b\w+\b")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def _estimate_complexity(text: str) -> float:
    """Very rough 0..1 complexity score.

    Combines three signals (each normalised to 0..1):

    * lexical diversity — type-token ratio capped at 1
    * sentence-length variability — std/mean capped at 1
    * average word length — (avg - 3) / 5 clamped to [0, 1]
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


def _estimate_entity_density(text: str) -> float:
    """Proportion of tokens that look like named entities (capitalised).

    Deliberately crude: we want a zero-dep signal. The real AdaGraph uses
    an NER model when ``extras[spacy]`` is installed.
    """
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return 0.0
    cap_tokens = _CAP_RE.findall(text)
    # Ignore sentence-initial capitalisations: crude but effective.
    # Count capitalised tokens excluding the first word of each sentence.
    sentence_starts: Counter[str] = Counter()
    for sent in _SENTENCE_RE.split(text):
        first = _WORD_RE.search(sent.lstrip())
        if first:
            sentence_starts[first.group(0)] += 1
    # A fairer count: capitalised tokens that also appear outside sentence-start positions.
    adjusted = sum(1 for tok in cap_tokens if sentence_starts.get(tok, 0) == 0)
    return max(0.0, min(1.0, adjusted / len(tokens)))


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------
@register("chunker", "adagraph.dynamic")
class DynamicChunker(BaseChunker):
    """AdaGraph dynamic chunker."""

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        base_chunk_size: int = 300,
        complexity_weight: float = 0.5,
        density_weight: float = 0.5,
        # also accepted for compatibility with defaults yaml "chunk_size: 300"
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

    # ------------------------------------------------------------------
    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []

        complexity = _estimate_complexity(text)
        density = _estimate_entity_density(text)
        score = self.complexity_weight * complexity + self.density_weight * density
        adaptive_size = compute_adaptive_chunk_size(
            score=score,
            min_size=self.min_chunk_size,
            max_size=self.max_chunk_size,
            base_size=self.base_chunk_size,
        )

        # Sentence-aware packing: fewer split artifacts than fixed windows.
        sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            sentences = [text]

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
            sent_len = len(sent) + 1  # +1 accounts for inter-sentence space
            if buf and buf_len + sent_len > adaptive_size:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::{idx}",
                        doc_id=document.doc_id,
                        index=idx,
                        text=" ".join(buf).strip(),
                        metadata=dict(trace_meta),
                    )
                )
                idx += 1
                buf, buf_len = [], 0
            buf.append(sent)
            buf_len += sent_len

        if buf:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{idx}",
                    doc_id=document.doc_id,
                    index=idx,
                    text=" ".join(buf).strip(),
                    metadata=dict(trace_meta),
                )
            )

        return chunks


__all__ = ["DynamicChunker", "compute_adaptive_chunk_size"]

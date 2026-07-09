"""Evidence processing tools: temporal filter, evidence dedup, chunk summarize, confidence calibrate."""
from __future__ import annotations

import logging
import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.defaults.extractor import _run_async
from chimera_rag.plugins.colla_rag.prompts import CHUNK_SUMMARIZE_PROMPT

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
    r"|\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b"
    r"|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\b(i (?:do not|don'?t) know|not enough (?:context|information)|unknown)\b", re.I,
)
_WORD_RE = re.compile(r"\w+")


class _TemporalFilterArgs(BaseModel):
    year_min: int = Field(default=0, ge=0, le=2100, description="Minimum year (inclusive).")
    year_max: int = Field(default=2100, ge=0, le=2100, description="Maximum year (inclusive).")


class _EvidenceDedupArgs(BaseModel):
    similarity_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Jaccard similarity threshold above which chunks are merged.",
    )


class _ChunkSummarizeArgs(BaseModel):
    query: str = Field(description="Query context for relevance-aware summarization.")
    chunk_id: str = Field(description="ID of the chunk to summarize.")
    max_length: int = Field(default=300, ge=50, le=2000, description="Max summary length in characters.")


class _ConfidenceCalibrateArgs(BaseModel):
    query: str = Field(description="Original query.")
    answer: str = Field(description="Candidate answer to calibrate.")


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / max(union, 1)


def build_evidence_tools(*, llm, accumulator, chunk_lookup) -> list[StructuredTool]:

    def _temporal_filter(year_min: int = 0, year_max: int = 2100) -> dict:
        """Filter accumulated evidence chunks by temporal range (year mentions)."""
        merged = accumulator.merged_result()
        kept = []
        removed = []
        for chunk in merged.chunks:
            years_found = [int(y) for y in _YEAR_RE.findall(chunk.text)]
            if not years_found:
                kept.append(chunk.chunk_id)
                continue
            if any(year_min <= y <= year_max for y in years_found):
                kept.append(chunk.chunk_id)
            else:
                removed.append(chunk.chunk_id)

        return {
            "tool": "temporal_filter",
            "year_range": [year_min, year_max],
            "kept": len(kept),
            "removed": len(removed),
            "removed_ids": removed[:20],
        }

    def _evidence_dedup(similarity_threshold: float = 0.8) -> dict:
        """Deduplicate accumulated evidence chunks by Jaccard token similarity."""
        merged = accumulator.merged_result()
        chunks = merged.chunks

        if not chunks:
            return {"tool": "evidence_dedup", "before": 0, "after": 0, "removed_ids": []}

        tokenized = []
        for c in chunks:
            tokens = {t.lower() for t in _WORD_RE.findall(c.text)}
            tokenized.append(tokens)

        keep_mask = [True] * len(chunks)
        removed_ids = []

        for i in range(len(chunks)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(chunks)):
                if not keep_mask[j]:
                    continue
                sim = _jaccard(tokenized[i], tokenized[j])
                if sim >= similarity_threshold:
                    keep_mask[j] = False
                    removed_ids.append(chunks[j].chunk_id)

        return {
            "tool": "evidence_dedup",
            "before": len(chunks),
            "after": sum(keep_mask),
            "removed_count": len(removed_ids),
            "removed_ids": removed_ids[:20],
        }

    def _chunk_summarize(query: str, chunk_id: str, max_length: int = 300) -> dict:
        """Summarize a single chunk for context compression."""
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            return {"tool": "chunk_summarize", "chunk_id": chunk_id, "error": "chunk not found"}

        prompt = CHUNK_SUMMARIZE_PROMPT.format(
            query=query, chunk_id=chunk_id,
            chunk_text=chunk.text[:2000], max_length=max_length,
        )
        try:
            summary = _run_async(llm.complete(prompt, max_tokens=max_length))
        except Exception as e:
            logger.warning("chunk_summarize LLM call failed: %s", e)
            summary = chunk.text[:max_length]

        return {
            "tool": "chunk_summarize",
            "chunk_id": chunk_id,
            "original_length": len(chunk.text),
            "summary_length": len(summary.strip()),
            "summary": summary.strip(),
        }

    def _confidence_calibrate(query: str, answer: str) -> dict:
        """Calibrate confidence of an answer using multiple heuristic signals."""
        merged = accumulator.merged_result()
        signals = {}

        # Signal 1: evidence coverage
        num_evidence = len(merged.chunks)
        signals["evidence_count"] = num_evidence
        evidence_score = min(1.0, num_evidence / 5.0)

        # Signal 2: answer length adequacy
        answer_len = len(answer.strip())
        if answer_len == 0:
            length_score = 0.0
        elif answer_len < 5:
            length_score = 0.3
        elif answer_len > 2000:
            length_score = 0.7
        else:
            length_score = min(1.0, answer_len / 100.0)

        # Signal 3: refusal detection
        is_refusal = bool(_REFUSAL_RE.search(answer))
        refusal_score = 0.1 if is_refusal else 1.0
        signals["is_refusal"] = is_refusal

        # Signal 4: query-answer token overlap
        q_tokens = {t.lower() for t in _WORD_RE.findall(query)}
        a_tokens = {t.lower() for t in _WORD_RE.findall(answer)}
        overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))
        signals["query_overlap"] = round(overlap, 4)

        # Signal 5: evidence-answer token overlap
        evidence_text = " ".join(c.text for c in merged.chunks[:10])
        e_tokens = {t.lower() for t in _WORD_RE.findall(evidence_text)}
        if e_tokens:
            evidence_overlap = len(a_tokens & e_tokens) / max(1, len(a_tokens))
        else:
            evidence_overlap = 0.0
        signals["evidence_overlap"] = round(evidence_overlap, 4)

        # Weighted combination
        confidence = (
            0.25 * evidence_score
            + 0.10 * length_score
            + 0.20 * refusal_score
            + 0.15 * overlap
            + 0.30 * evidence_overlap
        )
        confidence = max(0.0, min(1.0, confidence))

        return {
            "tool": "confidence_calibrate",
            "confidence": round(confidence, 4),
            "signals": signals,
            "evidence_score": round(evidence_score, 4),
            "length_score": round(length_score, 4),
        }

    return [
        StructuredTool.from_function(
            _temporal_filter, name="temporal_filter",
            description="Filter accumulated evidence by time range (year). Chunks without dates are kept. Use when queries mention time constraints.",
            args_schema=_TemporalFilterArgs,
        ),
        StructuredTool.from_function(
            _evidence_dedup, name="evidence_dedup",
            description="Deduplicate accumulated evidence chunks by Jaccard token similarity. Reduces redundant context before synthesis.",
            args_schema=_EvidenceDedupArgs,
        ),
        StructuredTool.from_function(
            _chunk_summarize, name="chunk_summarize",
            description="Summarize a single evidence chunk for context compression. Use when accumulated evidence exceeds context window.",
            args_schema=_ChunkSummarizeArgs,
        ),
        StructuredTool.from_function(
            _confidence_calibrate, name="confidence_calibrate",
            description="Calibrate answer confidence using multiple signals: evidence coverage, overlap, refusal detection, length. Returns 0-1 score.",
            args_schema=_ConfidenceCalibrateArgs,
        ),
    ]


__all__ = ["build_evidence_tools"]

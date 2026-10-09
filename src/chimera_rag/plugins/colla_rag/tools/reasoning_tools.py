"""Reasoning tools: sub-query decomposition, evidence assessment, reranking."""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.plugins.colla_rag.prompts import SUB_QUERY_DECOMPOSE_PROMPT


class _DecomposeArgs(BaseModel):
    query: str = Field(description="Complex query to decompose.")
    max_sub: int = Field(default=4, ge=1, le=8)


class _AssessArgs(BaseModel):
    query: str = Field(description="Original query to judge relevance against.")


class _RerankArgs(BaseModel):
    query: str = Field(description="Query for reranking.")
    chunk_ids: list[str] = Field(description="Chunk IDs to rerank.")
    top_k: int = Field(default=5, ge=1, le=50)


def build_reasoning_tools(*, llm, reflector, reranker, accumulator, chunk_lookup) -> list[StructuredTool]:
    def _sub_query_decompose(query: str, max_sub: int = 4) -> list[str]:
        prompt = SUB_QUERY_DECOMPOSE_PROMPT.format(query=query)
        raw = _run_async(llm.complete(prompt, max_tokens=300))
        try:
            subs = json.loads(raw.strip())
            if isinstance(subs, list):
                return [str(s) for s in subs[:max_sub]]
        except (json.JSONDecodeError, TypeError):
            pass
        return [query]

    def _assess_evidence(query: str) -> dict:
        merged = accumulator.merged_result()
        draft = Answer(
            text=" ".join(c.text for c in merged.chunks[:10]),
            evidence_chunk_ids=[c.chunk_id for c in merged.chunks],
        )
        feedback = _run_async(reflector.evaluate(Query(text=query), draft))
        return {
            "tool": "assess_evidence",
            "quality": round(float(feedback.quality), 4),
            "relevance": round(float(feedback.relevance), 4),
            "completeness": round(float(feedback.completeness), 4),
            "num_evidence_chunks": len(merged.chunks),
            "suggestion": feedback.suggestion,
        }

    def _rerank(query: str, chunk_ids: list[str], top_k: int = 5) -> dict:
        from chimera_rag.core.types import RetrievalResult

        chunks = [chunk_lookup[cid] for cid in chunk_ids if cid in chunk_lookup]
        if not chunks:
            return {"tool": "rerank", "num_chunks": 0, "chunks": []}
        rr = RetrievalResult(chunks=chunks, scores=[1.0] * len(chunks))
        if reranker is not None:
            rr = _run_async(reranker.rerank(Query(text=query), rr))
            rr_chunks = rr.chunks[:top_k]
        else:
            rr_chunks = chunks[:top_k]
        return {
            "tool": "rerank",
            "num_chunks": len(rr_chunks),
            "chunks": [{"chunk_id": c.chunk_id, "preview": c.text[:160]} for c in rr_chunks],
        }

    return [
        StructuredTool.from_function(_sub_query_decompose, name="sub_query_decompose",
            description="Decompose a complex query into simpler sub-queries.",
            args_schema=_DecomposeArgs),
        StructuredTool.from_function(_assess_evidence, name="assess_evidence",
            description="Score quality/relevance/completeness of gathered evidence.",
            args_schema=_AssessArgs),
        StructuredTool.from_function(_rerank, name="rerank",
            description="Rerank chunks by semantic relevance using a cross-encoder.",
            args_schema=_RerankArgs),
    ]


__all__ = ["build_reasoning_tools"]

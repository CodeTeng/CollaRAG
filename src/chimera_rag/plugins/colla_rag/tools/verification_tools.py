"""Verification & attribution tools: answer verify, claim decompose, source attribution."""
from __future__ import annotations

import json
import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.defaults.extractor import _run_async
from chimera_rag.plugins.colla_rag.prompts import (
    ANSWER_VERIFY_PROMPT,
    CLAIM_DECOMPOSE_PROMPT,
    SOURCE_ATTRIBUTION_PROMPT,
)

logger = logging.getLogger(__name__)


class _AnswerVerifyArgs(BaseModel):
    query: str = Field(description="Original question.")
    draft_answer: str = Field(description="Draft answer to verify.")


class _ClaimDecomposeArgs(BaseModel):
    text: str = Field(description="Text to decompose into atomic claims.")


class _SourceAttributionArgs(BaseModel):
    query: str = Field(description="Original question for context.")
    answer: str = Field(description="Answer to attribute sources for.")


def _safe_parse_json(raw: str, fallback):
    """Parse JSON from LLM output, stripping markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lbr = raw.find("{" if isinstance(fallback, dict) else "[")
        rbr = raw.rfind("}" if isinstance(fallback, dict) else "]")
        if lbr != -1 and rbr != -1 and rbr > lbr:
            try:
                return json.loads(raw[lbr : rbr + 1])
            except json.JSONDecodeError:
                pass
    return fallback


def build_verification_tools(*, llm, accumulator) -> list[StructuredTool]:

    def _answer_verify(query: str, draft_answer: str) -> dict:
        """Verify a draft answer against gathered evidence using LLM."""
        merged = accumulator.merged_result()
        evidence_text = "\n".join(
            f"[{c.chunk_id}] {c.text[:500]}" for c in merged.chunks[:10]
        )
        prompt = ANSWER_VERIFY_PROMPT.format(
            query=query, draft_answer=draft_answer, evidence=evidence_text,
        )
        try:
            raw = _run_async(llm.complete(prompt, max_tokens=512))
        except Exception as e:
            logger.warning("answer_verify LLM call failed: %s", e)
            return {"tool": "answer_verify", "verdict": "error", "reason": str(e)}

        result = _safe_parse_json(raw, {
            "verdict": "unknown",
            "supported_claims": [],
            "unsupported_claims": [],
            "reason": raw[:200],
        })
        result["tool"] = "answer_verify"
        return result

    def _claim_decompose(text: str) -> dict:
        """Decompose text into atomic, independently verifiable claims."""
        prompt = CLAIM_DECOMPOSE_PROMPT.format(text=text)
        try:
            raw = _run_async(llm.complete(prompt, max_tokens=512))
        except Exception as e:
            logger.warning("claim_decompose LLM call failed: %s", e)
            return {"tool": "claim_decompose", "claims": [], "count": 0}

        claims = _safe_parse_json(raw, [])
        if isinstance(claims, list):
            claims = [str(c) for c in claims]
        else:
            claims = []
        return {"tool": "claim_decompose", "claims": claims, "count": len(claims)}

    def _source_attribution(query: str, answer: str) -> dict:
        """Attribute each claim in an answer to its evidence chunk."""
        merged = accumulator.merged_result()
        evidence_text = "\n".join(
            f"[{c.chunk_id}] {c.text[:400]}" for c in merged.chunks[:15]
        )
        prompt = SOURCE_ATTRIBUTION_PROMPT.format(
            answer=answer, evidence_chunks=evidence_text,
        )
        try:
            raw = _run_async(llm.complete(prompt, max_tokens=512))
        except Exception as e:
            logger.warning("source_attribution LLM call failed: %s", e)
            return {"tool": "source_attribution", "attributions": []}

        attributions = _safe_parse_json(raw, [])
        if not isinstance(attributions, list):
            attributions = []
        sourced = sum(1 for a in attributions if isinstance(a, dict) and a.get("chunk_id"))
        return {
            "tool": "source_attribution",
            "attributions": attributions,
            "sourced": sourced,
            "total": len(attributions),
        }

    return [
        StructuredTool.from_function(
            _answer_verify, name="answer_verify",
            description="Verify a draft answer against gathered evidence. Returns verdict (verified/partially_verified/rejected) with reasons.",
            args_schema=_AnswerVerifyArgs,
        ),
        StructuredTool.from_function(
            _claim_decompose, name="claim_decompose",
            description="Decompose text into atomic, independently verifiable claims. Use before fine-grained verification.",
            args_schema=_ClaimDecomposeArgs,
        ),
        StructuredTool.from_function(
            _source_attribution, name="source_attribution",
            description="Attribute each claim in an answer to its evidence chunk_id. Use for traceable citations.",
            args_schema=_SourceAttributionArgs,
        ),
    ]


__all__ = ["build_verification_tools"]

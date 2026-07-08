"""Preprocessing tools: query rewrite, coreference resolution, follow-up merge."""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.defaults.extractor import _run_async
from chimera_rag.plugins.colla_rag.prompts import (
    COREFERENCE_RESOLVE_PROMPT,
    FOLLOWUP_MERGE_PROMPT,
    QUERY_REWRITE_PROMPT,
)


class _RewriteArgs(BaseModel):
    query: str = Field(description="The query to rewrite.")
    context: str = Field(default="", description="Optional context for rewriting.")


class _CorefArgs(BaseModel):
    query: str = Field(description="Query with potential pronouns.")
    history: list[str] = Field(default_factory=list, description="Previous conversation turns.")


class _FollowupArgs(BaseModel):
    query: str = Field(description="The follow-up query.")
    prev_query: str = Field(default="", description="Previous query.")
    prev_answer: str = Field(default="", description="Previous answer.")


def build_preprocess_tools(*, llm, session_memory) -> list[StructuredTool]:
    def _query_rewrite(query: str, context: str = "") -> str:
        prompt = QUERY_REWRITE_PROMPT.format(query=query, context=context)
        return _run_async(llm.complete(prompt, max_tokens=200)).strip()

    def _coreference_resolve(query: str, history: list[str] | None = None) -> str:
        hist_str = " | ".join(history or [])
        prompt = COREFERENCE_RESOLVE_PROMPT.format(history=hist_str, query=query)
        return _run_async(llm.complete(prompt, max_tokens=200)).strip()

    def _followup_merge(query: str, prev_query: str = "", prev_answer: str = "") -> str:
        prompt = FOLLOWUP_MERGE_PROMPT.format(
            query=query, prev_query=prev_query, prev_answer=prev_answer,
        )
        return _run_async(llm.complete(prompt, max_tokens=200)).strip()

    return [
        StructuredTool.from_function(_query_rewrite, name="query_rewrite",
            description="Rewrite a query for clarity and specificity.", args_schema=_RewriteArgs),
        StructuredTool.from_function(_coreference_resolve, name="coreference_resolve",
            description="Resolve pronouns using conversation history.", args_schema=_CorefArgs),
        StructuredTool.from_function(_followup_merge, name="followup_merge",
            description="Merge a follow-up with previous QA into a self-contained query.",
            args_schema=_FollowupArgs),
    ]


__all__ = ["build_preprocess_tools"]

"""AgentFactory (Factory pattern) + ToolSetBuilder (Builder pattern)."""
from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.tools import StructuredTool

from chimera_rag.interfaces.base_agent import BaseAgent

TOOL_MATRIX: dict[str, list[str]] = {
    "preprocessor": [
        "query_rewrite", "coreference_resolve", "followup_merge",
        "read_session_context", "read_shared_memory",
    ],
    "single_hop": [
        "vector_search", "bm25_search", "hybrid_search", "graph_neighbors",
        "entity_extract", "entity_link",
        "assess_evidence", "rerank",
        "read_shared_memory", "read_agent_memory",
        "save_to_file", "load_from_file",
        "answer_verify", "confidence_calibrate",
        "evidence_dedup",
    ],
    "multi_hop": [
        "vector_search", "bm25_search", "hybrid_search",
        "graph_neighbors", "graph_path_search", "graph_community_search",
        "graph_subgraph_extract", "graph_statistics",
        "entity_extract", "relation_extract", "entity_link",
        "sub_query_decompose", "assess_evidence", "rerank",
        "read_session_context", "read_shared_memory", "read_agent_memory",
        "save_to_file", "load_from_file",
        "answer_verify", "claim_decompose", "source_attribution",
        "temporal_filter", "evidence_dedup", "chunk_summarize",
        "confidence_calibrate",
    ],
    "summarization": [
        "vector_search", "bm25_search", "hybrid_search",
        "graph_community_search", "rerank",
        "entity_extract",
        "read_shared_memory", "read_agent_memory",
        "save_to_file", "load_from_file",
        "chunk_summarize", "evidence_dedup",
        "source_attribution", "confidence_calibrate",
    ],
    "other": [
        "hybrid_search", "web_search", "web_fetch",
        "entity_extract",
        "read_shared_memory", "read_agent_memory",
        "save_to_file", "load_from_file",
        "confidence_calibrate",
    ],
}


class ToolSetBuilder:
    @staticmethod
    def for_agent(agent_type: str, all_tools: list[StructuredTool]) -> list[StructuredTool]:
        allowed = set(TOOL_MATRIX.get(agent_type, []))
        return [t for t in all_tools if t.name in allowed]


class AgentFactory:
    _registry: ClassVar[dict[str, type[BaseAgent]]] = {}

    @classmethod
    def register(cls, intent: str):
        def decorator(agent_cls: type[BaseAgent]) -> type[BaseAgent]:
            cls._registry[intent] = agent_cls
            return agent_cls
        return decorator

    @classmethod
    def create(cls, intent: str, *, all_tools: list[StructuredTool],
               memory_manager: Any, config: dict) -> BaseAgent:
        agent_cls = cls._registry.get(intent)
        if agent_cls is None:
            raise ValueError(f"No agent registered for intent {intent!r}")
        tools = ToolSetBuilder.for_agent(intent, all_tools)
        return agent_cls(agent_type=intent, tools=tools, config=config)


__all__ = ["TOOL_MATRIX", "AgentFactory", "ToolSetBuilder"]

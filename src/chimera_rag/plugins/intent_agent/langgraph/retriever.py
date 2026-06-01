"""LangGraphAgentRetriever — public face of innovation #2b.

Registered as ``('retriever', 'intent_agent.langgraph')`` so the standard
:class:`ChimeraRAG` QueryPipeline picks it up by toggling
``query.retriever.active: intent_agent.langgraph`` in config.yaml.

It implements the synchronous :meth:`BaseRetriever.retrieve` contract (so it
is a drop-in for ``DirectRetriever`` / ``IntentAgentRetriever``) while running
a LangGraph agent under the hood via ``_run_async``. The agent dynamically
selects retrieval *tools* (see :mod:`tools`) instead of routing to a single
static strategy — the core difference from innovation #2's static selector.

Dependency injection: ``ChimeraRAG._instantiate_retriever`` detects the
``base_retriever`` constructor parameter and wires a ``DirectRetriever``
plus an ``LLMIntentClassifier`` and the shared ``llm`` provider. Extra
kwargs are tolerated for forward compatibility.
"""

from __future__ import annotations

import logging

from chimera_rag.core.registry import register
from chimera_rag.core.types import Query, RetrievalResult
from chimera_rag.defaults.llm_extractor import _run_async
from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier
from chimera_rag.interfaces.llm_provider import BaseLLMProvider
from chimera_rag.interfaces.memory import BaseMemory
from chimera_rag.interfaces.reflector import BaseReflector
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.plugins.intent_agent.langgraph.graph import build_agent_graph
from chimera_rag.plugins.intent_agent.langgraph.llm_adapter import ChimeraReActChatModel
from chimera_rag.plugins.intent_agent.langgraph.tools import ALL_TOOL_NAMES, ToolContext, build_tools

logger = logging.getLogger(__name__)


@register("retriever", "intent_agent.langgraph")
class LangGraphAgentRetriever(BaseRetriever):
    """Agentic retriever: LLM-orchestrated tool selection via LangGraph."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        classifier: BaseIntentClassifier,
        llm: BaseLLMProvider,
        *,
        memory: BaseMemory | None = None,
        reflector: BaseReflector | None = None,
        tools: list[str] | None = None,
        max_iterations: int = 2,
        quality_threshold: float = 0.6,
        multi_hop_hops: int = 2,
        # Accepted for ChimeraRAG kwarg-injection compatibility / config pass-through.
        llm_protocol: str = "react",
        vector_store=None,
        graph_store=None,
        embedder=None,
        chunk_lookup=None,
        neighbor_hops: int | None = None,
    ) -> None:
        self.base = base_retriever
        self.classifier = classifier
        self.llm = llm
        self.memory = memory
        self.reflector = reflector or self._default_reflector()
        self.enabled_tools = self._resolve_tool_names(tools)
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.multi_hop_hops = multi_hop_hops
        self.llm_protocol = llm_protocol
        # graph_store needed for graph_neighbors tool; fall back to base's.
        self.graph_store = graph_store or getattr(base_retriever, "graph_store", None)
        # Expose chunk_lookup for QueryPipeline compatibility.
        self.chunk_lookup = getattr(base_retriever, "chunk_lookup", {})

    # ------------------------------------------------------------------
    @staticmethod
    def _default_reflector() -> BaseReflector:
        from chimera_rag.plugins.intent_agent.hooks.reflection_hook import HeuristicReflectionHook

        return HeuristicReflectionHook()

    @staticmethod
    def _resolve_tool_names(tools: list[str] | None) -> list[str]:
        if not tools:
            return list(ALL_TOOL_NAMES)
        unknown = [t for t in tools if t not in ALL_TOOL_NAMES]
        if unknown:
            logger.warning("ignoring unknown langgraph tools: %s", unknown)
        resolved = [t for t in tools if t in ALL_TOOL_NAMES]
        return resolved or list(ALL_TOOL_NAMES)

    # ------------------------------------------------------------------
    # BaseRetriever interface — synchronous to plug into QueryPipeline.
    # ------------------------------------------------------------------
    def retrieve(self, query: Query, top_k: int = 5) -> RetrievalResult:
        if self.graph_store is None:
            raise ValueError(
                "LangGraphAgentRetriever requires a graph_store (inject one or use a "
                "base_retriever that exposes .graph_store)."
            )

        context = ToolContext()
        all_tools = build_tools(
            base_retriever=self.base,
            graph_store=self.graph_store,
            context=context,
            reflector=self.reflector,
            default_top_k=top_k,
            multi_hop_hops=self.multi_hop_hops,
        )
        tools = [t for t in all_tools if t.name in self.enabled_tools]

        chat_model = ChimeraReActChatModel(provider=self.llm)
        graph = build_agent_graph(
            chat_model=chat_model,
            tools=tools,
            context=context,
            classifier=self.classifier,
            reflector=self.reflector,
            memory=self.memory,
            quality_threshold=self.quality_threshold,
            max_iterations=self.max_iterations,
        )

        history: list[str] = []
        if self.memory and query.session_id:
            history = self.memory.get_history(query.session_id)

        final_state = graph.invoke(
            {"query": query.text, "session_id": query.session_id},
            {"recursion_limit": 50},
        )

        result = context.merged_result(top_k=top_k)

        # If the agent gathered no evidence at all, fall back to a direct pass
        # so the pipeline never returns empty-handed.
        if not result.chunks:
            result = self.base.retrieve(query, top_k=top_k)

        if self.memory and query.session_id:
            self.memory.append(query.session_id, query.text)

        result.metadata["retriever"] = "langgraph"
        result.metadata["intent"] = final_state.get("intent")
        result.metadata["strategy"] = "langgraph_agent"
        result.metadata["tool_calls"] = final_state.get("tool_call_log", [])
        result.metadata["iterations"] = final_state.get("iteration_count", 0)
        result.metadata["final_quality"] = final_state.get("quality")
        if history:
            result.metadata["history_turns_used"] = len(history)

        if len(result.chunks) > top_k:
            result.chunks = result.chunks[:top_k]
            result.scores = result.scores[:top_k]
        return result


__all__ = ["LangGraphAgentRetriever"]

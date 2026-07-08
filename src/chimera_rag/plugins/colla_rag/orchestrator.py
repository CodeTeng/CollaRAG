"""MultiAgentRetriever — top-level facade (Facade pattern)."""
from __future__ import annotations

import hashlib
import logging

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Query, RetrievalResult
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory
from chimera_rag.plugins.colla_rag.classifier import TreeIntentClassifier
from chimera_rag.plugins.colla_rag.memory.memory_manager import MemoryManager
from chimera_rag.plugins.colla_rag.tools.retrieval_tools import (
    BM25Index,
    ToolAccumulator,
    build_retrieval_tools,
)

logger = logging.getLogger(__name__)


@register("retriever", "colla_rag.multi_agent")
class MultiAgentRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        llm,
        vector_store,
        graph_store,
        embedder,
        chunk_lookup: dict[str, Chunk],
        memory_path: str = ".chimera_memory",
        classifier_config: dict | None = None,
        agent_configs: dict | None = None,
        reflector=None,
        reranker=None,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.chunk_lookup = chunk_lookup
        self.reflector = reflector
        self.reranker = reranker
        self.agent_configs = agent_configs or {}

        self.classifier = TreeIntentClassifier(llm=llm, **(classifier_config or {}))
        self.memory = MemoryManager(base_path=memory_path)
        self.bm25_index = BM25Index.from_chunks(chunk_lookup)

        # Trigger @AgentFactory.register side effects
        import chimera_rag.plugins.colla_rag.agents  # noqa: F401

    def _build_tools(self, accumulator: ToolAccumulator) -> list:
        from chimera_rag.plugins.colla_rag.tools.entity_tools import build_entity_tools
        from chimera_rag.plugins.colla_rag.tools.io_tools import build_io_tools
        from chimera_rag.plugins.colla_rag.tools.memory_tools import build_memory_tools
        from chimera_rag.plugins.colla_rag.tools.preprocess_tools import build_preprocess_tools
        from chimera_rag.plugins.colla_rag.tools.reasoning_tools import build_reasoning_tools
        from chimera_rag.plugins.colla_rag.tools.web_tools import build_web_tools

        all_tools = []
        all_tools.extend(build_retrieval_tools(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
            chunk_lookup=self.chunk_lookup,
            bm25_index=self.bm25_index,
            accumulator=accumulator,
        ))
        all_tools.extend(build_entity_tools(
            llm=self.llm, graph_store=self.graph_store,
        ))
        all_tools.extend(build_preprocess_tools(
            llm=self.llm, session_memory=self.memory.session,
        ))
        all_tools.extend(build_reasoning_tools(
            llm=self.llm,
            reflector=self.reflector or self._default_reflector(),
            reranker=self.reranker,
            accumulator=accumulator,
            chunk_lookup=self.chunk_lookup,
        ))
        all_tools.extend(build_memory_tools(
            shared_memory=self.memory.shared,
            agent_memory=self.memory.agent_private,
            session_memory=self.memory.session,
        ))
        all_tools.extend(build_web_tools())
        all_tools.extend(build_io_tools())
        return all_tools

    @staticmethod
    def _default_reflector():
        from chimera_rag.plugins.colla_rag.tools.quality_evaluator import (
            HeuristicQualityEvaluator,
        )
        return HeuristicQualityEvaluator()

    def retrieve(self, query: Query, top_k: int = 5) -> RetrievalResult:
        # 1. Classify
        intent = self.classifier.classify(query)
        label = intent.label
        logger.info("query classified as %s (conf=%.2f)", label, intent.confidence)

        # 2. Greeting short-circuit
        if label == "greeting":
            greeting_text = _run_async(self.llm.complete(
                f"Respond warmly to this greeting: {query.text}", max_tokens=100,
            ))
            return RetrievalResult(
                metadata={"agent": "greeting", "intent": label,
                          "answer_text": greeting_text.strip(), "strategy": "greeting"},
            )

        # 3. QA cache check
        q_hash = hashlib.md5(query.text.encode()).hexdigest()
        cached = self.memory.shared.get_qa_cache(q_hash)
        if cached:
            return RetrievalResult(
                metadata={"agent": "cache", "intent": label,
                          "answer_text": cached.get("answer", ""), "strategy": "cache"},
            )

        # 4. Preprocess
        accumulator = ToolAccumulator()
        all_tools = self._build_tools(accumulator)
        preprocess_agent = AgentFactory.create(
            "preprocessor", all_tools=all_tools, memory_manager=self.memory, config={},
        )
        ctx = AgentContext(
            session_id=query.session_id,
            shared_memory=self.memory.shared,
            agent_memory=self.memory.agent_private,
        )
        pre_answer = preprocess_agent.execute(query, ctx)
        rewritten = pre_answer.text or query.text

        # 5. Dispatch to specialized agent
        agent_config = {"llm": self.llm, **self.agent_configs.get(label, {})}
        if label == "summarization":
            ctx.tool_context = {"chunk_lookup": self.chunk_lookup}
        agent = AgentFactory.create(
            label, all_tools=all_tools, memory_manager=self.memory, config=agent_config,
        )
        answer = agent.execute(Query(text=rewritten, session_id=query.session_id), ctx)

        # 6. Post-process
        merged = accumulator.merged_result(top_k=top_k)
        merged.metadata["agent"] = label
        merged.metadata["intent"] = label
        merged.metadata["answer_text"] = answer.text
        merged.metadata["strategy"] = answer.strategy_name
        merged.metadata.update(answer.trace)

        self.memory.shared.write("routing_stats", q_hash, {"intent": label, "agent": label})
        if answer.trace.get("quality", 1.0) >= 0.6:
            self.memory.shared.write("qa_cache", q_hash, {"answer": answer.text})
        self.memory.shared.persist()

        self.memory.agent_private.remember(label, {
            "intent": label,
            "query_text": query.text,
            "tool_sequence": answer.trace.get("tool_call_log", []),
            "final_quality": answer.trace.get("quality", 0.0),
            "outcome": "success" if answer.trace.get("quality", 0.5) >= 0.5 else "lesson",
        })

        if query.session_id:
            self.memory.session.append(
                query.session_id, f"Q: {query.text} | A: {answer.text[:200]}",
            )

        return merged


__all__ = ["MultiAgentRetriever"]

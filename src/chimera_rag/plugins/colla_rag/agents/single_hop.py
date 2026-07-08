"""SingleHopAgent — single-round retrieval for simple factual queries."""
from __future__ import annotations

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory
from chimera_rag.plugins.colla_rag.prompts import NATIVE_RAG_GENERATE_PROMPT


@AgentFactory.register("single_hop")
class SingleHopAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        top_k = self.config.get("top_k", 10)
        tool_map = {t.name: t for t in self.tools}

        search_tool = tool_map.get("hybrid_search") or tool_map.get("vector_search")
        result = search_tool.invoke({"query": query.text, "top_k": top_k})
        chunks = result.get("chunks", [])
        evidence = "\n".join(c.get("preview", "") for c in chunks)
        chunk_ids = [c.get("chunk_id", "") for c in chunks]

        if "assess_evidence" in tool_map:
            assessment = tool_map["assess_evidence"].invoke({"query": query.text})
            quality = assessment.get("quality", 1.0)
            context.trace["quality"] = quality
            context.trace["low_confidence"] = quality < self.config.get("quality_threshold", 0.4)

        prompt = NATIVE_RAG_GENERATE_PROMPT.format(evidence=evidence, query=query.text)
        answer_text = _run_async(llm.complete(prompt, max_tokens=512))

        return Answer(
            text=answer_text.strip(),
            evidence_chunk_ids=chunk_ids,
            strategy_name="native_rag",
        )


__all__ = ["SingleHopAgent"]

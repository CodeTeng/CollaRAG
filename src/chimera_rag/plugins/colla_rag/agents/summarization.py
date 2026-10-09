"""SummarizationAgent — retrieve top-N chunks, map each to a summary, reduce into final answer."""
from __future__ import annotations

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory
from chimera_rag.plugins.colla_rag.prompts import MAP_PROMPT, REDUCE_PROMPT


@AgentFactory.register("summarization")
class SummarizationAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        retrieve_top_k = self.config.get("retrieve_top_k", 50)
        tool_map = {t.name: t for t in self.tools}

        search_tool = tool_map.get("hybrid_search") or tool_map.get("vector_search")
        result = search_tool.invoke({"query": query.text, "top_k": retrieve_top_k})
        chunks_data = result.get("chunks", [])

        chunk_lookup = {}
        if context.tool_context and isinstance(context.tool_context, dict):
            chunk_lookup = context.tool_context.get("chunk_lookup", {})

        texts = []
        chunk_ids = []
        for c in chunks_data:
            cid = c.get("chunk_id", "")
            chunk_ids.append(cid)
            ch = chunk_lookup.get(cid)
            if ch and hasattr(ch, "text"):
                texts.append(ch.text)
            else:
                texts.append(c.get("preview", ""))

        if not texts:
            return Answer(text="No relevant information found.", strategy_name="map_reduce")

        summaries = []
        for text in texts:
            prompt = MAP_PROMPT.format(query=query.text, text=text[:1000])
            summary = _run_async(llm.complete(prompt, max_tokens=200))
            summaries.append(summary.strip())

        combined = "\n---\n".join(f"[{i+1}] {s}" for i, s in enumerate(summaries) if s)
        reduce_prompt = REDUCE_PROMPT.format(query=query.text, summaries=combined)
        final = _run_async(llm.complete(reduce_prompt, max_tokens=1024))

        context.trace["map_count"] = len(summaries)
        return Answer(
            text=final.strip(),
            evidence_chunk_ids=chunk_ids,
            strategy_name="map_reduce",
        )


__all__ = ["SummarizationAgent"]

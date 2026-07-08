"""OtherAgent — search the web + optional local knowledge, then generate."""
from __future__ import annotations

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory
from chimera_rag.plugins.colla_rag.prompts import WEBSEARCH_GENERATE_PROMPT


@AgentFactory.register("other")
class OtherAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        tool_map = {t.name: t for t in self.tools}

        web_results_text = ""
        if "web_search" in tool_map:
            results = tool_map["web_search"].invoke({"query": query.text, "max_results": 5})
            if isinstance(results, list):
                web_results_text = "\n".join(
                    f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
                )

        local_text = ""
        if "hybrid_search" in tool_map:
            local = tool_map["hybrid_search"].invoke({"query": query.text, "top_k": 5})
            chunks = local.get("chunks", [])
            local_text = "\n".join(c.get("preview", "") for c in chunks)

        prompt = WEBSEARCH_GENERATE_PROMPT.format(
            web_results=web_results_text or "(none)",
            local_knowledge=local_text or "(none)",
            query=query.text,
        )
        answer_text = _run_async(llm.complete(prompt, max_tokens=1024))

        context.trace["web_results_count"] = (
            len(web_results_text.split("\n")) if web_results_text else 0
        )
        return Answer(text=answer_text.strip(), strategy_name="web_search")


__all__ = ["OtherAgent"]

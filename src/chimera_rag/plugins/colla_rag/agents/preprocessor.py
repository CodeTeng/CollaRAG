"""PreprocessorAgent — query rewriting, coreference resolution, follow-up merging."""
from __future__ import annotations

from chimera_rag.core.types import Answer, Query
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory


@AgentFactory.register("preprocessor")
class PreprocessorAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        tool_map = {t.name: t for t in self.tools}
        rewritten = query.text

        if "query_rewrite" in tool_map:
            rewritten = tool_map["query_rewrite"].invoke({"query": rewritten, "context": ""})

        context.trace["rewritten_query"] = rewritten
        context.trace["original_query"] = query.text
        return Answer(text=rewritten)


__all__ = ["PreprocessorAgent"]

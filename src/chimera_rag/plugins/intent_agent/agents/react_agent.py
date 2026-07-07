"""ReActAgent — multi-step reasoning with tool selection and reflection loop."""
from __future__ import annotations

import json
import logging
import re

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.intent_agent.agent_factory import AgentFactory
from chimera_rag.plugins.intent_agent.prompts import REACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


@AgentFactory.register("multi_hop")
class ReActAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        max_iters = self.config.get("max_iterations", 3)
        quality_threshold = self.config.get("quality_threshold", 0.6)
        tool_map = {t.name: t for t in self.tools}

        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in self.tools)
        conversation = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descs, query=query.text)
        tool_call_log: list[str] = []

        for iteration in range(max_iters):
            raw = _run_async(llm.complete(conversation, max_tokens=1024))
            conversation += "\n" + raw

            final_match = _FINAL_RE.search(raw)
            if final_match and not _ACTION_RE.search(raw):
                context.trace["iterations"] = iteration + 1
                context.trace["tool_call_log"] = tool_call_log
                return Answer(text=final_match.group(1).strip(), strategy_name="react")

            action_match = _ACTION_RE.search(raw)
            if action_match:
                tool_name = action_match.group(1)
                try:
                    tool_input = json.loads(action_match.group(2))
                except json.JSONDecodeError:
                    tool_input = {"query": query.text}

                tool = tool_map.get(tool_name)
                if tool:
                    tool_call_log.append(tool_name)
                    observation = tool.invoke(tool_input)
                    conversation += f"\nObservation: {json.dumps(observation, default=str)[:2000]}\n"
                else:
                    conversation += f"\nObservation: Tool '{tool_name}' not available.\n"
            else:
                context.trace["iterations"] = iteration + 1
                context.trace["tool_call_log"] = tool_call_log
                return Answer(text=raw.strip(), strategy_name="react")

            if "assess_evidence" in tool_map:
                assessment = tool_map["assess_evidence"].invoke({"query": query.text})
                quality = assessment.get("quality", 0.0)
                if quality >= quality_threshold:
                    conversation += (
                        f"\nThought: Evidence quality ({quality:.2f}) is sufficient. "
                        "I should give the final answer.\n"
                    )

        context.trace["iterations"] = max_iters
        context.trace["tool_call_log"] = tool_call_log
        final_prompt = (
            f"{conversation}\nThought: I've gathered enough evidence. "
            "Let me provide a final answer.\nFinal Answer:"
        )
        final_text = _run_async(llm.complete(final_prompt, max_tokens=512))
        return Answer(text=final_text.strip(), strategy_name="react")


__all__ = ["ReActAgent"]

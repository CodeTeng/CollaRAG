"""MultiHopAgent — Plan-Execute-Reflect architecture for multi-hop reasoning."""
from __future__ import annotations

import json
import logging
import re

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory
from chimera_rag.plugins.colla_rag.prompts import (
    PLAN_PROMPT,
    REFLECT_PROMPT,
    STEP_REACT_PROMPT,
    SYNTHESIZE_PROMPT,
)

logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})", re.DOTALL)
_STEP_RESULT_RE = re.compile(r"Step Result:\s*(.*)", re.DOTALL)


@AgentFactory.register("multi_hop")
class MultiHopAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        quality_threshold = self.config.get("quality_threshold", 0.6)
        max_plan_steps = self.config.get("max_plan_steps", 5)
        max_step_iterations = self.config.get("max_step_iterations", 3)
        reflect_retries = self.config.get("reflect_retries", 2)
        tool_map = {t.name: t for t in self.tools}
        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in self.tools)
        tool_call_log: list[str] = []

        # --- Phase 1: Plan ---
        plan = self._plan(llm, query.text, tool_descs, max_plan_steps)
        logger.info("plan generated with %d steps", len(plan))

        # --- Phase 2: Execute ---
        step_results = self._execute_plan(
            llm, query.text, plan, tool_descs, tool_map,
            tool_call_log, max_step_iterations,
        )

        # --- Phase 3: Reflect & Synthesize ---
        reflect_count = 0
        quality = 0.0

        if "assess_evidence" in tool_map:
            assessment = tool_map["assess_evidence"].invoke({"query": query.text})
            quality = assessment.get("quality", 0.0)

        while quality < quality_threshold and reflect_count < reflect_retries:
            reflect_count += 1
            logger.info(
                "reflect round %d: quality=%.2f < threshold=%.2f",
                reflect_count, quality, quality_threshold,
            )
            reflection = self._reflect(
                llm, query.text, plan, step_results,
                f"quality={quality:.2f}, threshold={quality_threshold}",
            )
            extra = self._execute_reflection(
                llm, query.text, reflection, tool_descs, tool_map,
                tool_call_log, max_step_iterations,
            )
            step_results.append(f"[Reflection round {reflect_count}] {extra}")

            if "assess_evidence" in tool_map:
                assessment = tool_map["assess_evidence"].invoke({"query": query.text})
                quality = assessment.get("quality", 0.0)

        answer_text = self._synthesize(llm, query.text, step_results)

        context.trace["plan"] = plan
        context.trace["step_results"] = step_results
        context.trace["iterations"] = sum(1 for _ in tool_call_log)
        context.trace["tool_call_log"] = tool_call_log
        context.trace["reflect_count"] = reflect_count
        context.trace["quality"] = quality

        return Answer(text=answer_text.strip(), strategy_name="plan_execute")

    # ------------------------------------------------------------------
    # Phase 1: Plan
    # ------------------------------------------------------------------
    def _plan(
        self, llm, query_text: str, tool_descs: str, max_steps: int,
    ) -> list[dict]:
        prompt = PLAN_PROMPT.format(tool_descriptions=tool_descs, query=query_text)
        raw = _run_async(llm.complete(prompt, max_tokens=512))

        try:
            bracket_start = raw.index("[")
            bracket_end = raw.rindex("]") + 1
            plan = json.loads(raw[bracket_start:bracket_end])
            if isinstance(plan, list):
                return plan[:max_steps]
        except (ValueError, json.JSONDecodeError):
            pass

        return [{"step": query_text, "tools_hint": []}]

    # ------------------------------------------------------------------
    # Phase 2: Execute
    # ------------------------------------------------------------------
    def _execute_plan(
        self,
        llm,
        query_text: str,
        plan: list[dict],
        tool_descs: str,
        tool_map: dict,
        tool_call_log: list[str],
        max_step_iterations: int,
    ) -> list[str]:
        step_results: list[str] = []
        for i, step in enumerate(plan):
            result = self._execute_step(
                llm, query_text, step, i + 1,
                "\n".join(step_results) if step_results else "(none)",
                tool_descs, tool_map, tool_call_log, max_step_iterations,
            )
            step_results.append(f"[Step {i + 1}] {result}")
        return step_results

    def _execute_step(
        self,
        llm,
        query_text: str,
        step: dict,
        step_num: int,
        previous_evidence: str,
        tool_descs: str,
        tool_map: dict,
        tool_call_log: list[str],
        max_iterations: int,
    ) -> str:
        step_desc = step.get("step", query_text)
        conversation = STEP_REACT_PROMPT.format(
            step_num=step_num,
            step_description=step_desc,
            query=query_text,
            previous_evidence=previous_evidence[:2000],
            tool_descriptions=tool_descs,
        )

        for _ in range(max_iterations):
            raw = _run_async(llm.complete(conversation, max_tokens=1024))
            conversation += "\n" + raw

            step_match = _STEP_RESULT_RE.search(raw)
            if step_match and not _ACTION_RE.search(raw):
                return step_match.group(1).strip()

            action_match = _ACTION_RE.search(raw)
            if action_match:
                tool_name = action_match.group(1)
                try:
                    tool_input = json.loads(action_match.group(2))
                except json.JSONDecodeError:
                    tool_input = {"query": query_text}

                tool = tool_map.get(tool_name)
                if tool:
                    tool_call_log.append(tool_name)
                    observation = tool.invoke(tool_input)
                    conversation += f"\nObservation: {json.dumps(observation, default=str)[:2000]}\n"
                else:
                    conversation += f"\nObservation: Tool '{tool_name}' not available.\n"
            else:
                return raw.strip()

        final_prompt = f"{conversation}\nStep Result:"
        return _run_async(llm.complete(final_prompt, max_tokens=256)).strip()

    # ------------------------------------------------------------------
    # Phase 3: Reflect & Synthesize
    # ------------------------------------------------------------------
    def _reflect(
        self,
        llm,
        query_text: str,
        plan: list[dict],
        step_results: list[str],
        quality_feedback: str,
    ) -> str:
        prompt = REFLECT_PROMPT.format(
            query=query_text,
            plan=json.dumps(plan, ensure_ascii=False),
            step_results="\n".join(step_results),
            quality_feedback=quality_feedback,
        )
        return _run_async(llm.complete(prompt, max_tokens=512)).strip()

    def _execute_reflection(
        self,
        llm,
        query_text: str,
        reflection: str,
        tool_descs: str,
        tool_map: dict,
        tool_call_log: list[str],
        max_iterations: int,
    ) -> str:
        return self._execute_step(
            llm, query_text,
            {"step": f"Based on reflection: {reflection}"},
            0, "(reflection round)", tool_descs, tool_map,
            tool_call_log, max_iterations,
        )

    def _synthesize(self, llm, query_text: str, step_results: list[str]) -> str:
        prompt = SYNTHESIZE_PROMPT.format(
            query=query_text,
            all_evidence="\n".join(step_results),
        )
        return _run_async(llm.complete(prompt, max_tokens=1024)).strip()


__all__ = ["MultiHopAgent"]

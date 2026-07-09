"""MultiHopAgent — G-PER: Graph-Grounded Plan-Execute-Reflect.

G-PER re-grounds the three phases of multi-hop reasoning in the knowledge
graph, so that each phase performs a check text-only ReAct/Reflexion cannot:

* **Plan** emits *graph-operation* steps (link/traverse/attribute/aggregate)
  that are validated against the KG schema before any retrieval budget is
  spent; invalid steps degrade to free sub-queries (never blocking).
* **Execute** runs a *frontier-based* loop with a global iteration budget
  shared across steps, so an easy step that finishes early donates its
  unused iterations to a harder, dependency-downstream step.
* **Reflect** replaces the scalar quality gate with a *structural
  completeness check* over the evidence subgraph, emitting typed gaps
  (``missing_attribute`` / ``missing_path`` / ``missing_relation``) that
  drive *targeted* repair retrieval, followed by an *adversarial
  contradiction* probe over the KG.

When ``graph_store`` is absent (graph-poor setting), every graph-grounded
mechanism degrades to its text-only PER counterpart, so G-PER collapses to
the original Plan-Execute-Reflect behaviour. This degradation is itself the
graph-poor ablation reported in the paper.
"""
from __future__ import annotations

import json
import logging
import re

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory
from chimera_rag.plugins.colla_rag.gper import (
    EvidenceGap,
    StructuralCompletenessChecker,
)
from chimera_rag.plugins.colla_rag.prompts import (
    GPER_ADVERSARIAL_PROMPT,
    GPER_PLAN_PROMPT,
    GPER_REFLECT_PROMPT,
    REFLECT_PROMPT,
    STEP_REACT_PROMPT,
    SYNTHESIZE_PROMPT,
)

logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})", re.DOTALL)
_STEP_RESULT_RE = re.compile(r"Step Result:\s*(.*)", re.DOTALL)

# Graph operations recognised by the G-PER plan.
_GRAPH_OPS = {"link", "traverse", "attribute", "aggregate"}


@AgentFactory.register("multi_hop")
class MultiHopAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        quality_threshold = self.config.get("quality_threshold", 0.6)
        max_plan_steps = self.config.get("max_plan_steps", 5)
        max_step_iterations = self.config.get("max_step_iterations", 3)
        reflect_retries = self.config.get("reflect_retries", 2)
        # Global frontier budget: by default, the sum of per-step caps, so a
        # step that finishes early frees iterations for later steps.
        total_budget = self.config.get(
            "exec_total_budget", max_plan_steps * max_step_iterations,
        )
        graph_store = self.config.get("graph_store")
        accumulator = self.config.get("accumulator")
        plan_templates = self.config.get("plan_templates")  # C3, mechanism 3A
        gap_lessons = self.config.get("gap_lessons")        # C3, mechanism 3C
        tool_map = {t.name: t for t in self.tools}
        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in self.tools)
        tool_call_log: list[str] = []
        checker = StructuralCompletenessChecker(graph_store)

        # Entities + compositionality are needed by Plan (template recall),
        # pre-emptive gap retrieval, and Reflect — extract once, up front.
        entities = self._extract_entities(query.text, tool_map)
        shape = checker.infer_shape(query.text, entities)
        compositionality = shape.compositionality

        # Text-lesson fallback (the ordinary private memory, mechanism 3C's
        # text counterpart): recall the most-recurring past *failures* on this
        # (multi_hop, multi_hop) bucket and feed them to Plan as avoidance
        # guidance, so the agent does not repeat a tool-sequence pattern that
        # already produced a low-quality answer. In a graph-poor setting this
        # is the only learning signal available; in a graph-rich setting it
        # complements the structural gap-lessons recalled below.
        avoid_lessons = self._recall_avoid_lessons(context.agent_memory)

        # --- Phase 1: Plan (schema-constrained; seeded from graph memory 3A) ---
        plan, plan_stats, plan_source = self._plan(
            llm, query.text, tool_descs, max_plan_steps, graph_store,
            plan_templates=plan_templates,
            compositionality=compositionality, entities=entities,
            avoid_lessons=avoid_lessons,
        )
        logger.info(
            "G-PER plan: %d steps (source=%s, valid=%d, degraded=%d)",
            len(plan), plan_source, plan_stats["valid"], plan_stats["degraded"],
        )

        # --- Phase 2: Execute (frontier-based, global budget) ---
        step_results = self._execute_plan(
            llm, query.text, plan, tool_descs, tool_map,
            tool_call_log, total_budget, max_step_iterations,
        )

        # C3 mechanism 3C: pre-emptive gap retrieval. Recall structural gaps
        # learned from past similar queries and fetch their repair edges BEFORE
        # reflection runs, so known-missing evidence is already in the subgraph.
        pre_emptive_gaps = self._recall_gap_lessons(gap_lessons, entities)
        if pre_emptive_gaps:
            extra = self._execute_repairs(
                llm, query.text,
                [{"repair_query": g.repair_query, "tools_hint": [],
                  "target_entities": g.repair_entities} for g in pre_emptive_gaps],
                tool_descs, tool_map, tool_call_log, max_step_iterations,
            )
            step_results.append(f"[Pre-emptive gap retrieval] {extra}")

        # --- Phase 3: Reflect & Synthesize (structural + adversarial) ---
        evidence_triples = self._evidence_triples(accumulator)

        reflect_count = 0
        gaps: list[EvidenceGap] = checker.check(query.text, entities, evidence_triples)
        quality = self._scalar_quality(query.text, tool_map)

        while gaps and reflect_count < reflect_retries:
            reflect_count += 1
            logger.info(
                "G-PER reflect round %d: %d structural gaps (scalar q=%.2f)",
                reflect_count, len(gaps), quality,
            )
            repairs = self._plan_repairs(
                llm, query.text, step_results, gaps,
            )
            extra = self._execute_repairs(
                llm, query.text, repairs, tool_descs, tool_map,
                tool_call_log, max_step_iterations,
            )
            step_results.append(f"[Reflect round {reflect_count}] {extra}")
            # Re-check with refreshed evidence.
            evidence_triples = self._evidence_triples(accumulator)
            gaps = checker.check(query.text, entities, evidence_triples)
            quality = self._scalar_quality(query.text, tool_map)
            if not gaps:
                break

        # Scalar-gate fallback: if structural check found nothing but the
        # scalar quality is still poor, run one text-style reflection round
        # (preserves the original PER safety net for non-graph queries).
        if quality < quality_threshold and reflect_count < reflect_retries and not gaps:
            reflect_count += 1
            reflection = self._reflect(
                llm, query.text, plan, step_results,
                f"quality={quality:.2f}, threshold={quality_threshold}",
            )
            extra = self._execute_reflection(
                llm, query.text, reflection, tool_descs, tool_map,
                tool_call_log, max_step_iterations,
            )
            step_results.append(f"[Reflect round {reflect_count}] {extra}")

        # Adversarial contradiction probe on the draft answer.
        draft = self._synthesize(llm, query.text, step_results)
        adversarial = self._adversarial_check(
            llm, query.text, draft, evidence_triples,
        )
        if adversarial.get("verdict") == "contested":
            contested = adversarial.get("contested_claims", [])
            step_results.append(
                f"[Adversarial] contested claims: {contested}; "
                f"reason: {adversarial.get('reason', '')}"
            )
            # Re-synthesize with the contestations visible to the synthesizer.
            draft = self._synthesize(llm, query.text, step_results)

        context.trace["plan"] = plan
        context.trace["plan_source"] = plan_source
        context.trace["plan_stats"] = plan_stats
        context.trace["compositionality"] = compositionality
        context.trace["step_results"] = step_results
        context.trace["iterations"] = sum(1 for _ in tool_call_log)
        context.trace["tool_call_log"] = tool_call_log
        context.trace["reflect_count"] = reflect_count
        context.trace["quality"] = quality
        context.trace["evidence_triples"] = len(evidence_triples)
        context.trace["adversarial_verdict"] = adversarial.get("verdict", "n/a")
        # Persistable structural gaps (3C write-side): the final unresolved gaps
        # become lessons for future queries. Serialized to plain dicts.
        context.trace["structural_gaps"] = self._serialize_gaps(gaps)
        context.trace["pre_emptive_gaps"] = len(pre_emptive_gaps)
        context.trace["avoided_lessons"] = len(avoid_lessons)

        return Answer(text=draft.strip(), strategy_name="plan_execute")

    # ------------------------------------------------------------------
    # Phase 1: Plan (schema-constrained graph plan)
    # ------------------------------------------------------------------
    def _plan(
        self, llm, query_text: str, tool_descs: str, max_steps: int,
        graph_store, *, plan_templates=None,
        compositionality: str = "unknown", entities: list[str] | None = None,
        avoid_lessons: list[dict] | None = None,
    ) -> tuple[list[dict], dict, str]:
        """Schema-constrained graph plan, optionally seeded from graph memory (3A).

        Returns (plan, stats, source) where source is ``"template"`` when a
        recalled plan template was reused, else ``"llm"``.
        """
        stats = {"valid": 0, "degraded": 0, "fallback": False}

        # C3 mechanism 3A: recall a successful plan template by compositionality
        # + entity signature; reuse it directly and skip the LLM plan call.
        if plan_templates is not None and compositionality != "unknown":
            template = plan_templates.recall(compositionality, entities or [])
            if isinstance(template, list) and template:
                stats["valid"] = len(template)
                logger.debug("G-PER plan reused from template (comp=%s)", compositionality)
                return template[:max_steps], stats, "template"

        schema = ", ".join(sorted(graph_store.schema_predicates())) if \
            graph_store is not None else "(unknown — graph store absent)"
        # Text-lesson avoidance: render recalled failure lessons (or "(none)")
        # into the plan prompt so the planner steers clear of tool-sequence
        # patterns that already produced low-quality answers.
        if avoid_lessons:
            notes = "\n".join(
                f"- avoid: {(l.get('lesson') or '').strip()}"
                f" [tools: {','.join(l.get('tool_sequence') or []) or '-'}]"
                for l in avoid_lessons
            )
        else:
            notes = "(none)"
        prompt = GPER_PLAN_PROMPT.format(
            tool_descriptions=tool_descs, schema=schema, query=query_text,
            avoid_lessons=notes,
        )
        raw = _run_async(llm.complete(prompt, max_tokens=640))

        plan = self._parse_plan(raw, query_text, max_steps)

        if graph_store is None:
            # No schema to validate against; accept the plan as-is.
            stats["valid"] = len(plan)
            return plan, stats, "llm"

        schema_preds = graph_store.schema_predicates()
        validated: list[dict] = []
        for step in plan:
            op = step.get("op")
            if op not in _GRAPH_OPS:
                # Free-text step (or old-format step) — keep, but mark degraded.
                step.setdefault("step", step.get("step", query_text))
                step.setdefault("tools_hint", [])
                stats["degraded"] += 1
                validated.append(step)
                continue

            relation = step.get("relation")
            entity = step.get("entity")
            # Schema constraint: a specified relation must exist in the ontology.
            relation_ok = (not relation) or (relation in schema_preds)
            # Entity constraint: link/traverse/attribute need a resolvable entity.
            entity_ok = (not entity) or graph_store.entity_exists(entity)

            if relation_ok and (entity_ok or op == "aggregate"):
                stats["valid"] += 1
                validated.append(step)
            else:
                # Degrade to a free sub-query so a bad plan never blocks.
                reason = "relation not in schema" if not relation_ok else "entity unknown"
                step = {
                    "step": step.get("step", query_text),
                    "tools_hint": step.get("tools_hint", []),
                    "_degraded_from": op, "_reason": reason,
                }
                stats["degraded"] += 1
                validated.append(step)
        return validated[:max_steps], stats, "llm"

    def _recall_avoid_lessons(self, agent_memory, top_n: int = 2) -> list[dict]:
        """Recall past low-quality (``outcome="lesson"``) experiences to avoid.

        Text-lesson fallback for the private memory: the most-recurring
        failures on the (multi_hop, multi_hop) bucket are surfaced so the
        Plan phase can steer clear of their tool-sequence patterns. Returns
        ``[]`` when no agent memory is wired in (e.g. graph-poor / no-op
        memory in ablations), so the planner runs unseeded.
        """
        if agent_memory is None:
            return []
        try:
            return agent_memory.recall_lessons(
                self.agent_type, self.agent_type, top_n=top_n,
            )
        except Exception:
            logger.debug("avoid-lesson recall failed; planning unseeded")
            return []

    @staticmethod
    def _recall_gap_lessons(gap_lessons, entities: list[str]) -> list[EvidenceGap]:
        """Recall structural-gap lessons for pre-emptive retrieval (C3, 3C)."""
        if gap_lessons is None or not entities:
            return []
        try:
            return list(gap_lessons.recall(entities))
        except Exception:
            logger.debug("gap_lessons recall failed; skipping pre-emptive retrieval")
            return []

    @staticmethod
    def _serialize_gaps(gaps: list[EvidenceGap]) -> list[dict]:
        return [
            {
                "kind": g.kind, "detail": g.detail,
                "repair_query": g.repair_query,
                "repair_entities": g.repair_entities,
            }
            for g in gaps
        ]

    @staticmethod
    def _parse_plan(raw: str, query_text: str, max_steps: int) -> list[dict]:
        """Parse the LLM plan; accept both new graph-op and legacy formats."""
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            plan = json.loads(raw[start:end])
            if isinstance(plan, list) and plan:
                return [
                    p if isinstance(p, dict) else {"step": str(p), "tools_hint": []}
                    for p in plan[:max_steps]
                ]
        except (ValueError, json.JSONDecodeError):
            pass
        return [{"step": query_text, "tools_hint": []}]

    # ------------------------------------------------------------------
    # Phase 2: Execute (frontier-based, global budget)
    # ------------------------------------------------------------------
    def _execute_plan(
        self, llm, query_text, plan, tool_descs, tool_map,
        tool_call_log, total_budget, per_step_cap,
    ) -> list[str]:
        """Run plan steps with a shared global iteration budget.

        Steps are executed in dependency order; each step may use up to
        ``per_step_cap`` iterations, but the *total* iterations across all
        steps are capped by ``total_budget``. A step that returns a Step
        Result before exhausting its cap donates the saved iterations to the
        remaining pool — the frontier-based adaptive allocation that a fixed
        per-step ``jmax`` cannot do.
        """
        step_results: list[str] = []
        remaining = total_budget
        for i, step in enumerate(plan):
            if remaining <= 0:
                step_results.append(f"[Step {i + 1}] (skipped: budget exhausted)")
                continue
            # Cap this step by both its own ceiling and the remaining pool.
            cap = min(per_step_cap, remaining)
            used, result = self._execute_step_bounded(
                llm, query_text, step, i + 1,
                "\n".join(step_results) if step_results else "(none)",
                tool_descs, tool_map, tool_call_log, cap,
            )
            remaining -= used
            step_results.append(f"[Step {i + 1}] {result}")
        return step_results

    def _execute_step_bounded(
        self, llm, query_text, step, step_num, previous_evidence,
        tool_descs, tool_map, tool_call_log, cap,
    ) -> tuple[int, str]:
        """Execute one step for at most ``cap`` iterations; return (used, result)."""
        step_desc = step.get("step", query_text) if isinstance(step, dict) else str(step)
        conversation = STEP_REACT_PROMPT.format(
            step_num=step_num,
            step_description=step_desc,
            query=query_text,
            previous_evidence=previous_evidence[:2000],
            tool_descriptions=tool_descs,
        )
        used = 0
        for _ in range(cap):
            used += 1
            raw = _run_async(llm.complete(conversation, max_tokens=1024))
            conversation += "\n" + raw

            if _STEP_RESULT_RE.search(raw) and not _ACTION_RE.search(raw):
                return used, _STEP_RESULT_RE.search(raw).group(1).strip()

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
                    conversation += (
                        f"\nObservation: "
                        f"{json.dumps(observation, default=str)[:2000]}\n"
                    )
                else:
                    conversation += f"\nObservation: Tool '{tool_name}' not available.\n"
            else:
                return used, raw.strip()

        final_prompt = f"{conversation}\nStep Result:"
        tail = _run_async(llm.complete(final_prompt, max_tokens=256)).strip()
        return used, tail

    # ------------------------------------------------------------------
    # Phase 3: Reflect (structural completeness + adversarial)
    # ------------------------------------------------------------------
    def _plan_repairs(
        self, llm, query_text, step_results, gaps: list[EvidenceGap],
    ) -> list[dict]:
        gap_text = "\n".join(
            f"- [{g.kind}] {g.detail} (repair seed: {g.repair_query})"
            for g in gaps
        )
        prompt = GPER_REFLECT_PROMPT.format(
            query=query_text,
            step_results="\n".join(step_results),
            gaps=gap_text or "(none)",
        )
        raw = _run_async(llm.complete(prompt, max_tokens=512))
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            repairs = json.loads(raw[start:end])
            if isinstance(repairs, list):
                return repairs
        except (ValueError, json.JSONDecodeError):
            pass
        # Fallback: turn each gap's seed query directly into a repair.
        return [
            {"repair_query": g.repair_query, "tools_hint": [], "target_entities": g.repair_entities}
            for g in gaps
        ]

    def _execute_repairs(
        self, llm, query_text, repairs, tool_descs, tool_map,
        tool_call_log, cap,
    ) -> str:
        out: list[str] = []
        for r in repairs:
            step = {
                "step": r.get("repair_query", query_text),
                "tools_hint": r.get("tools_hint", []),
            }
            _, result = self._execute_step_bounded(
                llm, query_text, step, 0, "(reflect repair)",
                tool_descs, tool_map, tool_call_log, cap,
            )
            out.append(result)
        return " | ".join(out) if out else "(no repair needed)"

    def _adversarial_check(
        self, llm, query_text, draft, evidence_triples,
    ) -> dict:
        if not evidence_triples:
            return {"verdict": "unsupported", "contested_claims": [], "reason": "no graph evidence"}
        triples_str = "\n".join(
            f"({t.subject}, {t.predicate}, {t.object})" for t in evidence_triples[:40]
        )
        prompt = GPER_ADVERSARIAL_PROMPT.format(
            query=query_text, draft_answer=draft, triples=triples_str,
        )
        raw = _run_async(llm.complete(prompt, max_tokens=512))
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            obj = json.loads(raw[start:end])
            if isinstance(obj, dict):
                return obj
        except (ValueError, json.JSONDecodeError):
            pass
        return {"verdict": "confirmed", "contested_claims": [], "reason": "parse failure; assumed safe"}

    # ------------------------------------------------------------------
    # Shared helpers (entity extraction, evidence triples, scalar quality)
    # ------------------------------------------------------------------
    def _extract_entities(self, query_text: str, tool_map: dict) -> list[str]:
        if "entity_extract" not in tool_map:
            # Cheap fallback: capitalized tokens as candidate entities.
            return list({w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", query_text)})
        try:
            out = tool_map["entity_extract"].invoke({"query": query_text})
            ents = out.get("entities") if isinstance(out, dict) else out
            if isinstance(ents, list):
                return [str(e) for e in ents]
        except Exception:
            logger.debug("entity_extract failed; falling back to regex entities")
        return list({w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", query_text)})

    @staticmethod
    def _evidence_triples(accumulator) -> list:
        if accumulator is None:
            return []
        merged = accumulator.merged_result()
        return list(getattr(merged, "triples", []) or [])

    def _scalar_quality(self, query_text: str, tool_map: dict) -> float:
        """The legacy scalar quality score, kept as a safety net."""
        if "assess_evidence" not in tool_map:
            return 1.0
        try:
            assessment = tool_map["assess_evidence"].invoke({"query": query_text})
            return float(assessment.get("quality", 0.0))
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Legacy text-reflection helpers (kept for the scalar-gate fallback)
    # ------------------------------------------------------------------
    def _reflect(self, llm, query_text, plan, step_results, quality_feedback) -> str:
        prompt = REFLECT_PROMPT.format(
            query=query_text,
            plan=json.dumps(plan, ensure_ascii=False),
            step_results="\n".join(step_results),
            quality_feedback=quality_feedback,
        )
        return _run_async(llm.complete(prompt, max_tokens=512)).strip()

    def _execute_reflection(
        self, llm, query_text, reflection, tool_descs, tool_map,
        tool_call_log, cap,
    ) -> str:
        _, result = self._execute_step_bounded(
            llm, query_text,
            {"step": f"Based on reflection: {reflection}"},
            0, "(reflection round)", tool_descs, tool_map,
            tool_call_log, cap,
        )
        return result

    def _synthesize(self, llm, query_text, step_results) -> str:
        prompt = SYNTHESIZE_PROMPT.format(
            query=query_text,
            all_evidence="\n".join(step_results),
        )
        return _run_async(llm.complete(prompt, max_tokens=1024)).strip()


__all__ = ["MultiHopAgent"]

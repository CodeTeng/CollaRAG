"""IntentAgentRetriever: the public face of innovation #2.

Registered as ``('retriever', 'intent_agent.strategy_selector')`` so the
standard :class:`ChimeraRAG` QueryPipeline picks it up just by toggling
``query.retriever.active: intent_agent.strategy_selector`` in config.yaml.

Orchestration (Memory → Reasoning → Decision → Execution):

1. **Memory** — if a session_id is present and a Memory implementation
   is bound, fetch history and inject into plan.params['history'].
2. **Reasoning** — the :class:`LLMIntentClassifier` + :class:`IntentReasoner`
   produce a :class:`ReasoningPlan`.
3. **Decision** — a :class:`StrategySelector` maps plan.strategy_name
   to a concrete :class:`BaseRetrievalStrategy` via the Registry.
4. **Execution** — run the strategy; optionally rerank; optionally
   reflect. **Self-Reflection** (innovation #2d): if a reflector is
   wired and produces ``quality < threshold``, the loop escalates the
   strategy via :meth:`IntentReasoner.replan` and re-retrieves, up to
   ``max_reflection_iterations`` rounds. Verbal ``reflection_text`` is
   injected into each re-plan so the agent avoids repeating the same
   mistake — Reflexion-style self-correction (Shinn et al., 2023).
"""

from __future__ import annotations

import logging

from chimera_rag.core.registry import get_registry, register
from chimera_rag.core.types import (
    Answer,
    QualityFeedback,
    Query,
    ReasoningPlan,
    RetrievalResult,
)
from chimera_rag.defaults.llm_extractor import _run_async
from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier
from chimera_rag.interfaces.memory import BaseMemory
from chimera_rag.interfaces.reflector import BaseReflector
from chimera_rag.interfaces.reranker import BaseReranker
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.interfaces.strategy import BaseRetrievalStrategy
from chimera_rag.plugins.intent_agent.reasoning import IntentReasoner

logger = logging.getLogger(__name__)


class StrategySelector:
    """Thin wrapper resolving plan.strategy_name -> strategy instance."""

    def __init__(self, base_retriever: BaseRetriever) -> None:
        self.base_retriever = base_retriever
        self._cache: dict[str, BaseRetrievalStrategy] = {}

    def pick(self, plan: ReasoningPlan) -> BaseRetrievalStrategy:
        name = f"intent_agent.{plan.strategy_name}"
        if name in self._cache:
            return self._cache[name]
        try:
            cls = get_registry().get("strategy", name)
        except Exception as e:  # PluginNotFoundError
            logger.warning(
                "strategy %r not registered (%s); falling back to intent_agent.direct",
                name, e,
            )
            cls = get_registry().get("strategy", "intent_agent.direct")
        inst = cls(retriever=self.base_retriever)
        self._cache[name] = inst
        return inst


@register("retriever", "intent_agent.strategy_selector")
class IntentAgentRetriever(BaseRetriever):
    """Public retriever slot for innovation #2."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        classifier: BaseIntentClassifier,
        reasoner: IntentReasoner | None = None,
        memory: BaseMemory | None = None,
        reranker: BaseReranker | None = None,
        reflector: BaseReflector | None = None,
        # Self-Reflection parameters (innovation #2d)
        max_reflection_iterations: int = 3,
        reflection_quality_threshold: float = 0.6,
        # below are accepted for ChimeraRAG kwarg-injection compatibility
        vector_store=None,
        graph_store=None,
        embedder=None,
        chunk_lookup=None,
        neighbor_hops: int | None = None,
    ) -> None:
        self.base = base_retriever
        self.classifier = classifier
        self.reasoner = reasoner or IntentReasoner()
        self.memory = memory
        self.reranker = reranker
        self.reflector = reflector
        self.max_reflection_iterations = max_reflection_iterations
        self.reflection_quality_threshold = reflection_quality_threshold
        self._selector = StrategySelector(base_retriever)
        # Expose chunk_lookup via the base retriever for pipeline compatibility.
        self.chunk_lookup = getattr(base_retriever, "chunk_lookup", {})

    # ------------------------------------------------------------------
    # BaseRetriever interface: keep synchronous to plug into QueryPipeline
    # ------------------------------------------------------------------
    def retrieve(self, query: Query, top_k: int = 5) -> RetrievalResult:
        # Stage 1. Memory
        history: list[str] = []
        if self.memory and query.session_id:
            history = self.memory.get_history(query.session_id)

        # Stage 2. Reasoning — initial plan
        intent = self.classifier.classify(query)
        plan = self.reasoner.plan(query, intent)
        if history:
            plan.params.setdefault("history", history)

        # Stage 3–4: Decision + Execution with Self-Reflection loop
        result = self._execute_with_reflection(query, plan, top_k)

        # Remember the last turn.
        if self.memory and query.session_id:
            self.memory.append(query.session_id, query.text)

        # Annotate metadata for downstream debuggability.
        result.metadata["intent"] = intent.label
        result.metadata["intent_confidence"] = intent.confidence
        result.metadata.setdefault("strategy", plan.strategy_name)

        # Respect top_k contract.
        if len(result.chunks) > top_k:
            result.chunks = result.chunks[:top_k]
            result.scores = result.scores[:top_k]
        return result

    # ------------------------------------------------------------------
    # Self-Reflection loop (innovation #2d)
    # ------------------------------------------------------------------
    def _execute_with_reflection(
        self, query: Query, initial_plan: ReasoningPlan, top_k: int
    ) -> RetrievalResult:
        """Run Decision → Execution, then reflect and re-plan if quality is low.

        Up to ``max_reflection_iterations`` rounds. Each round:
        1. Pick a strategy from the current plan.
        2. Execute it → produce a RetrievalResult.
        3. Optionally rerank.
        4. Reflect: if quality >= threshold, return immediately.
        5. If quality < threshold, escalate via IntentReasoner.replan()
           using the verbal reflection_text, and loop again.

        If the reflector is None or the loop exhausts its iterations,
        the last result is returned (never empty-handed).
        """
        plan = initial_plan
        result: RetrievalResult | None = None
        reflection_iterations = 0
        strategy_trace: list[str] = [plan.strategy_name]
        reflection_texts: list[str] = []

        for _ in range(self.max_reflection_iterations + 1):
            # Stage 3. Decision
            strategy = self._selector.pick(plan)

            # Stage 4. Execution
            result = _run_async(strategy.execute(query, plan))

            # Optional rerank.
            if self.reranker is not None:
                result = _run_async(self.reranker.rerank(query, result))

            # If no reflector is wired, return immediately (no loop).
            if self.reflector is None:
                break

            # Reflect.
            dummy = Answer(text=" ".join(c.text for c in result.chunks))
            feedback: QualityFeedback = _run_async(self.reflector.evaluate(query, dummy))
            result.metadata["reflection_quality"] = feedback.quality

            if feedback.reflection_text:
                reflection_texts.append(feedback.reflection_text)

            # High quality → stop.
            if feedback.quality >= self.reflection_quality_threshold:
                result.metadata["reflection_converged"] = True
                break

            # Low quality → attempt re-plan. If no fallback strategy
            # exists, stop (we've exhausted the ladder).
            reflection_iterations += 1
            if reflection_iterations >= self.max_reflection_iterations:
                result.metadata["reflection_converged"] = False
                logger.info(
                    "Self-Reflection: exhausted %d iterations for query '%s' "
                    "(final quality %.2f < threshold %.2f)",
                    reflection_iterations, query.text[:50],
                    feedback.quality, self.reflection_quality_threshold,
                )
                break

            next_plan = self.reasoner.replan(plan, feedback.reflection_text)
            if next_plan is None:
                result.metadata["reflection_converged"] = False
                logger.info(
                    "Self-Reflection: no fallback strategy after '%s' for query '%s'",
                    plan.strategy_name, query.text[:50],
                )
                break

            logger.info(
                "Self-Reflection round %d: quality=%.2f, escalating '%s' → '%s'",
                reflection_iterations, feedback.quality,
                plan.strategy_name, next_plan.strategy_name,
            )
            strategy_trace.append(next_plan.strategy_name)
            plan = next_plan

        # Final metadata.
        if result is not None:
            result.metadata["reflection_iterations"] = reflection_iterations
            result.metadata["strategy_trace"] = strategy_trace
            if reflection_texts:
                result.metadata["reflection_texts"] = reflection_texts

        return result or RetrievalResult()


__all__ = ["IntentAgentRetriever", "StrategySelector"]

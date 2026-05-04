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
   reflect and log the QualityFeedback. The reflection loop is MVP-bounded
   to one pass (full Reflexion-style iteration lands in a later cycle).
"""

from __future__ import annotations

import logging

from chimera_rag.core.registry import get_registry, register
from chimera_rag.core.types import Query, ReasoningPlan, RetrievalResult
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

        # Stage 2. Reasoning
        intent = self.classifier.classify(query)
        plan = self.reasoner.plan(query, intent)
        if history:
            plan.params.setdefault("history", history)

        # Stage 3. Decision
        strategy = self._selector.pick(plan)

        # Stage 4. Execution (+ optional rerank/reflect hooks)
        result: RetrievalResult = _run_async(strategy.execute(query, plan))

        if self.reranker is not None:
            result = _run_async(self.reranker.rerank(query, result))

        if self.reflector is not None:
            # Reflection here is observability only; actual re-retrieval
            # loops live in AdaptiveExecutor (future).
            from chimera_rag.core.types import Answer  # local to avoid cycles

            dummy = Answer(text=" ".join(c.text for c in result.chunks))
            feedback = _run_async(self.reflector.evaluate(query, dummy))
            result.metadata["reflection_quality"] = feedback.quality

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


__all__ = ["IntentAgentRetriever", "StrategySelector"]

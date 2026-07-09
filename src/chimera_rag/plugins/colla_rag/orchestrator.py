"""MultiAgentRetriever — top-level facade (Facade pattern)."""
from __future__ import annotations

import hashlib
import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Query, RetrievalResult
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory, GraphFootprint
from chimera_rag.plugins.colla_rag.classifier import TreeIntentClassifier
from chimera_rag.plugins.colla_rag.memory.memory_manager import MemoryManager
from chimera_rag.plugins.colla_rag.tools.retrieval_tools import (
    BM25Index,
    ToolAccumulator,
    build_retrieval_tools,
)

logger = logging.getLogger(__name__)

_CAP_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


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
        from chimera_rag.plugins.colla_rag.tools.evidence_tools import build_evidence_tools
        from chimera_rag.plugins.colla_rag.tools.graph_advanced_tools import (
            build_graph_advanced_tools,
        )
        from chimera_rag.plugins.colla_rag.tools.io_tools import build_io_tools
        from chimera_rag.plugins.colla_rag.tools.memory_tools import build_memory_tools
        from chimera_rag.plugins.colla_rag.tools.preprocess_tools import build_preprocess_tools
        from chimera_rag.plugins.colla_rag.tools.reasoning_tools import build_reasoning_tools
        from chimera_rag.plugins.colla_rag.tools.verification_tools import build_verification_tools
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
        all_tools.extend(build_verification_tools(
            llm=self.llm, accumulator=accumulator,
        ))
        all_tools.extend(build_graph_advanced_tools(
            llm=self.llm, graph_store=self.graph_store, accumulator=accumulator,
        ))
        all_tools.extend(build_evidence_tools(
            llm=self.llm, accumulator=accumulator, chunk_lookup=self.chunk_lookup,
        ))
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
        # 3b. Graph-grounded overlap cache (Contribution 3, mechanism 3B): a hit
        # is keyed by entity-neighborhood overlap, not text hash, so it reflects
        # structural reuse of the same evidence neighborhood rather than question
        # duplication — legitimate on non-repeating benchmarks.
        probe_entities = self._probe_entities(query.text)
        if probe_entities:
            ev_hit = self.memory.shared_evidence.lookup(probe_entities)
            if ev_hit is not None and ev_hit.quality >= 0.6:
                return RetrievalResult(
                    metadata={
                        "agent": "overlap_cache", "intent": label,
                        "answer_text": ev_hit.answer, "strategy": "overlap_cache",
                        "overlap_entities": ev_hit.entities,
                    },
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
        # C4: compute the graph footprint and condition the agent's tool subset.
        footprint = self._footprint(probe_entities)
        # G-PER (MultiHop) needs direct graph_store + accumulator access for
        # schema-constrained plan validation and structural completeness checks.
        # The graph-poor ablation sets gper.graph_grounded=False so the
        # graph_store is withheld and G-PER degrades to text-only PER.
        if label == "multi_hop":
            gper_cfg = agent_config.get("gper", {})
            graph_grounded = gper_cfg.get("graph_grounded", True)
            agent_config.setdefault(
                "graph_store", self.graph_store if graph_grounded else None,
            )
            agent_config.setdefault("accumulator", accumulator)
            # C3: graph-grounded private memory feeds G-PER's Plan + Reflect.
            agent_config.setdefault("plan_templates", self.memory.plan_templates)
            agent_config.setdefault("gap_lessons", self.memory.gap_lessons)
        agent = AgentFactory.create(
            label, all_tools=all_tools, memory_manager=self.memory,
            config=agent_config, footprint=footprint,
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

        self.memory.agent_private.remember(label, self._experience_entry(label, query, answer))

        # C3 write-side: persist graph-grounded memory layers.
        self._persist_graph_memory(label, query.text, probe_entities, accumulator, answer)

        if query.session_id:
            self.memory.session.append(
                query.session_id, f"Q: {query.text} | A: {answer.text[:200]}",
            )

        return merged

    # ------------------------------------------------------------------
    # C3/C4 helpers
    # ------------------------------------------------------------------
    def _probe_entities(self, query_text: str) -> list[str]:
        """Cheap capitalized-token entity probe for cache/footprint (no LLM)."""
        return list(dict.fromkeys(_CAP_ENTITY_RE.findall(query_text)))[:8]

    def _footprint(self, entities: list[str]) -> GraphFootprint | None:
        """Compute a GraphFootprint from a quick graph_store probe (C4).

        Returns None when there is no graph store or nothing to probe, so the
        toolset falls back to the static matrix.
        """
        if self.graph_store is None or not entities:
            return None
        hits = [e for e in entities if self.graph_store.entity_exists(e)]
        hit_rate = len(hits) / len(entities) if entities else 0.0
        has_path = False
        if len(hits) >= 2:
            has_path = bool(self.graph_store.shortest_path(hits[0], hits[-1]))
        n_preds = len(self.graph_store.schema_predicates())
        density = "dense" if n_preds >= 10 else ("sparse" if n_preds > 0 else "unknown")
        return GraphFootprint(
            entity_hit_rate=hit_rate,
            has_path=has_path,
            density=density,
            n_entities=len(entities),
        )

    def _persist_graph_memory(
        self, label: str, query_text: str, entities: list[str],
        accumulator: ToolAccumulator, answer,
    ) -> None:
        """Write the graph-grounded memory layers after a run (C3)."""
        merged = accumulator.merged_result()
        triples = list(getattr(merged, "triples", []) or [])
        quality = float(answer.trace.get("quality", 0.0) or 0.0)

        # Shared evidence subgraph (3B) — only when we actually gathered graph evidence.
        if triples or entities:
            self.memory.shared_evidence.store(
                query_text=query_text, entities=entities, triples=triples,
                answer=answer.text, quality=quality,
            )

        # Private plan template (3A) + gap lessons (3C): only MultiHop produces
        # a structural plan and typed gaps.
        if label == "multi_hop":
            plan = answer.trace.get("plan")
            compositionality = answer.trace.get("compositionality")
            if plan and compositionality:
                self.memory.plan_templates.store(
                    compositionality=compositionality, entities=entities,
                    plan=plan, quality=quality,
                )
            gaps = answer.trace.get("structural_gaps")
            if gaps:
                self.memory.gap_lessons.store(entities=entities, gaps=gaps)

        self.memory.persist()

    def _experience_entry(self, label: str, query: Query, answer) -> dict:
        """Build a private-memory entry, deriving a failure lesson when needed.

        On a low-quality run (``quality < 0.5``) the entry is tagged
        ``outcome="lesson"`` and carries a structured ``lesson`` text derived
        from the trace (no extra LLM call): for G-PER this captures the
        unresolved typed gaps and reflect count; for other agents it records
        the query and quality. The MultiHop agent later recalls these lessons
        in its Plan phase to avoid repeating failed tool patterns.
        """
        quality = float(answer.trace.get("quality", 0.0) or 0.0)
        outcome = "success" if quality >= 0.5 else "lesson"
        entry = {
            "intent": label,
            "query_text": query.text,
            "tool_sequence": answer.trace.get("tool_call_log", []),
            "final_quality": quality,
            "outcome": outcome,
        }
        if outcome == "lesson":
            entry["lesson"] = self._derive_lesson(label, query, answer, quality)
        return entry

    @staticmethod
    def _derive_lesson(label: str, query: Query, answer, quality: float) -> str:
        """Derive a structured failure lesson from trace signals (no LLM)."""
        if label == "multi_hop":
            gaps = answer.trace.get("structural_gaps") or []
            gap_summary = "; ".join(
                f"{g.get('kind', '?')}({(g.get('detail') or '')[:60]})"
                for g in gaps[:3]
            ) or "none"
            return (
                f"low_quality q={quality:.2f}; "
                f"reflect={answer.trace.get('reflect_count', 0)}; "
                f"unresolved_gaps=[{gap_summary}]"
            )
        return f"low_quality q={quality:.2f}; query='{query.text[:80]}'"


__all__ = ["MultiAgentRetriever"]

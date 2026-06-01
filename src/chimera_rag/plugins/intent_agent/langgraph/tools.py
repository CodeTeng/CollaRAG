"""LangChain Tools wrapping Chimera-RAG's existing retrieval primitives.

Design principle: **thin wrappers, zero new retrieval logic**. Every tool
delegates to code that already exists (``DirectRetriever`` /
``MultiHopStrategy`` / ``ParallelStrategy`` / ``NetworkXGraphStore`` /
``HeuristicReflectionHook``). The agent (see :mod:`graph`) only decides
*which* tools to call and *in what order* — it never re-implements search.

To keep token cost bounded, tools return a compact structured dict (chunk
ids + short previews + scores) instead of dumping full chunk text into the
LLM context. The retriever drives the real ``chunk_lookup`` -> ``Chunk``
re-hydration at the end (see :mod:`retriever`).

Tools are built per retriever instance via :func:`build_tools`, which
closes over the runtime dependencies (base retriever, graph store, chunk
lookup, reflector). This avoids global singletons that would otherwise be
unsafe under concurrent queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.core.types import Answer, Chunk, Query, ReasoningPlan, RetrievalResult
from chimera_rag.defaults.llm_extractor import _run_async
from chimera_rag.interfaces.graph_store import BaseGraphStore
from chimera_rag.interfaces.reflector import BaseReflector
from chimera_rag.interfaces.retriever import BaseRetriever

# Canonical tool names — also used as the allow-list in config.yaml.
TOOL_VECTOR_SEARCH = "vector_search"
TOOL_GRAPH_NEIGHBORS = "graph_neighbors"
TOOL_MULTI_HOP = "multi_hop_retrieve"
TOOL_PARALLEL = "parallel_retrieve"
TOOL_ASSESS = "assess_evidence"

ALL_TOOL_NAMES = (
    TOOL_VECTOR_SEARCH,
    TOOL_GRAPH_NEIGHBORS,
    TOOL_MULTI_HOP,
    TOOL_PARALLEL,
    TOOL_ASSESS,
)

_PREVIEW_CHARS = 160


# ---------------------------------------------------------------------------
# Shared accumulator: tools push retrieved chunks/triples here so the graph
# can merge results across multiple tool calls (de-duplicated, score-keyed).
# ---------------------------------------------------------------------------
@dataclass
class ToolContext:
    """Mutable per-query accumulator shared by all tools of one retriever run.

    The LangGraph state references this object so that successive tool calls
    within a single query accumulate evidence rather than overwrite it.
    """

    chunk_lookup: dict[str, Chunk] = field(default_factory=dict)
    # chunk_id -> best score seen so far
    collected: dict[str, float] = field(default_factory=dict)
    triples: list = field(default_factory=list)
    _seen_triple_keys: set = field(default_factory=set)

    def add_result(self, result: RetrievalResult) -> None:
        scores = list(result.scores) + [0.0] * (len(result.chunks) - len(result.scores))
        for chunk, score in zip(result.chunks, scores):
            self.chunk_lookup.setdefault(chunk.chunk_id, chunk)
            previous = self.collected.get(chunk.chunk_id)
            if previous is None or score > previous:
                self.collected[chunk.chunk_id] = float(score)
        for triple in result.triples:
            key = (triple.subject, triple.predicate, triple.object)
            if key not in self._seen_triple_keys:
                self._seen_triple_keys.add(key)
                self.triples.append(triple)

    def merged_result(self, top_k: int | None = None) -> RetrievalResult:
        ordered = sorted(self.collected.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            ordered = ordered[:top_k]
        chunks = [self.chunk_lookup[cid] for cid in (cid for cid, _ in ordered)]
        scores = [score for _, score in ordered]
        return RetrievalResult(chunks=chunks, triples=list(self.triples), scores=scores)


# ---------------------------------------------------------------------------
# Pydantic arg schemas (strict validation, matching project style)
# ---------------------------------------------------------------------------
class _QueryTopKArgs(BaseModel):
    query: str = Field(description="The natural-language query to retrieve evidence for.")
    top_k: int = Field(default=5, ge=1, le=50, description="How many chunks to fetch.")


class _GraphNeighborsArgs(BaseModel):
    entity: str = Field(description="An entity name to look up in the knowledge graph.")
    hops: int = Field(default=1, ge=1, le=4, description="Number of graph hops to expand.")


class _QueryArgs(BaseModel):
    query: str = Field(description="The natural-language query.")


class _AssessArgs(BaseModel):
    query: str = Field(description="The original user query, to judge relevance against.")


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"


def _summarise(result: RetrievalResult, label: str) -> dict:
    """Compact, token-friendly summary returned to the LLM."""
    scores = list(result.scores) + [0.0] * (len(result.chunks) - len(result.scores))
    return {
        "tool": label,
        "num_chunks": len(result.chunks),
        "chunks": [
            {"chunk_id": c.chunk_id, "score": round(float(s), 4), "preview": _preview(c.text)}
            for c, s in zip(result.chunks, scores)
        ],
        "num_triples": len(result.triples),
    }


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------
def build_tools(
    *,
    base_retriever: BaseRetriever,
    graph_store: BaseGraphStore,
    context: ToolContext,
    reflector: BaseReflector | None = None,
    default_top_k: int = 5,
    multi_hop_hops: int = 2,
) -> list[StructuredTool]:
    """Construct the five retrieval tools bound to one retriever run.

    Parameters mirror the dependencies a retriever already holds; nothing
    here issues network calls beyond what the wrapped components do.
    """
    from chimera_rag.plugins.intent_agent.strategies.multi_hop import MultiHopStrategy
    from chimera_rag.plugins.intent_agent.strategies.parallel import ParallelStrategy

    def _empty_plan() -> ReasoningPlan:
        from chimera_rag.core.types import Intent

        return ReasoningPlan(
            intent=Intent(label="factual", confidence=0.0, rationale="langgraph-tool"),
            strategy_name="tool",
            steps=[],
            params={},
        )

    # -- vector_search ----------------------------------------------------
    def _vector_search(query: str, top_k: int = default_top_k) -> dict:
        result = base_retriever.retrieve(Query(text=query), top_k=top_k)
        context.add_result(result)
        return _summarise(result, TOOL_VECTOR_SEARCH)

    # -- graph_neighbors --------------------------------------------------
    def _graph_neighbors(entity: str, hops: int = 1) -> dict:
        triples = graph_store.get_neighbors(entity, hops=hops)
        context.add_result(RetrievalResult(chunks=[], triples=triples, scores=[]))
        return {
            "tool": TOOL_GRAPH_NEIGHBORS,
            "entity": entity,
            "hops": hops,
            "num_triples": len(triples),
            "triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in triples[:20]
            ],
        }

    # -- multi_hop_retrieve ----------------------------------------------
    def _multi_hop(query: str) -> dict:
        strategy = MultiHopStrategy(
            retriever=base_retriever, top_k=default_top_k, hops=multi_hop_hops
        )
        result = _run_async(strategy.execute(Query(text=query), _empty_plan()))
        context.add_result(result)
        return _summarise(result, TOOL_MULTI_HOP)

    # -- parallel_retrieve -----------------------------------------------
    def _parallel(query: str) -> dict:
        strategy = ParallelStrategy(retriever=base_retriever, top_k=default_top_k)
        result = _run_async(strategy.execute(Query(text=query), _empty_plan()))
        context.add_result(result)
        return _summarise(result, TOOL_PARALLEL)

    # -- assess_evidence --------------------------------------------------
    def _assess(query: str) -> dict:
        merged = context.merged_result()
        draft = Answer(
            text=" ".join(c.text for c in merged.chunks),
            evidence_chunk_ids=[c.chunk_id for c in merged.chunks],
        )
        reflector_impl = reflector
        if reflector_impl is None:
            from chimera_rag.plugins.intent_agent.hooks.reflection_hook import (
                HeuristicReflectionHook,
            )

            reflector_impl = HeuristicReflectionHook()
        feedback = _run_async(reflector_impl.evaluate(Query(text=query), draft))
        return {
            "tool": TOOL_ASSESS,
            "quality": round(float(feedback.quality), 4),
            "relevance": round(float(feedback.relevance), 4),
            "completeness": round(float(feedback.completeness), 4),
            "num_evidence_chunks": len(merged.chunks),
            "suggestion": feedback.suggestion,
        }

    tool_specs: list[tuple[str, str, object, type[BaseModel]]] = [
        (
            TOOL_VECTOR_SEARCH,
            "Semantic vector search over the corpus; returns the most similar chunks. "
            "Use for direct factual lookups or to seed evidence.",
            _vector_search,
            _QueryTopKArgs,
        ),
        (
            TOOL_GRAPH_NEIGHBORS,
            "Expand the knowledge graph around an entity by N hops; returns connecting "
            "triples. Use to bridge entities in multi-hop questions.",
            _graph_neighbors,
            _GraphNeighborsArgs,
        ),
        (
            TOOL_MULTI_HOP,
            "Seeded retrieval that widens graph neighbour hops to gather bridging "
            "evidence. Use for questions requiring chained reasoning across entities.",
            _multi_hop,
            _QueryArgs,
        ),
        (
            TOOL_PARALLEL,
            "Split a comparative query into sub-queries and retrieve them concurrently, "
            "then merge. Use for 'compare A and B' style questions.",
            _parallel,
            _QueryArgs,
        ),
        (
            TOOL_ASSESS,
            "Score the quality/relevance/completeness of the evidence gathered so far. "
            "Use to decide whether to stop or retrieve more.",
            _assess,
            _AssessArgs,
        ),
    ]

    return [
        StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=args_schema,
        )
        for name, description, func, args_schema in tool_specs
    ]


__all__ = [
    "ALL_TOOL_NAMES",
    "TOOL_ASSESS",
    "TOOL_GRAPH_NEIGHBORS",
    "TOOL_MULTI_HOP",
    "TOOL_PARALLEL",
    "TOOL_VECTOR_SEARCH",
    "ToolContext",
    "build_tools",
]

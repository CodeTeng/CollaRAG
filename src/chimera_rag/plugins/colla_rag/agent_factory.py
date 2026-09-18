"""AgentFactory (Factory pattern) + ToolSetBuilder (Builder pattern).

Contribution 4 — Graph-Conditioned Capability Specialization
-------------------------------------------------------------
The tool-permission matrix is no longer static. :class:`ToolSetBuilder.for_agent`
accepts an optional :class:`GraphFootprint` summarizing what the knowledge
graph can structurally support for the current query (entity hit rate, whether a
path connects the queried entities, KG density), and dynamically adjusts the
agent's tool subset:

  * a multi_hop query whose entities sit in a sparse region with no connecting
    path has its graph-traversal tools (``graph_path_search``,
    ``graph_subgraph_extract``, ``graph_neighbors``) **withdrawn** and web /
    dense-hybrid fallback **added** — forcing the agent to retrieve by text
    rather than waste iterations probing an empty graph;
  * a single_hop query whose entity is present in a dense region keeps
    ``graph_neighbors`` so it can pull attribute triples directly.

Each agent also carries a :data:`GRAPH_VIEWS` projection
(attribute / path / community / none) that formalizes which slice of the KG it
is meant to see — the static matrix already realizes this projection; the
dynamic rules collapse it when the graph cannot support the view.

When ``footprint`` is ``None`` (graph-poor, or no graph probe available), the
builder falls back to the static matrix, so the original behaviour is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from langchain_core.tools import StructuredTool

from chimera_rag.interfaces.base_agent import BaseAgent

Density = Literal["dense", "sparse", "unknown"]

# Per-agent KG projection (Contribution 4, mechanism 4B). The static tool matrix
# below already realizes these views; this mapping makes the projection explicit.
GRAPH_VIEWS: dict[str, str] = {
    "preprocessor": "none",
    "single_hop": "attribute",     # entity -> attribute lookups
    "multi_hop": "path",           # entity -> relation -> entity paths
    "summarization": "community",  # community-level subgraphs
    "other": "none",
}

TOOL_MATRIX: dict[str, list[str]] = {
    "preprocessor": [
        "query_rewrite", "coreference_resolve", "followup_merge",
        "read_session_context", "write_session_context",
        "read_shared_memory",
    ],
    "single_hop": [
        "vector_search", "bm25_search", "hybrid_search", "graph_neighbors",
        "entity_extract", "entity_link",
        "assess_evidence", "rerank",
        "read_shared_memory", "write_shared_memory",
        "read_agent_memory", "write_agent_memory",
        "save_to_file", "load_from_file",
        "answer_verify", "confidence_calibrate",
        "evidence_dedup",
    ],
    "multi_hop": [
        "vector_search", "bm25_search", "hybrid_search",
        "graph_neighbors", "graph_path_search", "graph_community_search",
        "graph_subgraph_extract", "graph_statistics",
        "entity_extract", "relation_extract", "entity_link",
        "sub_query_decompose", "assess_evidence", "rerank",
        "read_session_context", "write_session_context",
        "read_shared_memory", "write_shared_memory",
        "read_agent_memory", "write_agent_memory",
        "save_to_file", "load_from_file",
        "answer_verify", "claim_decompose", "source_attribution",
        "temporal_filter", "evidence_dedup", "chunk_summarize",
        "confidence_calibrate",
    ],
    "summarization": [
        "vector_search", "bm25_search", "hybrid_search",
        "graph_community_search", "rerank",
        "entity_extract",
        "read_shared_memory", "write_shared_memory",
        "read_agent_memory", "write_agent_memory",
        "save_to_file", "load_from_file",
        "chunk_summarize", "evidence_dedup",
        "source_attribution", "confidence_calibrate",
    ],
    "other": [
        "hybrid_search", "web_search", "web_fetch",
        "entity_extract",
        "read_shared_memory",
        "read_agent_memory", "write_agent_memory",
        "save_to_file", "load_from_file",
        "confidence_calibrate",
    ],
}

# Graph-traversal tools that only make sense when the KG has structure to traverse.
_GRAPH_TRAVERSAL_TOOLS = {
    "graph_path_search", "graph_subgraph_extract", "graph_neighbors",
    "graph_community_search", "graph_statistics",
}

# Fallback tools added when graph traversal is withdrawn.
_FALLBACK_TOOLS = {"web_search", "web_fetch"}


@dataclass
class GraphFootprint:
    """A cheap summary of what the KG can structurally support for a query.

    Computed before dispatch from a quick entity probe of the graph_store; used
    by :class:`ToolSetBuilder` to condition the agent's tool subset (C4).
    """

    entity_hit_rate: float = 0.0   # fraction of probed entities present in the KG
    has_path: bool = False         # is there a KG path between the queried entities?
    density: Density = "unknown"   # coarse density of the relevant KG region
    n_entities: int = 0            # number of entities probed

    @property
    def graph_poor(self) -> bool:
        """True when the graph offers no structural support for this query."""
        return self.entity_hit_rate == 0.0 or (self.density == "sparse" and not self.has_path)


class ToolSetBuilder:
    @staticmethod
    def for_agent(
        agent_type: str,
        all_tools: list[StructuredTool],
        footprint: GraphFootprint | None = None,
    ) -> list[StructuredTool]:
        allowed = set(TOOL_MATRIX.get(agent_type, []))
        tools = [t for t in all_tools if t.name in allowed]
        if footprint is not None:
            tools = ToolSetBuilder._apply_dynamic_rules(agent_type, tools, all_tools, footprint)
        return tools

    @staticmethod
    def _apply_dynamic_rules(
        agent_type: str,
        tools: list[StructuredTool],
        all_tools: list[StructuredTool],
        fp: GraphFootprint,
    ) -> list[StructuredTool]:
        """Dynamic, graph-conditioned tool permission (mechanism 4A)."""
        if agent_type == "multi_hop":
            # Multi-hop in a graph-poor region: withdraw traversal tools and
            # add web fallback so the agent retrieves by text instead of
            # burning iterations probing an empty graph (mechanism 4A).
            if fp.graph_poor:
                kept = [t for t in tools if t.name not in _GRAPH_TRAVERSAL_TOOLS]
                kept_names = {t.name for t in kept}
                # web_search / web_fetch are built into all_tools (build_web_tools)
                # but normally excluded from the multi_hop matrix; add them here.
                web_fallback = [
                    t for t in all_tools if t.name in _FALLBACK_TOOLS and t.name not in kept_names
                ]
                tools = kept + web_fallback
            return tools

        if agent_type == "single_hop":
            # Single-hop whose entity is absent from the KG: drop graph_neighbors
            # (it would return nothing) and rely on dense/BM25 retrieval.
            if fp.entity_hit_rate == 0.0 and fp.n_entities > 0:
                tools = [t for t in tools if t.name != "graph_neighbors"]
            return tools

        # Summarization / other / preprocessor: no graph-conditioned changes.
        return tools

    @staticmethod
    def graph_view(agent_type: str) -> str:
        """The per-agent KG projection this specialist is meant to see (4B)."""
        return GRAPH_VIEWS.get(agent_type, "none")


class AgentFactory:
    _registry: ClassVar[dict[str, type[BaseAgent]]] = {}

    @classmethod
    def register(cls, intent: str):
        def decorator(agent_cls: type[BaseAgent]) -> type[BaseAgent]:
            cls._registry[intent] = agent_cls
            return agent_cls
        return decorator

    @classmethod
    def create(
        cls,
        intent: str,
        *,
        all_tools: list[StructuredTool],
        memory_manager: any,
        config: dict,
        footprint: GraphFootprint | None = None,
    ) -> BaseAgent:
        agent_cls = cls._registry.get(intent)
        if agent_cls is None:
            raise ValueError(f"No agent registered for intent {intent!r}")
        tools = ToolSetBuilder.for_agent(intent, all_tools, footprint=footprint)
        return agent_cls(agent_type=intent, tools=tools, config=config)


__all__ = [
    "GRAPH_VIEWS",
    "TOOL_MATRIX",
    "AgentFactory",
    "GraphFootprint",
    "ToolSetBuilder",
]

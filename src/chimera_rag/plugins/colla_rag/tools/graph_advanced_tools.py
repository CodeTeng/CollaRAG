"""Advanced graph tools: subgraph extraction, graph statistics."""
from __future__ import annotations

import json
import logging
from collections import Counter

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.defaults.extractor import _run_async
from chimera_rag.plugins.colla_rag.prompts import GRAPH_SUBGRAPH_PROMPT

logger = logging.getLogger(__name__)


class _SubgraphExtractArgs(BaseModel):
    entities: list[str] = Field(description="Seed entity names to extract subgraph around.")
    hops: int = Field(default=2, ge=1, le=4, description="Number of hops from seed entities.")
    max_triples: int = Field(default=50, ge=1, le=200, description="Max triples to return.")


class _GraphStatisticsArgs(BaseModel):
    pass


def build_graph_advanced_tools(*, llm, graph_store, accumulator) -> list[StructuredTool]:

    def _graph_subgraph_extract(
        entities: list[str], hops: int = 2, max_triples: int = 50,
    ) -> dict:
        """Extract a focused subgraph around seed entities and describe it."""
        all_triples = []
        seen_keys = set()
        for entity in entities:
            neighbors = graph_store.get_neighbors(entity, hops=hops)
            for t in neighbors:
                key = (t.subject, t.predicate, t.object)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_triples.append(t)

        all_triples = all_triples[:max_triples]
        accumulator.add_triples(all_triples)

        nodes = set()
        for t in all_triples:
            nodes.add(t.subject)
            nodes.add(t.object)

        triple_strs = [
            f"({t.subject}, {t.predicate}, {t.object})" for t in all_triples
        ]

        if all_triples and llm is not None:
            prompt = GRAPH_SUBGRAPH_PROMPT.format(
                seed_entities=json.dumps(entities, ensure_ascii=False),
                triples="\n".join(triple_strs[:30]),
            )
            try:
                raw = _run_async(llm.complete(prompt, max_tokens=512))
                description = raw.strip()
            except Exception as e:
                logger.warning("subgraph description LLM call failed: %s", e)
                description = f"Subgraph with {len(nodes)} entities and {len(all_triples)} relationships"
        else:
            description = f"Subgraph with {len(nodes)} entities and {len(all_triples)} relationships"

        node_degree = Counter()
        for t in all_triples:
            node_degree[t.subject] += 1
            node_degree[t.object] += 1
        hubs = [n for n, _ in node_degree.most_common(5)]

        return {
            "tool": "graph_subgraph_extract",
            "seed_entities": entities,
            "num_nodes": len(nodes),
            "num_triples": len(all_triples),
            "hubs": hubs,
            "triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in all_triples[:20]
            ],
            "description": description[:500],
        }

    def _graph_statistics() -> dict:
        """Return knowledge graph statistics: node count, edge count, top hubs, isolated nodes."""
        all_triples = graph_store.all_triples()

        nodes = set()
        node_degree = Counter()
        predicates = Counter()
        for t in all_triples:
            nodes.add(t.subject)
            nodes.add(t.object)
            node_degree[t.subject] += 1
            node_degree[t.object] += 1
            predicates[t.predicate] += 1

        subjects = {t.subject for t in all_triples}
        objects = {t.object for t in all_triples}
        leaf_nodes = (objects - subjects)
        isolated = {n for n in nodes if node_degree[n] == 1} & leaf_nodes

        return {
            "tool": "graph_statistics",
            "num_nodes": len(nodes),
            "num_edges": len(all_triples),
            "num_predicates": len(predicates),
            "top_hubs": [
                {"entity": n, "degree": d} for n, d in node_degree.most_common(10)
            ],
            "top_predicates": [
                {"predicate": p, "count": c} for p, c in predicates.most_common(10)
            ],
            "num_isolated_leaves": len(isolated),
            "density": round(
                len(all_triples) / max(1, len(nodes) * (len(nodes) - 1)), 6
            ),
        }

    return [
        StructuredTool.from_function(
            _graph_subgraph_extract, name="graph_subgraph_extract",
            description="Extract a focused subgraph around seed entities with N-hop expansion. Returns structured description with hubs and triples.",
            args_schema=_SubgraphExtractArgs,
        ),
        StructuredTool.from_function(
            _graph_statistics, name="graph_statistics",
            description="Return knowledge graph statistics: node/edge counts, top hubs, predicate distribution, density. Use to assess graph coverage.",
            args_schema=_GraphStatisticsArgs,
        ),
    ]


__all__ = ["build_graph_advanced_tools"]

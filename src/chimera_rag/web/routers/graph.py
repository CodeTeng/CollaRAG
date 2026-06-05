"""GET /api/graph + /api/graph/stats — surface the knowledge graph for the UI."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from chimera_rag import ChimeraRAG
from chimera_rag.web.deps import get_chimera
from chimera_rag.web.schemas import GraphEdge, GraphNode, GraphResponse

router = APIRouter(tags=["graph"])


@router.get("/graph", response_model=GraphResponse)
def graph(
    chimera: ChimeraRAG = Depends(get_chimera),
    entity: str | None = Query(None, description="restrict to a subgraph around this entity"),
    hops: int = Query(1, ge=1, le=3),
    limit_edges: int = Query(500, ge=1, le=5000),
) -> GraphResponse:
    gs = chimera.graph_store

    triples = gs.get_neighbors(entity, hops=hops) if entity else gs.all_triples()

    # Drop triples whose subject or object is blank: the LLM extractor
    # occasionally emits incomplete relations (e.g. a predicate with no
    # object), and a graph node/edge with an empty id breaks downstream
    # renderers like cytoscape ("invalid string ID").
    triples = [t for t in triples if t.subject.strip() and t.object.strip()]

    if len(triples) > limit_edges:
        triples = triples[:limit_edges]

    node_ids: set[str] = set()
    edges: list[GraphEdge] = []
    for t in triples:
        node_ids.add(t.subject)
        node_ids.add(t.object)
        edges.append(
            GraphEdge(source=t.subject, target=t.object, label=t.predicate, layer=t.layer)
        )

    nodes = [GraphNode(id=i, label=i) for i in sorted(node_ids)]

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        stats={
            "nodes": len(nodes),
            "edges": len(edges),
            "filtered": entity is not None,
            "entity": entity,
        },
    )


@router.get("/graph/stats")
def graph_stats(chimera: ChimeraRAG = Depends(get_chimera)) -> dict:
    all_triples = chimera.graph_store.all_triples()
    node_ids: set[str] = set()
    layer_counts: dict[str, int] = {}
    for t in all_triples:
        node_ids.add(t.subject)
        node_ids.add(t.object)
        layer_counts[t.layer] = layer_counts.get(t.layer, 0) + 1

    return {
        "nodes": len(node_ids),
        "edges": len(all_triples),
        "chunks": chimera.stats()["chunks"],
        "triples": chimera.stats()["triples"],
        "by_layer": layer_counts,
    }


__all__ = ["router"]

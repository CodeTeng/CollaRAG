"""POST /api/query — single-shot question answering."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from chimera_rag import ChimeraRAG
from chimera_rag.core.types import Query
from chimera_rag.web.deps import get_chimera
from chimera_rag.web.schemas import AgentTracePayload, ChunkPayload, QueryRequest, QueryResponse, TriplePayload

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    chimera: ChimeraRAG = Depends(get_chimera),
) -> QueryResponse:
    # Resolve top_k (body override > config default).
    top_k = body.top_k or int(chimera.config.query.retriever.params.get("top_k", 5))

    # Build Query, run pipeline (uses configured intent_classifier + retriever).
    q = Query(text=body.text, session_id=body.session_id)
    answer = chimera.query(q)

    # Also expose the raw retrieval for trace visualisation in the UI.
    retrieval = chimera._query.retriever.retrieve(q, top_k=top_k)

    chunks = [
        ChunkPayload(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            index=c.index,
            text=c.text,
            metadata=c.metadata,
        )
        for c in retrieval.chunks[:top_k]
    ]
    triples = [
        TriplePayload(
            subject=t.subject,
            predicate=t.predicate,
            object=t.object,
            confidence=t.confidence,
            layer=t.layer,
            source_chunk_id=t.source_chunk_id,
        )
        for t in retrieval.triples
    ]

    trace_data = {**answer.trace, "retrieval_metadata": retrieval.metadata}

    agent_trace = AgentTracePayload(
        agent_type=trace_data.get("agent_type") or retrieval.metadata.get("agent"),
        rewritten_query=trace_data.get("rewritten_query"),
        original_query=trace_data.get("original_query"),
        plan=trace_data.get("plan", []),
        step_results=trace_data.get("step_results", []),
        tool_call_log=trace_data.get("tool_call_log", []),
        reflect_count=trace_data.get("reflect_count", 0),
        quality=trace_data.get("quality"),
        map_count=trace_data.get("map_count"),
        web_results_count=trace_data.get("web_results_count"),
        iterations=trace_data.get("iterations"),
    )

    return QueryResponse(
        answer=answer.text,
        intent=answer.intent_label,
        strategy=retrieval.metadata.get("strategy"),
        confidence=answer.confidence,
        evidence_chunk_ids=answer.evidence_chunk_ids,
        chunks=chunks,
        triples=triples,
        trace=trace_data,
        agent_trace=agent_trace,
    )


__all__ = ["router"]

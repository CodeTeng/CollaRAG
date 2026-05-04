"""POST /api/ingest — build the knowledge graph from a dataset or inline docs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from chimera_rag import ChimeraRAG
from chimera_rag.core.exceptions import DatasetError
from chimera_rag.core.types import Document
from chimera_rag.datasets import get_dataset_loader
from chimera_rag.web.deps import get_chimera
from chimera_rag.web.schemas import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    body: IngestRequest,
    chimera: ChimeraRAG = Depends(get_chimera),
) -> IngestResponse:
    if not body.dataset and not body.documents:
        raise HTTPException(
            status_code=400,
            detail="either 'dataset' (name from config) or 'documents' (inline) is required",
        )

    docs: list[Document] = []

    if body.dataset:
        ds_cfg = chimera.config.datasets.get(body.dataset)
        if ds_cfg is None:
            raise HTTPException(
                status_code=404,
                detail=f"dataset {body.dataset!r} not declared in config.datasets",
            )
        try:
            loader_cls = get_dataset_loader(ds_cfg.loader)
        except DatasetError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        kwargs: dict = {"path": ds_cfg.path}
        if ds_cfg.eval_field_mapping:
            kwargs["eval_field_mapping"] = ds_cfg.eval_field_mapping
        try:
            docs = loader_cls(**kwargs).load_documents()
        except DatasetError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    if body.documents:
        docs.extend(
            Document(
                doc_id=str(item.get("doc_id") or f"inline-{i}"),
                content=str(item.get("content", "")),
                metadata=item.get("metadata", {}) or {},
            )
            for i, item in enumerate(body.documents)
        )

    if not docs:
        raise HTTPException(status_code=400, detail="no documents to ingest")

    stats = chimera.ingest(docs)
    return IngestResponse(stats=stats, state=chimera.stats())


__all__ = ["router"]

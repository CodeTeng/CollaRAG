"""Health-check and status endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from chimera_rag import ChimeraRAG, __version__
from chimera_rag.web.deps import get_chimera
from chimera_rag.web.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(chimera: ChimeraRAG = Depends(get_chimera)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        active_plugins={
            "adagraph": chimera.config.plugins.adagraph.enabled,
            "colla_rag": chimera.config.plugins.colla_rag.enabled,
        },
    )


__all__ = ["router"]

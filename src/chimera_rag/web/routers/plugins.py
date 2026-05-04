"""GET /api/plugins + POST /api/plugins/toggle (hot swap)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from chimera_rag import ChimeraRAG
from chimera_rag.web.deps import get_chimera
from chimera_rag.web.schemas import (
    PluginsResponse,
    PluginStatus,
    PluginToggleRequest,
)

router = APIRouter(tags=["plugins"])


@router.get("/plugins", response_model=PluginsResponse)
def list_plugins(chimera: ChimeraRAG = Depends(get_chimera)) -> PluginsResponse:
    cfg = chimera.config
    active = chimera.active_implementations()

    adagraph_slots = {
        "chunker": cfg.ingestion.chunker.active,
        "extractor": cfg.ingestion.extractor.active,
        "pruner": cfg.ingestion.pruner.active,
    }
    intent_agent_slots = {
        "intent_classifier": cfg.query.intent_classifier.active,
        "retriever": cfg.query.retriever.active,
    }

    plugins = [
        PluginStatus(
            name="adagraph",
            enabled=cfg.plugins.adagraph.enabled,
            active_slots=adagraph_slots,
        ),
        PluginStatus(
            name="intent_agent",
            enabled=cfg.plugins.intent_agent.enabled,
            active_slots=intent_agent_slots,
        ),
    ]
    return PluginsResponse(plugins=plugins, active_implementations=active)


@router.post("/plugins/toggle")
def toggle(
    body: PluginToggleRequest,
    chimera: ChimeraRAG = Depends(get_chimera),
) -> dict:
    if body.name not in {"adagraph", "intent_agent"}:
        raise HTTPException(
            status_code=404,
            detail=f"unknown plugin: {body.name!r}; supported: adagraph, intent_agent",
        )
    chimera.toggle_plugin(body.name, body.enabled)
    return {
        "name": body.name,
        "enabled": body.enabled,
        "active_implementations": chimera.active_implementations(),
    }


__all__ = ["router"]

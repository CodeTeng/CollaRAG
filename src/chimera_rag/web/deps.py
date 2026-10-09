"""FastAPI dependency helpers: expose the shared ChimeraRAG instance."""

from __future__ import annotations

from fastapi import Request

from chimera_rag import ChimeraRAG
from chimera_rag.web.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_chimera(request: Request) -> ChimeraRAG:
    return get_state(request).chimera


__all__ = ["get_chimera", "get_state"]

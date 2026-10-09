"""Shared runtime state holder for the FastAPI app.

The app factory mutates this object so handlers can swap the active
ChimeraRAG instance after a PUT /api/config (hot reload).
"""

from __future__ import annotations

from dataclasses import dataclass

from chimera_rag import ChimeraRAG


@dataclass
class AppState:
    chimera: ChimeraRAG


__all__ = ["AppState"]

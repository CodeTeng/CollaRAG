"""FastAPI app factory.

Usage::

    from chimera_rag import ChimeraRAG
    from chimera_rag.web.app import create_app

    crag = ChimeraRAG.from_config("configs/mock_full.yaml")
    app = create_app(crag)
    # uvicorn.run(app, host=..., port=...)

The factory:
1. stores the ChimeraRAG instance on ``app.state.app_state`` so handlers
   can retrieve it via :func:`chimera_rag.web.deps.get_chimera`,
2. registers all routers under the ``/api`` prefix,
3. mounts the built frontend under ``/`` (if it exists) so a single
   ``serve`` process hosts both API + static UI,
4. installs permissive CORS in dev mode for the Vite 5173 origin.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from chimera_rag import ChimeraRAG, __version__
from chimera_rag.web.state import AppState

logger = logging.getLogger(__name__)


def create_app(chimera: ChimeraRAG) -> FastAPI:
    app = FastAPI(
        title="Chimera-RAG API",
        version=__version__,
        description="Knowledge-graph based RAG with AdaGraph + CollaRAG plugins.",
    )
    app.state.app_state = AppState(chimera=chimera)

    # ------------------------------------------------------------------
    # CORS — dev mode only allows the Vite dev server origin.
    # ------------------------------------------------------------------
    cors_origins = chimera.config.web.cors_origins or []
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------
    # Routers — added progressively as each TDD cycle lands.
    # ------------------------------------------------------------------
    from chimera_rag.web.routers import health as health_router

    app.include_router(health_router.router, prefix="/api")

    # The following routers are wired in by _maybe_include_router as they
    # become available; keeps the module import-safe during incremental
    # development.
    _maybe_include_router(app, "ingest")
    _maybe_include_router(app, "query")
    _maybe_include_router(app, "graph")
    _maybe_include_router(app, "plugins")
    _maybe_include_router(app, "config")
    _maybe_include_router(app, "evaluate")
    _maybe_include_router(app, "agents")

    # ------------------------------------------------------------------
    # Static (frontend build)
    # ------------------------------------------------------------------
    static_dir = Path(chimera.config.web.static_dir)
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
        logger.info("mounted frontend from %s", static_dir)
    else:
        logger.debug("frontend static dir %s not present; API-only mode", static_dir)

    return app


def _maybe_include_router(app: FastAPI, name: str) -> None:
    """Include a router if its module exists and exposes a ``router`` attr."""
    import importlib

    try:
        mod = importlib.import_module(f"chimera_rag.web.routers.{name}")
    except ModuleNotFoundError:
        logger.debug("router %r not available yet (pending TDD cycle)", name)
        return
    router = getattr(mod, "router", None)
    if router is None:
        return
    app.include_router(router, prefix="/api")


__all__ = ["create_app"]

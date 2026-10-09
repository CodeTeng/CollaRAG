"""FastAPI routers — one module per domain.

Each router registers its own path prefix (e.g. ``/ingest``) on top of
the ``/api`` prefix applied by :func:`chimera_rag.web.app.create_app`.
"""

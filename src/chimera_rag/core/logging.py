"""Lightweight logging setup for Chimera-RAG.

We avoid configuring root-level logging at import time (which would
interfere with application hosts); instead :func:`setup_logging` is an
explicit call made by ``main.py`` and the FastAPI app factory.
"""

from __future__ import annotations

import logging
from typing import Literal

_SENTINEL_ATTR = "_chimera_rag_configured"


def setup_logging(
    level: str = "INFO",
    format: Literal["rich", "plain", "json"] = "rich",
    file: str | None = None,
) -> None:
    """Install a handler on the root logger at ``level``.

    Idempotent: calling multiple times does *not* add duplicate handlers.
    ``format='rich'`` uses :mod:`rich.logging` when available, falling back
    to plain text otherwise. ``format='json'`` is reserved for future use
    (currently treated as plain).
    """
    root = logging.getLogger()

    # Idempotency: marker attribute on any existing handler means we've
    # already been here in this process.
    if any(getattr(h, _SENTINEL_ATTR, False) for h in root.handlers):
        root.setLevel(getattr(logging, level))
        return

    handler: logging.Handler
    if format == "rich":
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, markup=False, show_path=False)
        except ImportError:  # pragma: no cover - rich is a hard dep
            handler = logging.StreamHandler()
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    setattr(handler, _SENTINEL_ATTR, True)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level))

    if file:
        fh = logging.FileHandler(file)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        )
        setattr(fh, _SENTINEL_ATTR, True)
        root.addHandler(fh)


__all__ = ["setup_logging"]

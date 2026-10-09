"""Tests for :mod:`chimera_rag.core.logging`."""

from __future__ import annotations

import logging


def test_setup_logging_installs_a_handler_at_requested_level():
    from chimera_rag.core.logging import setup_logging

    setup_logging(level="DEBUG", format="plain")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert any(isinstance(h, logging.Handler) for h in root.handlers)


def test_setup_logging_is_idempotent_no_duplicate_handlers():
    from chimera_rag.core.logging import setup_logging

    setup_logging(level="INFO", format="plain")
    n1 = len(logging.getLogger().handlers)
    setup_logging(level="INFO", format="plain")
    n2 = len(logging.getLogger().handlers)
    assert n1 == n2


def test_setup_logging_accepts_rich_format():
    from chimera_rag.core.logging import setup_logging

    # Should not raise even when rich is installed.
    setup_logging(level="INFO", format="rich")

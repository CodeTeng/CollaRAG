"""Plugin registry — the backbone of Chimera-RAG's pluggable architecture.

Any class that implements an ABC in :mod:`chimera_rag.interfaces` can be
attached to a *slot* (e.g. ``chunker``) with a unique *name*
(e.g. ``adagraph.dynamic``). The :class:`Pipeline` resolves the active
implementation for each slot by reading ``config.<slot>.active`` and
calling :meth:`Registry.get`.

Typical usage::

    from chimera_rag.core.registry import register

    @register("chunker", "defaults.fixed")
    class FixedSizeChunker(BaseChunker):
        ...

The module also exposes a *global* registry via :func:`get_registry`
which is what plugin packages implicitly extend at import time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from chimera_rag.core.exceptions import PluginNotFoundError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type)


class Registry:
    """In-memory, two-level map ``slot -> name -> class``.

    Instances are cheap; tests create a fresh one per test, production code
    uses the module-level singleton via :func:`get_registry`.
    """

    def __init__(self) -> None:
        self._map: dict[str, dict[str, type]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, slot: str, name: str) -> Callable[[T], T]:
        """Return a class decorator that attaches ``cls`` to ``(slot, name)``.

        Raises ``ValueError`` if the pair is already registered, which makes
        accidental double-registration visible instead of silently shadowed.
        """

        def decorator(cls: T) -> T:
            bucket = self._map.setdefault(slot, {})
            if name in bucket:
                raise ValueError(
                    f"{name!r} is already registered in slot {slot!r} "
                    f"(existing={bucket[name].__name__}, new={cls.__name__})"
                )
            bucket[name] = cls
            logger.debug("registered %s::%s -> %s", slot, name, cls.__name__)
            return cls

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(self, slot: str, name: str) -> type:
        bucket = self._map.get(slot, {})
        if name not in bucket:
            raise PluginNotFoundError(slot=slot, name=name)
        return bucket[name]

    def get_or_default(self, slot: str, name: str, default: str) -> type:
        """Return the class for ``name`` if present, else fall back to ``default``.

        If both are missing we raise :class:`PluginNotFoundError` for
        ``default`` (the most actionable hint at that point).
        """
        bucket = self._map.get(slot, {})
        if name in bucket:
            return bucket[name]
        logger.warning(
            "plugin %r for slot %r not registered, falling back to default %r",
            name,
            slot,
            default,
        )
        if default not in bucket:
            raise PluginNotFoundError(slot=slot, name=default)
        return bucket[default]

    def list_slot(self, slot: str) -> list[str]:
        """Return registered names in ``slot``, sorted for stable output."""
        return sorted(self._map.get(slot, {}).keys())


# ---------------------------------------------------------------------------
# Global singleton + shortcut decorator
# ---------------------------------------------------------------------------
_GLOBAL: Registry | None = None


def get_registry() -> Registry:
    """Return the process-wide Registry singleton."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = Registry()
    return _GLOBAL


def register(slot: str, name: str) -> Callable[[T], T]:
    """Shortcut decorator delegating to the global registry."""
    return get_registry().register(slot, name)


__all__ = ["Registry", "get_registry", "register"]

"""Tests for :mod:`chimera_rag.core.registry`.

The Registry is the backbone of Chimera-RAG's pluggable architecture.
Anything that lives behind an interface (chunker, extractor, retriever, ...)
gets registered here and looked up by ``(slot, name)``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def fresh_registry():
    """Yield a fresh Registry instance for full isolation between tests."""
    from chimera_rag.core.registry import Registry

    yield Registry()


# ---------------------------------------------------------------------------
# Register / get
# ---------------------------------------------------------------------------
def test_register_then_get_returns_the_class(fresh_registry):
    @fresh_registry.register("chunker", "demo.simple")
    class DemoChunker:
        pass

    assert fresh_registry.get("chunker", "demo.simple") is DemoChunker


def test_register_decorator_returns_the_original_class(fresh_registry):
    @fresh_registry.register("chunker", "demo.simple")
    class DemoChunker:
        label = "demo"

    # The decorator must not wrap/replace the class.
    assert DemoChunker.label == "demo"


def test_register_rejects_duplicate_slot_name_pair(fresh_registry):
    @fresh_registry.register("chunker", "demo.simple")
    class A:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @fresh_registry.register("chunker", "demo.simple")
        class B:
            pass


def test_get_raises_plugin_not_found_with_slot_and_name(fresh_registry):
    from chimera_rag.core.exceptions import PluginNotFoundError

    with pytest.raises(PluginNotFoundError) as excinfo:
        fresh_registry.get("chunker", "does.not.exist")
    assert excinfo.value.slot == "chunker"
    assert excinfo.value.name == "does.not.exist"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------
def test_list_slot_returns_all_registered_names_sorted(fresh_registry):
    for n in ["demo.b", "demo.a", "demo.c"]:

        @fresh_registry.register("chunker", n)
        class _C:
            pass

    assert fresh_registry.list_slot("chunker") == ["demo.a", "demo.b", "demo.c"]


def test_list_slot_for_unknown_slot_returns_empty(fresh_registry):
    assert fresh_registry.list_slot("never_registered") == []


# ---------------------------------------------------------------------------
# Namespacing between slots
# ---------------------------------------------------------------------------
def test_same_name_in_different_slots_does_not_collide(fresh_registry):
    @fresh_registry.register("chunker", "x")
    class Ch:
        pass

    @fresh_registry.register("extractor", "x")
    class Ex:
        pass

    assert fresh_registry.get("chunker", "x") is Ch
    assert fresh_registry.get("extractor", "x") is Ex


# ---------------------------------------------------------------------------
# get_or_default: fallback convenience
# ---------------------------------------------------------------------------
def test_get_or_default_returns_requested_when_registered(fresh_registry):
    @fresh_registry.register("chunker", "custom")
    class C:
        pass

    @fresh_registry.register("chunker", "defaults.fixed")
    class D:
        pass

    assert (
        fresh_registry.get_or_default("chunker", "custom", "defaults.fixed") is C
    )


def test_get_or_default_falls_back_when_requested_missing(fresh_registry, caplog):
    @fresh_registry.register("chunker", "defaults.fixed")
    class D:
        pass

    import logging

    with caplog.at_level(logging.WARNING):
        assert (
            fresh_registry.get_or_default("chunker", "missing", "defaults.fixed") is D
        )
    # Should emit a warning noting the fallback.
    assert any("missing" in rec.getMessage() for rec in caplog.records)


def test_get_or_default_raises_if_both_missing(fresh_registry):
    from chimera_rag.core.exceptions import PluginNotFoundError

    with pytest.raises(PluginNotFoundError):
        fresh_registry.get_or_default("chunker", "a", "b")


# ---------------------------------------------------------------------------
# Global singleton accessor
# ---------------------------------------------------------------------------
def test_global_registry_is_a_singleton():
    from chimera_rag.core.registry import get_registry

    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


def test_module_exports_register_as_shortcut_decorator():
    """`@register('slot','name')` should delegate to the global registry."""
    from chimera_rag.core.registry import get_registry, register

    @register("chunker", "global.demo")
    class Demo:
        pass

    assert get_registry().get("chunker", "global.demo") is Demo

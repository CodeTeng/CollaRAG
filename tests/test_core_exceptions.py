"""Tests for :mod:`chimera_rag.core.exceptions`.

We enforce a small, documented error hierarchy so callers can catch
``ChimeraError`` broadly or specific subclasses narrowly.
"""

from __future__ import annotations

import pytest


def test_chimera_error_is_the_base_class():
    from chimera_rag.core.exceptions import ChimeraError

    with pytest.raises(ChimeraError):
        raise ChimeraError("oops")


@pytest.mark.parametrize(
    "subclass_name",
    [
        "ConfigError",
        "ProviderError",
        "DatasetError",
        "PipelineError",
    ],
)
def test_subclass_inherits_from_chimera_error(subclass_name: str):
    import chimera_rag.core.exceptions as exc_mod
    from chimera_rag.core.exceptions import ChimeraError

    cls = getattr(exc_mod, subclass_name)
    assert issubclass(cls, ChimeraError)
    # Also: raising the subclass is catchable as ChimeraError.
    with pytest.raises(ChimeraError):
        raise cls("boom")


def test_plugin_not_found_error_inherits_and_needs_slot_name():
    # PluginNotFoundError is part of the hierarchy but has a richer ctor.
    from chimera_rag.core.exceptions import ChimeraError, PluginNotFoundError

    assert issubclass(PluginNotFoundError, ChimeraError)
    with pytest.raises(ChimeraError):
        raise PluginNotFoundError(slot="chunker", name="x")


def test_plugin_not_found_error_carries_slot_and_name():
    from chimera_rag.core.exceptions import PluginNotFoundError

    err = PluginNotFoundError(slot="chunker", name="adagraph.dynamic")
    assert err.slot == "chunker"
    assert err.name == "adagraph.dynamic"
    # The message should mention both so debugging is easy.
    assert "chunker" in str(err)
    assert "adagraph.dynamic" in str(err)

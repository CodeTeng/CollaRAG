"""Regression tests around ChimeraRAG._filter_params and plugin toggle."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_filter_params_rejects_kwargs_for_classes_without_own_init():
    """Classes that don't declare __init__ (NoOpPruner) inherit object's
    (self, *args, **kwargs) — *filtered* to empty, not passed verbatim."""
    from chimera_rag import ChimeraRAG
    from chimera_rag.defaults.pruner import NoOpPruner

    assert ChimeraRAG._filter_params(NoOpPruner, {"foo": 1, "bar": 2}) == {}


def test_plugin_toggle_from_full_to_vanilla_does_not_crash(
    mock_full_yaml: str, tmp_path: Path
):
    """Regression: toggling adagraph off while its params are still in
    config.ingestion.pruner.params must not crash on NoOpPruner instantiation.
    """
    from chimera_rag import ChimeraRAG

    crag = ChimeraRAG.from_config(mock_full_yaml)
    assert crag.active_implementations()["pruner"] == "adagraph.dual_redundancy"

    # Flip adagraph off — config.ingestion.pruner.active becomes 'defaults.noop'
    # but pruner.params still holds {similarity_threshold: 0.85, ...}.
    crag.toggle_plugin("adagraph", False)
    assert crag.active_implementations()["pruner"] == "defaults.noop"

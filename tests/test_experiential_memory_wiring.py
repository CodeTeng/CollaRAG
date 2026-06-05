"""Regression tests for the experiential-memory wiring in ChimeraRAG.

Innovation #2c lives behind ``ChimeraRAG._instantiate_retriever``: the yaml
carries ``query.retriever.params.experiential_memory`` as a *config dict*
(enabled/path/recall_top_n/max_per_bucket), and the facade must translate it
into a concrete ``FileExperientialMemory`` instance before constructing the
agentic retriever. Without this, a langgraph retriever silently degrades to
pure 2b even when the yaml asks for memory.

These tests target the pure ``_wire_experiential_memory`` helper directly so
they stay fast and free of faiss / langgraph imports.
"""

from __future__ import annotations

from chimera_rag import ChimeraRAG
from chimera_rag.interfaces.experiential_memory import BaseExperientialMemory


def test_enabled_dict_is_translated_into_instance(tmp_path):
    call_params = {
        "experiential_memory": {
            "enabled": True,
            "path": str(tmp_path / "mem"),
            "recall_top_n": 4,
            "max_per_bucket": 50,
        }
    }
    wiring: dict = {}

    ChimeraRAG._wire_experiential_memory(call_params, wiring)

    assert "experiential_memory" not in call_params
    assert isinstance(wiring["experiential_memory"], BaseExperientialMemory)
    assert wiring["experiential_recall_top_n"] == 4


def test_disabled_dict_injects_nothing(tmp_path):
    call_params = {
        "experiential_memory": {
            "enabled": False,
            "path": str(tmp_path / "mem"),
        }
    }
    wiring: dict = {}

    ChimeraRAG._wire_experiential_memory(call_params, wiring)

    assert "experiential_memory" not in wiring
    assert "experiential_recall_top_n" not in wiring


def test_missing_config_is_a_noop():
    call_params: dict = {}
    wiring: dict = {}

    ChimeraRAG._wire_experiential_memory(call_params, wiring)

    assert wiring == {}


def test_recall_top_n_defaults_when_absent(tmp_path):
    call_params = {
        "experiential_memory": {
            "enabled": True,
            "path": str(tmp_path / "mem"),
        }
    }
    wiring: dict = {}

    ChimeraRAG._wire_experiential_memory(call_params, wiring)

    assert isinstance(wiring["experiential_memory"], BaseExperientialMemory)
    # recall_top_n omitted -> let the retriever apply its own default
    assert "experiential_recall_top_n" not in wiring

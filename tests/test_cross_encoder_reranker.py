"""Tests for the CrossEncoder reranker hook and its wiring in ChimeraRAG facade.

Unit tests: registration, interface conformance, reranking logic with mock model.
Integration tests: facade _wire_reranker reading config and producing the correct
reranker instance. Uses the mock_intent_only_yaml fixture from conftest.py.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Registration & interface conformance
# ---------------------------------------------------------------------------
def test_cross_encoder_registers_under_slot():
    pytest.importorskip("sentence_transformers")
    import chimera_rag.plugins.intent_agent.hooks.cross_encoder_reranker as mod
    from chimera_rag.core.registry import get_registry

    assert get_registry().get("reranker", "intent_agent.cross_encoder") is mod.CrossEncoderReranker


async def test_cross_encoder_implements_base_reranker():
    pytest.importorskip("sentence_transformers")
    from chimera_rag.interfaces.reranker import BaseReranker
    from chimera_rag.plugins.intent_agent.hooks.cross_encoder_reranker import CrossEncoderReranker

    assert issubclass(CrossEncoderReranker, BaseReranker)


# ---------------------------------------------------------------------------
# Reranking logic with mock cross-encoder (no download)
# ---------------------------------------------------------------------------
async def test_cross_encoder_reranker_reorders_chunks_with_mock_model():
    pytest.importorskip("sentence_transformers")

    from unittest.mock import MagicMock, patch

    from chimera_rag.core.types import Chunk, Query, RetrievalResult
    from chimera_rag.plugins.intent_agent.hooks.cross_encoder_reranker import CrossEncoderReranker

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.9]

    with patch.object(CrossEncoderReranker, "__init__", lambda self, **kw: None):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = mock_model
        reranker.top_n = None
        reranker._model_name = "mock-cross-encoder"

    chunks = [
        Chunk(chunk_id="a", doc_id="d", index=0, text="completely unrelated"),
        Chunk(chunk_id="b", doc_id="d", index=1, text="Alfred Nobel invented dynamite"),
    ]
    res = RetrievalResult(chunks=chunks, scores=[0.9, 0.1])

    reranked = await reranker.rerank(Query(text="who invented dynamite?"), res)

    assert reranked.chunks[0].chunk_id == "b"
    assert reranked.chunks[1].chunk_id == "a"
    assert reranked.metadata["reranked_by"] == "cross_encoder/mock-cross-encoder"


async def test_cross_encoder_reranker_top_n_truncates():
    pytest.importorskip("sentence_transformers")

    from unittest.mock import MagicMock, patch

    from chimera_rag.core.types import Chunk, Query, RetrievalResult
    from chimera_rag.plugins.intent_agent.hooks.cross_encoder_reranker import CrossEncoderReranker

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.5, 0.1]

    with patch.object(CrossEncoderReranker, "__init__", lambda self, **kw: None):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = mock_model
        reranker.top_n = 2
        reranker._model_name = "mock"

    chunks = [
        Chunk(chunk_id="c0", doc_id="d", index=0, text="text 0"),
        Chunk(chunk_id="c1", doc_id="d", index=1, text="text 1"),
        Chunk(chunk_id="c2", doc_id="d", index=2, text="text 2"),
    ]
    res = RetrievalResult(chunks=chunks, scores=[0.9, 0.5, 0.1])

    reranked = await reranker.rerank(Query(text="test"), res)
    assert len(reranked.chunks) == 2


async def test_cross_encoder_reranker_empty_input_returns_unchanged():
    pytest.importorskip("sentence_transformers")

    from unittest.mock import MagicMock, patch

    from chimera_rag.core.types import Query, RetrievalResult
    from chimera_rag.plugins.intent_agent.hooks.cross_encoder_reranker import CrossEncoderReranker

    with patch.object(CrossEncoderReranker, "__init__", lambda self, **kw: None):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = MagicMock()
        reranker.top_n = None
        reranker._model_name = "mock"

    res = RetrievalResult(chunks=[], scores=[])
    reranked = await reranker.rerank(Query(text="anything"), res)
    assert reranked.chunks == []


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
def test_require_cross_encoder_raises_import_error_without_extra():
    from unittest.mock import patch

    from chimera_rag.plugins.intent_agent.hooks.cross_encoder_reranker import _require_cross_encoder

    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(ImportError, match="CrossEncoder reranker requires"):
            _require_cross_encoder()


# ---------------------------------------------------------------------------
# Facade wiring: _wire_reranker (uses conftest fixtures)
# ---------------------------------------------------------------------------
def test_wire_reranker_disabled_leaves_wiring_unchanged(mock_intent_only_yaml):
    """When reranking.enabled is false, no reranker is injected."""
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.config import load_config

    config = load_config(mock_intent_only_yaml)
    wiring: dict = {}
    rag = ChimeraRAG.__new__(ChimeraRAG)
    rag.config = config
    rag._wire_reranker(wiring)
    assert "reranker" not in wiring


def test_wire_reranker_cross_encoder_type_creates_reranker(mock_intent_only_yaml):
    """When reranking.enabled=true and type=cross_encoder, a reranker is wired."""
    pytest.importorskip("sentence_transformers")

    from unittest.mock import patch

    from chimera_rag import ChimeraRAG
    from chimera_rag.core.config import load_config
    from chimera_rag.plugins.intent_agent.hooks.cross_encoder_reranker import CrossEncoderReranker

    config = load_config(mock_intent_only_yaml)
    # Enable cross_encoder reranking in-memory.
    config.plugins.intent_agent.hooks["reranking"] = {
        "enabled": True,
        "type": "cross_encoder",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_n": 10,
    }

    wiring: dict = {}
    rag = ChimeraRAG.__new__(ChimeraRAG)
    rag.config = config

    with patch.object(CrossEncoderReranker, "__init__", return_value=None):
        rag._wire_reranker(wiring)

    assert "reranker" in wiring
    assert isinstance(wiring["reranker"], CrossEncoderReranker)


def test_wire_reranker_token_overlap_type_creates_reranker(mock_intent_only_yaml):
    """When type=token_overlap, TokenOverlapRerankerHook is used (no extra needed)."""
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.config import load_config
    from chimera_rag.plugins.intent_agent.hooks.reranking_hook import TokenOverlapRerankerHook

    config = load_config(mock_intent_only_yaml)
    config.plugins.intent_agent.hooks["reranking"] = {
        "enabled": True,
        "type": "token_overlap",
        "top_n": 5,
    }

    wiring: dict = {}
    rag = ChimeraRAG.__new__(ChimeraRAG)
    rag.config = config
    rag._wire_reranker(wiring)

    assert "reranker" in wiring
    assert isinstance(wiring["reranker"], TokenOverlapRerankerHook)
    assert wiring["reranker"].top_n == 5

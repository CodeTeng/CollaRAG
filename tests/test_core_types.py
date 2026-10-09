"""Tests for :mod:`chimera_rag.core.types`.

These cover the foundational Pydantic models used across the whole framework.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
def test_document_requires_doc_id_and_content():
    from chimera_rag.core.types import Document

    doc = Document(doc_id="d1", content="hello world")
    assert doc.doc_id == "d1"
    assert doc.content == "hello world"
    # metadata defaults to empty dict
    assert doc.metadata == {}


def test_document_rejects_empty_doc_id():
    from pydantic import ValidationError

    from chimera_rag.core.types import Document

    with pytest.raises(ValidationError):
        Document(doc_id="", content="x")


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------
def test_chunk_links_back_to_doc_id_and_has_index():
    from chimera_rag.core.types import Chunk

    ch = Chunk(chunk_id="d1::0", doc_id="d1", index=0, text="sentence A.")
    assert ch.chunk_id == "d1::0"
    assert ch.doc_id == "d1"
    assert ch.index == 0
    assert ch.text == "sentence A."
    assert ch.metadata == {}


def test_chunk_index_must_be_non_negative():
    from pydantic import ValidationError

    from chimera_rag.core.types import Chunk

    with pytest.raises(ValidationError):
        Chunk(chunk_id="x", doc_id="d1", index=-1, text="x")


# ---------------------------------------------------------------------------
# Triple
# ---------------------------------------------------------------------------
def test_triple_captures_subject_predicate_object_and_source_chunk():
    from chimera_rag.core.types import Triple

    t = Triple(
        subject="Alfred Nobel",
        predicate="invented",
        object="dynamite",
        source_chunk_id="d1::0",
    )
    assert t.subject == "Alfred Nobel"
    assert t.predicate == "invented"
    assert t.object == "dynamite"
    assert t.source_chunk_id == "d1::0"
    # confidence defaults to 1.0 and layer defaults to 'L1'
    assert t.confidence == 1.0
    assert t.layer == "L1"


def test_triple_confidence_is_bounded_0_to_1():
    from pydantic import ValidationError

    from chimera_rag.core.types import Triple

    with pytest.raises(ValidationError):
        Triple(subject="a", predicate="b", object="c", confidence=1.5)


# ---------------------------------------------------------------------------
# Query / Intent / RetrievalResult / Answer
# ---------------------------------------------------------------------------
def test_query_carries_text_and_optional_session_id():
    from chimera_rag.core.types import Query

    q = Query(text="who invented dynamite?")
    assert q.text == "who invented dynamite?"
    assert q.session_id is None

    q2 = Query(text="follow-up", session_id="s-42")
    assert q2.session_id == "s-42"


def test_intent_uses_known_labels_and_confidence():
    from chimera_rag.core.types import Intent

    intent = Intent(label="factual", confidence=0.9)
    assert intent.label == "factual"
    assert intent.confidence == 0.9


def test_retrieval_result_collects_chunks_and_triples():
    from chimera_rag.core.types import Chunk, RetrievalResult, Triple

    chunk = Chunk(chunk_id="c1", doc_id="d1", index=0, text="x")
    triple = Triple(subject="a", predicate="p", object="b", source_chunk_id="c1")
    result = RetrievalResult(chunks=[chunk], triples=[triple], scores=[0.9])

    assert result.chunks[0].chunk_id == "c1"
    assert result.triples[0].subject == "a"
    assert result.scores == [0.9]


def test_answer_links_back_to_evidence():
    from chimera_rag.core.types import Answer

    ans = Answer(text="dynamite", evidence_chunk_ids=["c1", "c2"], intent_label="factual")
    assert ans.text == "dynamite"
    assert ans.evidence_chunk_ids == ["c1", "c2"]
    assert ans.intent_label == "factual"

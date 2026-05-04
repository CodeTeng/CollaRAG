"""Tests for :mod:`chimera_rag.datasets` loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Base registry / factory
# ---------------------------------------------------------------------------
def test_get_dataset_loader_returns_matching_class_by_name():
    from chimera_rag.datasets import get_dataset_loader
    from chimera_rag.datasets.plain_text import PlainTextLoader

    assert get_dataset_loader("plain_text") is PlainTextLoader


def test_get_dataset_loader_raises_for_unknown_name():
    from chimera_rag.core.exceptions import DatasetError
    from chimera_rag.datasets import get_dataset_loader

    with pytest.raises(DatasetError, match="unknown dataset loader"):
        get_dataset_loader("nonexistent")


# ---------------------------------------------------------------------------
# PlainTextLoader
# ---------------------------------------------------------------------------
def test_plain_text_loader_returns_documents_from_file(tmp_path: Path):
    from chimera_rag.datasets.plain_text import PlainTextLoader

    p = tmp_path / "doc.txt"
    p.write_text("Hello.\n\nThis is a test.\n\nThird paragraph.")

    docs = PlainTextLoader(path=str(p)).load_documents()
    assert len(docs) >= 1
    # doc_ids must be non-empty strings
    assert all(isinstance(d.doc_id, str) and d.doc_id for d in docs)


def test_plain_text_loader_raises_for_missing_file(tmp_path: Path):
    from chimera_rag.core.exceptions import DatasetError
    from chimera_rag.datasets.plain_text import PlainTextLoader

    with pytest.raises(DatasetError):
        PlainTextLoader(path=str(tmp_path / "nope.txt")).load_documents()


# ---------------------------------------------------------------------------
# GenericJsonlLoader - for QA evaluation
# ---------------------------------------------------------------------------
def test_generic_jsonl_loader_reads_qa_examples(tmp_path: Path):
    from chimera_rag.datasets.generic_jsonl import GenericJsonlLoader

    p = tmp_path / "qa.jsonl"
    data = [
        {"qid": "q1", "question": "Who invented dynamite?", "answer": "Alfred Nobel"},
        {"qid": "q2", "question": "What is relativity?", "answer": "A theory by Einstein"},
    ]
    p.write_text("\n".join(json.dumps(d) for d in data))

    examples = GenericJsonlLoader(path=str(p)).load_examples(limit=None)
    assert len(examples) == 2
    assert examples[0].qid == "q1"
    assert examples[0].question == "Who invented dynamite?"
    assert examples[0].answer == "Alfred Nobel"


def test_generic_jsonl_loader_supports_limit(tmp_path: Path):
    from chimera_rag.datasets.generic_jsonl import GenericJsonlLoader

    p = tmp_path / "qa.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"qid": f"q{i}", "question": f"Q{i}?", "answer": f"A{i}"})
            for i in range(10)
        )
    )
    examples = GenericJsonlLoader(path=str(p)).load_examples(limit=3)
    assert len(examples) == 3


def test_generic_jsonl_loader_field_mapping(tmp_path: Path):
    """Allow remapping column names when a source dataset uses different keys."""
    from chimera_rag.datasets.generic_jsonl import GenericJsonlLoader

    p = tmp_path / "qa.jsonl"
    data = [{"id": "x", "q": "?", "a": "!"}]
    p.write_text(json.dumps(data[0]))

    examples = GenericJsonlLoader(
        path=str(p),
        eval_field_mapping={"qid": "id", "question": "q", "answer": "a"},
    ).load_examples()
    assert examples[0].qid == "x"
    assert examples[0].question == "?"
    assert examples[0].answer == "!"

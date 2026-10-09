"""Tests for the MockWikiLoader — a combined corpus + QA loader.

Assembled from an existing plain-text corpus file and a JSONL QA file
so a single ``config.datasets.mock_wiki`` entry drives both
ingestion and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    # Mini corpus + QA file for tests.
    (tmp_path / "corpus.txt").write_text(
        "Alice works at Acme Corp.\n\n"
        "Acme Corp is headquartered in Paris.\n",
        encoding="utf-8",
    )
    with (tmp_path / "qa.jsonl").open("w", encoding="utf-8") as f:
        for row in [
            {"qid": "t-001", "question": "Where does Alice work?", "answer": "Acme Corp"},
            {"qid": "t-002", "question": "Where is Acme Corp based?", "answer": "Paris"},
        ]:
            f.write(json.dumps(row) + "\n")
    return tmp_path


def test_loader_reads_documents_from_corpus_file(workspace: Path):
    from chimera_rag.datasets.mock_wiki import MockWikiLoader

    loader = MockWikiLoader(path=str(workspace), corpus_file="corpus.txt", qa_file="qa.jsonl")
    docs = loader.load_documents()
    assert len(docs) == 2
    assert docs[0].content.startswith("Alice works at")
    assert docs[1].content.startswith("Acme Corp is")


def test_loader_reads_examples_from_qa_file(workspace: Path):
    from chimera_rag.datasets.mock_wiki import MockWikiLoader

    loader = MockWikiLoader(path=str(workspace), corpus_file="corpus.txt", qa_file="qa.jsonl")
    examples = loader.load_examples()
    assert [e.qid for e in examples] == ["t-001", "t-002"]
    assert examples[0].question == "Where does Alice work?"
    assert examples[0].answer == "Acme Corp"


def test_loader_limit_caps_examples(workspace: Path):
    from chimera_rag.datasets.mock_wiki import MockWikiLoader

    loader = MockWikiLoader(path=str(workspace), corpus_file="corpus.txt", qa_file="qa.jsonl")
    assert len(loader.load_examples(limit=1)) == 1


def test_loader_registered_in_factory():
    from chimera_rag.datasets import get_dataset_loader
    from chimera_rag.datasets.mock_wiki import MockWikiLoader

    assert get_dataset_loader("mock_wiki") is MockWikiLoader


def test_loader_supports_path_pointing_to_dir(workspace: Path):
    # When ``path`` is a directory, corpus_file / qa_file are looked up inside.
    from chimera_rag.datasets.mock_wiki import MockWikiLoader

    loader = MockWikiLoader(path=str(workspace))
    assert len(loader.load_documents()) == 2
    assert len(loader.load_examples()) == 2

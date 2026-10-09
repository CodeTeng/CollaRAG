"""Tests: DatasetConfig supports corpus_path / qa_path / output_dir.

Priority: new explicit fields > legacy ``path`` (kept for backward compat).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_dataset_config_accepts_corpus_and_qa_paths():
    from chimera_rag.core.config import DatasetConfig

    cfg = DatasetConfig(
        loader="mock_wiki",
        corpus_path="data/mock_wiki/corpus.txt",
        qa_path="data/mock_wiki/qa.jsonl",
        output_dir="output/mock_wiki",
    )
    assert cfg.corpus_path == "data/mock_wiki/corpus.txt"
    assert cfg.qa_path == "data/mock_wiki/qa.jsonl"
    assert cfg.output_dir == "output/mock_wiki"


def test_dataset_config_legacy_path_still_works():
    from chimera_rag.core.config import DatasetConfig

    cfg = DatasetConfig(loader="plain_text", path="data/foo.txt")
    assert cfg.path == "data/foo.txt"
    assert cfg.corpus_path is None
    assert cfg.qa_path is None
    assert cfg.output_dir is None


def test_dataset_config_rejects_no_path():
    """At least one of path / corpus_path / qa_path must be provided."""
    from chimera_rag.core.config import ConfigError, DatasetConfig

    with pytest.raises((ValueError, ConfigError)):
        DatasetConfig(loader="mock_wiki")  # nothing at all -> invalid


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "c.txt").write_text(
        "Alice works at Acme.\n\nAcme is in Paris.\n", encoding="utf-8"
    )
    with (tmp_path / "q.jsonl").open("w", encoding="utf-8") as f:
        for row in [
            {"qid": "t1", "question": "Where does Alice work?", "answer": "Acme"},
        ]:
            f.write(json.dumps(row) + "\n")
    return tmp_path


def test_mock_wiki_loader_accepts_explicit_corpus_and_qa_paths(workspace: Path):
    from chimera_rag.datasets.mock_wiki import MockWikiLoader

    loader = MockWikiLoader(
        corpus_path=str(workspace / "c.txt"),
        qa_path=str(workspace / "q.jsonl"),
    )
    assert len(loader.load_documents()) == 2
    assert len(loader.load_examples()) == 1


def test_cli_builds_loader_from_new_config(workspace: Path, monkeypatch):
    """cli._load_dataset_from_config should honour corpus_path/qa_path."""
    from chimera_rag.cli import _load_dataset_from_config
    from chimera_rag.core.config import AppConfig

    # Minimal AppConfig stub with one mock_wiki dataset
    cfg_raw = {
        "app": {"name": "t", "version": "0.1", "workspace_dir": "./ws"},
        "logging": {"level": "INFO", "format": "rich"},
        "llm": {"provider": "mock", "model": "m"},
        "embedding": {"provider": "mock", "model": "m", "dim": 16},
        "storage": {
            "graph": {"backend": "networkx", "persist_path": "./g.pkl"},
            "vector": {"backend": "faiss", "persist_path": "./v"},
        },
        "datasets": {
            "mock": {
                "loader": "mock_wiki",
                "corpus_path": str(workspace / "c.txt"),
                "qa_path": str(workspace / "q.jsonl"),
                "output_dir": str(workspace / "out"),
            }
        },
        "ingestion": {
            "chunker": {"active": "defaults.fixed", "params": {}},
            "extractor": {"active": "defaults.simple_llm", "params": {}},
            "pruner": {"active": "defaults.noop", "params": {}},
        },
        "query": {
            "intent_classifier": {"active": "defaults.rule", "params": {}},
            "retriever": {"active": "defaults.direct", "params": {}},
            "generator": {"active": "defaults.prompt", "params": {}},
        },
    }
    cfg = AppConfig.model_validate(cfg_raw)

    loader = _load_dataset_from_config(cfg, "mock")
    assert len(loader.load_documents()) == 2
    assert len(loader.load_examples()) == 1


def test_dataset_config_get_output_path_helper():
    """Helper that resolves the output file path for a given config name."""
    from chimera_rag.core.config import DatasetConfig

    cfg = DatasetConfig(
        loader="mock_wiki",
        corpus_path="c.txt",
        qa_path="q.jsonl",
        output_dir="output/mock",
    )
    # output_for(config_name) -> output_dir/<config_name>.json
    p = cfg.output_for("full")
    assert str(p).replace("\\", "/") == "output/mock/full.json"

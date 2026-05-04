"""Ensure ChimeraRAG.from_config loads .env sitting next to the config.

We expect:
1. a ``.env`` file in the workspace root (cwd) is loaded automatically
2. values already set in os.environ take precedence (not overwritten)
3. missing .env does not crash
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_load_dotenv_is_called_when_env_file_exists(tmp_workspace: Path, monkeypatch):
    env_path = tmp_workspace / ".env"
    env_path.write_text("CHIMERA_TEST_VAR=from_dotenv\n")
    monkeypatch.delenv("CHIMERA_TEST_VAR", raising=False)

    # Triggering the loader directly to verify behaviour.
    from chimera_rag.core.env_loader import load_env_file

    load_env_file()
    assert os.environ.get("CHIMERA_TEST_VAR") == "from_dotenv"


def test_os_environ_takes_precedence_over_dotenv(tmp_workspace: Path, monkeypatch):
    (tmp_workspace / ".env").write_text("CHIMERA_TEST_VAR=from_dotenv\n")
    monkeypatch.setenv("CHIMERA_TEST_VAR", "from_env")

    from chimera_rag.core.env_loader import load_env_file

    load_env_file()
    # Existing env should win
    assert os.environ["CHIMERA_TEST_VAR"] == "from_env"


def test_missing_env_file_is_a_noop(tmp_workspace: Path):
    # No .env file in tmp_workspace
    from chimera_rag.core.env_loader import load_env_file

    load_env_file()  # should not raise


def test_from_config_triggers_env_load(tmp_workspace: Path, monkeypatch):
    cfg_path = tmp_workspace / "cfg.yaml"
    cfg_path.write_text(textwrap.dedent("""
        app: {name: t, version: 0.1.0, workspace_dir: ./ws}
        logging: {level: INFO, format: rich}
        llm: {provider: mock, model: mock-llm}
        embedding: {provider: mock, model: mock-embedding, dim: 16}
        storage:
          graph: {backend: networkx, persist_path: ./g.pkl}
          vector: {backend: faiss, index_type: flat_ip, persist_path: ./v, top_k: 3}
        datasets: {}
        ingestion:
          chunker: {active: defaults.fixed, params: {}}
          extractor: {active: defaults.simple_llm, params: {}}
          pruner: {active: defaults.noop, params: {}}
        query:
          intent_classifier: {active: defaults.rule, params: {}}
          retriever: {active: defaults.direct, params: {}}
          generator: {active: defaults.prompt, params: {}}
        plugins: {}
        web: {host: 127.0.0.1, port: 8000, cors_origins: [], static_dir: ./static}
        evaluation: {metrics: [em], max_samples: 10, parallel: 1}
    """).strip())
    (tmp_workspace / ".env").write_text("CHIMERA_TEST_FROM_CONFIG=value\n")
    monkeypatch.delenv("CHIMERA_TEST_FROM_CONFIG", raising=False)

    from chimera_rag import ChimeraRAG

    ChimeraRAG.from_config(str(cfg_path))
    assert os.environ.get("CHIMERA_TEST_FROM_CONFIG") == "value"

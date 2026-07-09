"""Tests for :mod:`chimera_rag.core.config`.

AppConfig loads a ``config.yaml`` and exposes typed sub-configs for every
downstream module. We test the loader contract first and let the specific
sub-model shapes grow with real use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MINIMAL_YAML = """\
app:
  name: chimera-rag
  version: 0.1.0
  workspace_dir: ./storage/test

logging: { level: INFO, format: rich, file: null }

llm:
  model: test-llm
  temperature: 0.1
  max_tokens: 256
  timeout: 10
  retry: { max_attempts: 3, initial_wait: 1.0, max_wait: 10.0 }

embedding:
  model: test-embedding
  dim: 16
  batch_size: 8
  normalize: true

rerank:
  enabled: false

storage:
  graph:
    backend: networkx
    persist_path: ./storage/test/graph.gpickle
  vector:
    backend: faiss
    index_type: flat_ip
    persist_path: ./storage/test/vectors
    top_k: 5

datasets:
  sample:
    loader: plain_text
    path: ./examples/data/sample.txt

ingestion:
  chunker:   { active: defaults.fixed, params: { chunk_size: 300, overlap: 50 } }
  extractor: { active: defaults.simple_llm, params: {} }
  pruner:    { active: defaults.noop, params: {} }

query:
  intent_classifier: { active: defaults.rule, params: {} }
  retriever:         { active: defaults.direct, params: { top_k: 5, neighbor_hops: 1 } }
  generator:         { active: defaults.prompt, params: {} }

plugins:
  adagraph:     { enabled: false }
  colla_rag: { enabled: false }

web: { host: 0.0.0.0, port: 8000, cors_origins: [http://localhost:5173], static_dir: x }

evaluation: { metrics: [em, f1], max_samples: 10, parallel: 1 }
"""


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(MINIMAL_YAML)
    return p


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
def test_load_config_returns_app_config_with_populated_sections(tmp_config):
    from chimera_rag.core.config import AppConfig, load_config

    cfg = load_config(tmp_config)
    assert isinstance(cfg, AppConfig)
    assert cfg.app.name == "chimera-rag"
    assert cfg.llm.model == "test-llm"
    assert cfg.embedding.dim == 16
    assert cfg.storage.vector.top_k == 5
    assert cfg.ingestion.chunker.active == "defaults.fixed"
    assert cfg.ingestion.chunker.params["chunk_size"] == 300
    assert cfg.query.retriever.active == "defaults.direct"
    assert cfg.plugins.adagraph.enabled is False
    assert cfg.plugins.colla_rag.enabled is False
    assert cfg.web.port == 8000
    assert cfg.evaluation.metrics == ["em", "f1"]


def test_load_config_raises_config_error_for_missing_file(tmp_path: Path):
    from chimera_rag.core.config import load_config
    from chimera_rag.core.exceptions import ConfigError

    with pytest.raises(ConfigError, match="not found|does not exist"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_raises_config_error_on_invalid_yaml(tmp_path: Path):
    from chimera_rag.core.config import load_config
    from chimera_rag.core.exceptions import ConfigError

    bad = tmp_path / "bad.yaml"
    bad.write_text("app: [this is: not valid")

    with pytest.raises(ConfigError):
        load_config(bad)


def test_load_config_raises_config_error_on_missing_required_field(tmp_path: Path):
    from chimera_rag.core.config import load_config
    from chimera_rag.core.exceptions import ConfigError

    bad = tmp_path / "partial.yaml"
    # Remove the whole llm section on purpose.
    bad.write_text(MINIMAL_YAML.replace(
        "llm:\n  model: test-llm\n  temperature: 0.1\n  max_tokens: 256\n  timeout: 10\n  retry: { max_attempts: 3, initial_wait: 1.0, max_wait: 10.0 }\n",
        "",
    ))

    with pytest.raises(ConfigError):
        load_config(bad)


# ---------------------------------------------------------------------------
# Shipped configs — only the 4 plugin-combo variants live in configs/ now.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename",
    [
        "vanilla.yaml",
        "adagraph_only.yaml",
        "colla_rag_only.yaml",
        "full.yaml",
    ],
)
def test_shipped_configs_are_valid(filename: str):
    """Every yaml we ship must pass full Pydantic validation."""
    from chimera_rag.core.config import load_config

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "configs" / filename
    cfg = load_config(cfg_path)
    assert cfg.app.name == "chimera-rag"


def test_plugin_enabled_flag_reflected_per_config(
    mock_vanilla_yaml: str, mock_full_yaml: str
):
    """The generated Mock yamls must faithfully reflect their plugin toggles."""
    from chimera_rag.core.config import load_config

    vanilla = load_config(mock_vanilla_yaml)
    assert vanilla.plugins.adagraph.enabled is False
    assert vanilla.plugins.colla_rag.enabled is False

    full = load_config(mock_full_yaml)
    assert full.plugins.adagraph.enabled is True
    assert full.plugins.colla_rag.enabled is True

"""Shared pytest fixtures.

Most tests want a zero-cost config just to build a ChimeraRAG
instance. We provide FakeLLMProvider / FakeEmbeddingProvider (from
tests/_fakes.py) and monkeypatch them into chimera_rag.providers so
all tests are offline.

We ship 4 canonical configs here as fixtures so we don't need to keep
``configs/mock_*.yaml`` around in the repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _fakes import FakeEmbeddingProvider, FakeLLMProvider

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Monkeypatch the provider factories so ALL tests are offline
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fake_providers(monkeypatch):
    """Replace the real provider factories with fake doubles.

    The config yaml still declares model/dim/... but those values are
    forwarded to the fakes so the config schema is still exercised.
    """
    import chimera_rag.providers

    def _fake_build_llm(cfg):
        return FakeLLMProvider()

    def _fake_build_embedding(cfg):
        return FakeEmbeddingProvider(
            dim=getattr(cfg, "dim", 64),
            normalize=getattr(cfg, "normalize", True),
        )

    def _fake_build_rerank(cfg):
        return None

    monkeypatch.setattr(chimera_rag.providers, "build_llm_provider", _fake_build_llm)
    monkeypatch.setattr(chimera_rag.providers, "build_embedding_provider", _fake_build_embedding)
    monkeypatch.setattr(chimera_rag.providers, "build_rerank_provider", _fake_build_rerank)


# ---------------------------------------------------------------------------
# Config YAML builder (uses the new schema — no "provider" field)
# ---------------------------------------------------------------------------
def _mock_config_yaml(
    *,
    workspace: str,
    adagraph: bool,
    colla_rag: bool,
) -> str:
    sample_path = REPO_ROOT / "examples" / "data" / "sample.txt"

    if adagraph:
        ingestion_lines = [
            "ingestion:",
            "  chunker:",
            "    active: adagraph.dynamic",
            "    params: { chunk_size: 300 }",
            "  extractor:",
            "    active: adagraph.layered",
            "    params: { layers_enabled: [L1] }",
            "  pruner:",
            "    active: adagraph.dual_redundancy",
            "    params: { similarity_threshold: 0.85 }",
        ]
    else:
        ingestion_lines = [
            "ingestion:",
            "  chunker: { active: defaults.fixed, params: { chunk_size: 300, overlap: 50 } }",
            "  extractor: { active: defaults.simple_llm, params: {} }",
            "  pruner: { active: defaults.noop, params: {} }",
        ]

    if colla_rag:
        query_lines = [
            "query:",
            "  intent_classifier:",
            "    active: colla_rag.tree",
            "    params: {}",
            "  retriever:",
            "    active: colla_rag.multi_agent",
            "    params: { top_k: 5, neighbor_hops: 1 }",
            "  generator:",
            "    active: defaults.prompt",
            "    params: { prompt_template: default }",
        ]
    else:
        query_lines = [
            "query:",
            "  intent_classifier: { active: defaults.rule, params: {} }",
            "  retriever: { active: defaults.direct, params: { top_k: 5, neighbor_hops: 1 } }",
            "  generator: { active: defaults.prompt, params: { prompt_template: default } }",
        ]

    plugins_lines = [
        "plugins:",
        "  adagraph:",
        f"    enabled: {str(adagraph).lower()}",
        "    dynamic_chunker: { min_chunk_size: 100, max_chunk_size: 500, base_chunk_size: 300 }",
        "    layered_extractor: { layers_enabled: [L1] }",
        "    dual_pruner: { similarity_threshold: 0.85, enable_transitive: false }",
        "  colla_rag:",
        f"    enabled: {str(colla_rag).lower()}",
        "    memory: { max_history: 10 }",
        "    intent_classifier: { confidence_threshold: 0.6 }",
        "    strategies: { enabled: [direct, multi_hop, parallel, iterative, expansion, context_aware] }",
        "    hooks:",
        "      reflection: { enabled: false }",
        "      reranking: { enabled: false }",
        "      parallel_pool: { enabled: true, max_workers: 4 }",
    ]

    lines = [
        f"app: {{ name: chimera-rag, version: 0.1.0, workspace_dir: {workspace} }}",
        "logging: { level: WARNING, format: plain, file: null }",
        "",
        "llm:",
        "  model: fake-llm",
        "  temperature: 0.1",
        "  max_tokens: 512",
        "  timeout: 30",
        "  retry: { max_attempts: 1, initial_wait: 0.01, max_wait: 0.05 }",
        "",
        "embedding:",
        "  model: fake-embedding",
        "  dim: 64",
        "  batch_size: 32",
        "  normalize: true",
        "",
        "rerank:",
        "  enabled: false",
        "",
        "storage:",
        "  graph:",
        "    backend: networkx",
        f"    persist_path: {workspace}/graph.gpickle",
        "  vector:",
        "    backend: faiss",
        "    index_type: hnsw",
        "    metric: ip",
        f"    persist_path: {workspace}/vectors",
        "    top_k: 5",
        "    hnsw: { M: 16, ef_construction: 100, ef_search: 32 }",
        "    ivf: { nlist: 10, nprobe: 2 }",
        "",
        "datasets:",
        "  sample:",
        "    loader: plain_text",
        f"    path: {sample_path}",
        "",
        *ingestion_lines,
        "",
        *query_lines,
        "",
        *plugins_lines,
        "",
        "web: { host: 127.0.0.1, port: 0, cors_origins: [], static_dir: src/chimera_rag/web/static }",
        "evaluation: { metrics: [em, f1], max_samples: 100, parallel: 1 }",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_vanilla_yaml(tmp_path: Path) -> str:
    ws = str(tmp_path / "vanilla_ws")
    p = tmp_path / "mock_vanilla.yaml"
    p.write_text(_mock_config_yaml(workspace=ws, adagraph=False, colla_rag=False))
    return str(p)


@pytest.fixture
def mock_adagraph_only_yaml(tmp_path: Path) -> str:
    ws = str(tmp_path / "adagraph_ws")
    p = tmp_path / "mock_adagraph_only.yaml"
    p.write_text(_mock_config_yaml(workspace=ws, adagraph=True, colla_rag=False))
    return str(p)


@pytest.fixture
def mock_colla_rag_only_yaml(tmp_path: Path) -> str:
    ws = str(tmp_path / "intent_ws")
    p = tmp_path / "mock_colla_rag_only.yaml"
    p.write_text(_mock_config_yaml(workspace=ws, adagraph=False, colla_rag=True))
    return str(p)


@pytest.fixture
def mock_full_yaml(tmp_path: Path) -> str:
    ws = str(tmp_path / "full_ws")
    p = tmp_path / "mock_full.yaml"
    p.write_text(_mock_config_yaml(workspace=ws, adagraph=True, colla_rag=True))
    return str(p)


@pytest.fixture
def mock_config_yaml_builder():
    """Expose the builder directly for tests that want custom combos."""
    return _mock_config_yaml

"""End-to-end integration test for innovation 2b (LangGraph agentic retrieval).

Builds a full :class:`ChimeraRAG` from a Mock-provider config whose retriever
slot is ``intent_agent.langgraph``, then runs ingest -> query. Deterministic:
the MockLLMProvider returns a canned ReAct ``Action`` so the agent calls a
real tool over the in-memory stores, with no network / API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _langgraph_mock_config(workspace: str) -> str:
    """A Mock-provider config that activates the intent_agent.langgraph retriever.

    The mock LLM is told (via a keyword rule) to emit a ReAct Action so the
    agent exercises the vector_search tool deterministically.
    """
    sample_path = REPO_ROOT / "examples" / "data" / "sample.txt"
    react = (
        'Action: vector_search\\nAction Input: {\\"query\\": \\"sample\\", \\"top_k\\": 3}'
    )
    return "\n".join(
        [
            f"app: {{ name: chimera-rag, version: 0.1.0, workspace_dir: {workspace} }}",
            "logging: { level: WARNING, format: plain, file: null }",
            "",
            "llm:",
            "  provider: mock",
            "  model: mock-llm",
            "  temperature: 0.1",
            "  max_tokens: 512",
            "  timeout: 30",
            "  retry: { max_attempts: 1, initial_wait: 0.01, max_wait: 0.05 }",
            "",
            "embedding:",
            "  provider: mock",
            "  model: mock-embedding",
            "  dim: 64",
            "  batch_size: 32",
            "  normalize: true",
            "  device: cpu",
            "",
            "storage:",
            "  graph:",
            "    backend: networkx",
            f"    persist_path: {workspace}/graph.gpickle",
            "  vector:",
            "    backend: faiss",
            "    index_type: flat_ip",
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
            "ingestion:",
            "  chunker: { active: defaults.fixed, params: { chunk_size: 300, overlap: 50 } }",
            "  extractor: { active: defaults.simple_llm, params: {} }",
            "  pruner: { active: defaults.noop, params: {} }",
            "",
            "query:",
            "  intent_classifier:",
            "    active: intent_agent.llm",
            "    params: {}",
            "  retriever:",
            "    active: intent_agent.langgraph",
            "    params:",
            "      top_k: 5",
            "      neighbor_hops: 1",
            "      max_iterations: 1",
            "      quality_threshold: 0.6",
            "      tools: [vector_search, graph_neighbors, assess_evidence]",
            "  generator:",
            "    active: defaults.prompt",
            "    params: { prompt_template: default }",
            "",
            "plugins:",
            "  adagraph: { enabled: false }",
            "  intent_agent:",
            "    enabled: true",
            "    memory: { max_history: 10 }",
            "    intent_classifier: { confidence_threshold: 0.6 }",
            "    strategies: { enabled: [direct, multi_hop, parallel] }",
            "    hooks:",
            "      reflection: { enabled: true, quality_threshold: 0.6, max_iterations: 1 }",
            "      reranking: { enabled: false }",
            "      parallel_pool: { enabled: true, max_workers: 4 }",
            "",
            "web: { host: 127.0.0.1, port: 0, cors_origins: [], static_dir: src/chimera_rag/web/static }",
            "evaluation: { metrics: [em, f1], max_samples: 100, parallel: 1 }",
            "",
        ]
    )


@pytest.fixture()
def langgraph_config(tmp_path: Path) -> str:
    ws = str(tmp_path / "langgraph_ws")
    p = tmp_path / "mock_langgraph.yaml"
    p.write_text(_langgraph_mock_config(ws))
    return str(p)


def test_langgraph_retriever_activates_when_configured(langgraph_config: str):
    from chimera_rag import ChimeraRAG

    crag = ChimeraRAG.from_config(langgraph_config)
    active = crag.active_implementations()
    assert active["retriever"] == "intent_agent.langgraph"
    assert active["intent_classifier"] == "intent_agent.llm"


def test_langgraph_end_to_end_ingest_then_query(langgraph_config: str):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document, Query

    crag = ChimeraRAG.from_config(langgraph_config)
    crag.ingest(
        [
            Document(
                doc_id="nobel",
                content=(
                    "Alfred Nobel was a Swedish chemist. "
                    "Nobel invented dynamite in 1867. "
                    "Einstein formulated the theory of relativity."
                ),
            )
        ]
    )
    ans = crag.query(Query(text="Who invented dynamite?"))
    assert isinstance(ans.text, str)
    # The langgraph retriever stamps its provenance into the trace.
    assert ans.trace["retrieval_metadata"].get("retriever") == "langgraph"
    assert ans.intent_label in {
        "factual", "analytical", "comparative",
        "multi_hop", "exploratory", "follow_up",
    }

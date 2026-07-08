"""End-to-end test: ChimeraRAG with multi-agent retriever (MockProvider)."""
from chimera_rag import ChimeraRAG
from chimera_rag.core.types import Document


def test_multi_agent_ingest_and_query(tmp_path):
    config_text = f"""
app:
  name: test
  version: "0.1"
  workspace_dir: "{tmp_path}"
logging:
  level: DEBUG
llm:
  provider: mock
  model: mock
embedding:
  provider: mock
  model: mock
  dim: 64
storage:
  graph:
    persist_path: "{tmp_path}/graph.gpickle"
  vector:
    persist_path: "{tmp_path}/vectors"
    index_type: flat_ip
ingestion:
  chunker:
    active: defaults.fixed
    params:
      chunk_size: 100
  extractor:
    active: defaults.simple_llm
    params: {{}}
  pruner:
    active: defaults.noop
    params: {{}}
query:
  intent_classifier:
    active: colla_rag.tree
    params: {{}}
  retriever:
    active: colla_rag.multi_agent
    params: {{}}
  generator:
    active: defaults.prompt
    params: {{}}
plugins:
  colla_rag:
    enabled: true
"""
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(config_text)

    crag = ChimeraRAG.from_config(str(cfg_path))
    docs = [
        Document(doc_id="d1", content="Albert Einstein was born in Ulm, Germany in 1879."),
        Document(doc_id="d2", content="Einstein developed the theory of relativity."),
    ]
    stats = crag.ingest(docs)
    assert stats["documents"] == 2

    answer = crag.query("Where was Einstein born?")
    assert answer.text
    assert answer.intent_label is not None

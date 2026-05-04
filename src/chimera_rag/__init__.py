"""Chimera-RAG: a pluggable knowledge-graph based RAG framework.

Two research innovations are optional plugins, controlled by ``config.yaml``:

- **AdaGraph** (Adaptive Graph Construction) — dynamic chunking + layered
  triple extraction + dual redundancy pruning.
- **IntentAgent** — intent-aware agent with Memory / Reasoning / Decision /
  Execution stages plus pluggable Reflection / Reranking / ParallelPool hooks.

With both plugins disabled the framework degrades gracefully to a working
"vanilla" KG-RAG baseline.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

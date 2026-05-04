"""Chimera-RAG: a pluggable knowledge-graph based RAG framework.

Two research innovations are optional plugins, controlled by ``config.yaml``:

- **AdaGraph** (Adaptive Graph Construction) — dynamic chunking + layered
  triple extraction + dual redundancy pruning.
- **IntentAgent** — intent-aware agent with Memory / Reasoning / Decision /
  Execution stages plus pluggable Reflection / Reranking / ParallelPool hooks.

With both plugins disabled the framework degrades gracefully to a working
"vanilla" KG-RAG baseline.

The top-level facade :class:`ChimeraRAG` wires together all slots from an
``AppConfig`` and exposes three user-facing methods:

* :meth:`ChimeraRAG.from_config` — classmethod constructor from a yaml path,
* :meth:`ChimeraRAG.ingest`      — build the knowledge graph from documents,
* :meth:`ChimeraRAG.query`       — ask a question and get an :class:`Answer`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from pathlib import Path
from typing import TYPE_CHECKING

from chimera_rag.core.config import AppConfig, load_config
from chimera_rag.core.pipeline import IngestionPipeline, PipelineState, QueryPipeline
from chimera_rag.core.registry import get_registry
from chimera_rag.core.types import Answer, Document, Query

# Trigger default @register decorators.
from chimera_rag import defaults as _defaults  # noqa: F401

if TYPE_CHECKING:
    from chimera_rag.interfaces.chunker import BaseChunker
    from chimera_rag.interfaces.extractor import BaseTripleExtractor
    from chimera_rag.interfaces.generator import BaseAnswerGenerator
    from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier
    from chimera_rag.interfaces.pruner import BasePruner
    from chimera_rag.interfaces.retriever import BaseRetriever


class ChimeraRAG:
    """Top-level facade. Build it with :meth:`from_config`."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._build()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, path: str | Path) -> "ChimeraRAG":
        return cls(load_config(path))

    # ------------------------------------------------------------------
    def _build(self) -> None:
        # Trigger plugin registration side-effects (only if enabled) before
        # resolving slots through the Registry.
        self._load_enabled_plugins()

        registry = get_registry()
        cfg = self.config

        # Providers.
        from chimera_rag.providers import build_embedding_provider, build_llm_provider

        self.llm = build_llm_provider(cfg.llm)
        self.embedder = build_embedding_provider(cfg.embedding)

        # Stores.
        self.graph_store, self.vector_store = self._build_stores()
        self.state = PipelineState(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
        )

        # Resolve slot classes via Registry.
        chunker_cls = registry.get("chunker", cfg.ingestion.chunker.active)
        extractor_cls = registry.get("extractor", cfg.ingestion.extractor.active)
        pruner_cls = registry.get("pruner", cfg.ingestion.pruner.active)
        intent_cls = registry.get("intent_classifier", cfg.query.intent_classifier.active)
        retriever_cls = registry.get("retriever", cfg.query.retriever.active)
        generator_cls = registry.get("generator", cfg.query.generator.active)

        # Instantiate slot implementations, injecting dependencies as
        # required.
        self._active_implementations = {
            "chunker": cfg.ingestion.chunker.active,
            "extractor": cfg.ingestion.extractor.active,
            "pruner": cfg.ingestion.pruner.active,
            "intent_classifier": cfg.query.intent_classifier.active,
            "retriever": cfg.query.retriever.active,
            "generator": cfg.query.generator.active,
        }

        chunker = self._instantiate_chunker(chunker_cls, cfg.ingestion.chunker.params)
        extractor = self._instantiate_extractor(extractor_cls, cfg.ingestion.extractor.params)
        pruner = self._instantiate_pruner(pruner_cls, cfg.ingestion.pruner.params)
        intent_classifier = self._instantiate_intent(intent_cls, cfg.query.intent_classifier.params)
        retriever = self._instantiate_retriever(retriever_cls, cfg.query.retriever.params)
        generator = self._instantiate_generator(generator_cls, cfg.query.generator.params)

        self._ingest = IngestionPipeline(
            chunker=chunker,
            extractor=extractor,
            pruner=pruner,
            embedder=self.embedder,
            state=self.state,
        )
        self._query = QueryPipeline(
            intent_classifier=intent_classifier,
            retriever=retriever,
            generator=generator,
            top_k=int(cfg.query.retriever.params.get("top_k", cfg.storage.vector.top_k)),
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    def ingest(self, documents: list[Document]) -> dict:
        return self._ingest.run(documents)

    def query(self, query: Query | str) -> Answer:
        if isinstance(query, str):
            query = Query(text=query)
        return self._query.run(query)

    def stats(self) -> dict:
        return self.state.stats()

    def active_implementations(self) -> dict[str, str]:
        return dict(self._active_implementations)

    # ------------------------------------------------------------------
    # Live reconfiguration (Web hot-toggle)
    # ------------------------------------------------------------------
    def toggle_plugin(self, name: str, enabled: bool) -> None:
        """Flip a plugin's enabled flag and rebuild the pipelines.

        Preserves the existing :class:`PipelineState` (graph + vector +
        chunk_lookup) so callers don't have to re-ingest. When a plugin
        is enabled we also swap the active slot names to the plugin's
        canonical names; when disabled, we revert to the ``defaults.*``
        baseline.
        """
        if name == "adagraph":
            self.config.plugins.adagraph.enabled = enabled
            if enabled:
                self.config.ingestion.chunker.active = "adagraph.dynamic"
                self.config.ingestion.extractor.active = "adagraph.layered"
                self.config.ingestion.pruner.active = "adagraph.dual_redundancy"
            else:
                self.config.ingestion.chunker.active = "defaults.fixed"
                self.config.ingestion.extractor.active = "defaults.simple_llm"
                self.config.ingestion.pruner.active = "defaults.noop"
        elif name == "intent_agent":
            self.config.plugins.intent_agent.enabled = enabled
            if enabled:
                self.config.query.intent_classifier.active = "intent_agent.llm"
                self.config.query.retriever.active = "intent_agent.strategy_selector"
            else:
                self.config.query.intent_classifier.active = "defaults.rule"
                self.config.query.retriever.active = "defaults.direct"
        else:
            raise ValueError(f"unknown plugin name: {name!r}")

        self._rebuild_pipelines_preserving_state()

    def _rebuild_pipelines_preserving_state(self) -> None:
        """Re-run _build() but keep the existing stores + chunk_lookup."""
        saved_state = self.state
        saved_graph = self.graph_store
        saved_vector = self.vector_store
        self._build()
        # Restore prior state (graph + vectors + chunks).
        self.graph_store = saved_graph
        self.vector_store = saved_vector
        self.state = saved_state
        # Re-attach to the query retriever so it sees the same chunk_lookup.
        retr = self._query.retriever
        if hasattr(retr, "chunk_lookup"):
            retr.chunk_lookup = self.state.chunk_lookup
        if hasattr(retr, "graph_store"):
            retr.graph_store = self.graph_store
        if hasattr(retr, "vector_store"):
            retr.vector_store = self.vector_store
        # IntentAgentRetriever wraps a base retriever; patch its base too.
        base = getattr(retr, "base", None)
        if base is not None:
            base.chunk_lookup = self.state.chunk_lookup
            base.graph_store = self.graph_store
            base.vector_store = self.vector_store

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_enabled_plugins(self) -> None:
        import importlib

        plugins_cfg = self.config.plugins
        if plugins_cfg.adagraph.enabled:
            try:
                importlib.import_module("chimera_rag.plugins.adagraph")
            except ImportError as e:  # pragma: no cover - optional
                import logging

                logging.getLogger(__name__).warning("adagraph plugin not available: %s", e)

        if plugins_cfg.intent_agent.enabled:
            try:
                importlib.import_module("chimera_rag.plugins.intent_agent")
            except ImportError as e:  # pragma: no cover - optional
                import logging

                logging.getLogger(__name__).warning("intent_agent plugin not available: %s", e)

    def _build_stores(self):
        from chimera_rag.stores.graph.networkx_store import NetworkXGraphStore
        from chimera_rag.stores.vector.faiss_store import FAISSVectorStore

        graph_cfg = self.config.storage.graph
        if graph_cfg.backend == "networkx":
            graph = NetworkXGraphStore()
        elif graph_cfg.backend == "neo4j":  # pragma: no cover - placeholder
            from chimera_rag.stores.graph.neo4j_store import Neo4jGraphStore

            graph = Neo4jGraphStore(
                uri=graph_cfg.neo4j_uri or "bolt://localhost:7687",
                user=graph_cfg.neo4j_user or "neo4j",
                password="",
            )
        else:  # pragma: no cover - guarded by Pydantic
            raise ValueError(f"unknown graph backend: {graph_cfg.backend}")

        vec = FAISSVectorStore(dim=self.config.embedding.dim, index_type=self.config.storage.vector.index_type)
        return graph, vec

    # --- instantiators isolate slot-specific dependency wiring --------
    @staticmethod
    def _filter_params(cls: type, params: dict) -> dict:
        """Keep only kwargs accepted by ``cls.__init__`` to avoid TypeError.

        This lets us reuse the same ``params`` dict across different
        implementations of the same slot (e.g. defaults.fixed vs
        adagraph.dynamic), which is essential for live plugin toggling.
        """
        import inspect

        # If the class does not define its own __init__ (uses object's
        # default), reject all params — object.__init__ actually takes none.
        own_init = cls.__init__ is not object.__init__
        if not own_init:
            return {}

        sig = inspect.signature(cls.__init__)
        allowed = {p for p in sig.parameters if p != "self"}
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(params)  # ctor accepts **kwargs
        return {k: v for k, v in params.items() if k in allowed}

    def _instantiate_chunker(self, cls: type, params: dict) -> "BaseChunker":
        return cls(**self._filter_params(cls, params))

    def _instantiate_extractor(self, cls: type, params: dict) -> "BaseTripleExtractor":
        call_params = self._filter_params(cls, params)
        # Strip labels like "default" that refer to the built-in template.
        if call_params.get("prompt_template") == "default":
            call_params.pop("prompt_template", None)
        return cls(llm=self.llm, **call_params)

    def _instantiate_pruner(self, cls: type, params: dict) -> "BasePruner":
        return cls(**self._filter_params(cls, params))

    def _instantiate_intent(self, cls: type, params: dict) -> "BaseIntentClassifier":
        import inspect

        sig = inspect.signature(cls.__init__)
        call_params = self._filter_params(cls, params)
        # If the class wants an LLM (e.g. LLMIntentClassifier), inject ours.
        if "llm" in sig.parameters and "llm" not in call_params:
            call_params["llm"] = self.llm
        return cls(**call_params)

    def _instantiate_retriever(self, cls: type, params: dict) -> "BaseRetriever":
        import inspect

        sig = inspect.signature(cls.__init__)
        call_params = self._filter_params(cls, params)
        # top_k is handled by QueryPipeline.
        call_params.pop("top_k", None)

        # Classic retrievers (DirectRetriever) want vector_store + graph_store + embedder
        # + chunk_lookup. Smart retrievers (IntentAgentRetriever) want a base_retriever
        # + classifier + optional hooks. We inspect the ctor and adapt.
        param_names = set(sig.parameters.keys())

        if "base_retriever" in param_names:
            # IntentAgent-style retriever: wrap a DirectRetriever under the hood.
            from chimera_rag.defaults.direct_retriever import DirectRetriever

            inner_hops = int(call_params.pop("neighbor_hops", 1))
            base = DirectRetriever(
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                embedder=self.embedder,
                chunk_lookup=self.state.chunk_lookup,
                neighbor_hops=inner_hops,
            )
            wiring: dict = {"base_retriever": base}
            if "classifier" in param_names:
                # Use the already-instantiated query intent classifier if we have one,
                # otherwise create a default LLM one.
                from chimera_rag.plugins.intent_agent.intent_classifier import (
                    LLMIntentClassifier,
                )

                wiring["classifier"] = LLMIntentClassifier(llm=self.llm)
            return cls(**{**wiring, **call_params})

        # Classic DirectRetriever signature.
        return cls(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
            chunk_lookup=self.state.chunk_lookup,
            **call_params,
        )

    def _instantiate_generator(self, cls: type, params: dict) -> "BaseAnswerGenerator":
        call_params = self._filter_params(cls, params)
        if call_params.get("prompt_template") == "default":
            call_params.pop("prompt_template", None)
        return cls(llm=self.llm, **call_params)


__all__ = ["ChimeraRAG", "__version__"]

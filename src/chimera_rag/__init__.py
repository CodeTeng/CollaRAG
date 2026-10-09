"""Chimera-RAG: a pluggable knowledge-graph based RAG framework.

Two research innovations are optional plugins, controlled by ``config.yaml``:

- **AdaGraph** (build-side baseline) — dynamic chunking + layered
  triple extraction + dual redundancy pruning. NOTE: this is the
  engineering baseline reserved for innovation #1 (a novel chunking
  method, not yet implemented), not itself a claimed contribution.
- **CollaRAG** (innovation #2) — graph-grounded multi-agent retrieval
  with four sub-contributions: intent-aware query triage (C1),
  G-PER plan-execute-reflect (C2), graph-grounded dual-layer memory
  (C3), and graph-conditioned capability specialization (C4).

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

# Trigger default @register decorators.
from chimera_rag import defaults as _defaults  # noqa: F401
from chimera_rag.core.config import AppConfig, load_config
from chimera_rag.core.pipeline import IngestionPipeline, PipelineState, QueryPipeline
from chimera_rag.core.registry import get_registry
from chimera_rag.core.types import Answer, Document, Query

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
    def from_config(cls, path: str | Path) -> ChimeraRAG:
        # Load a local .env before building so providers can resolve
        # api_key_env / base_url_env from DEEPSEEK_API_KEY etc.
        from chimera_rag.core.env_loader import load_env_file

        load_env_file()
        return cls(load_config(path))

    # ------------------------------------------------------------------
    def _build(self) -> None:
        # Trigger plugin registration side-effects (only if enabled) before
        # resolving slots through the Registry.
        self._load_enabled_plugins()

        registry = get_registry()
        cfg = self.config

        # Providers.
        from chimera_rag.providers import (
            build_embedding_provider,
            build_llm_provider,
            build_rerank_provider,
        )

        self.llm = build_llm_provider(cfg.llm)
        self.embedder = build_embedding_provider(cfg.embedding)
        self.reranker = build_rerank_provider(cfg.rerank)

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

        # Parallelism for the per-chunk LLM extraction step. Read from the
        # extractor params (ingestion.extractor.params.extract_workers); falls
        # back to 1 (legacy serial) when unset. This is the main lever to cut
        # ingest wall-clock against slow local LLMs (e.g. ollama 8B).
        extract_workers = int(cfg.ingestion.extractor.params.get("extract_workers", 1))

        self._ingest = IngestionPipeline(
            chunker=chunker,
            extractor=extractor,
            pruner=pruner,
            embedder=self.embedder,
            state=self.state,
            extract_workers=extract_workers,
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
    # Graph persistence — save/restore the built KG so repeated runs can
    # skip the (slow) ingest. Artefacts are stored under a per-dataset
    # subdirectory so that different datasets do NOT overwrite each other:
    #
    #   storage/<config_stem>/<dataset>/graph.gpickle
    #   storage/<config_stem>/<dataset>/vectors.faiss + .ids.pkl
    #   storage/<config_stem>/<dataset>/graph.gpickle.chunks.pkl
    #   storage/<config_stem>/<dataset>/graph.gpickle.meta.json
    #
    # When dataset is None or empty, the paths fall back to the configured
    # persist_path directly (backward compatible).
    # ------------------------------------------------------------------

    def _resolve_graph_path(self, dataset: str | None) -> str:
        """Return graph persist path, optionally scoped by dataset."""
        base = self.config.storage.graph.persist_path
        if not dataset:
            return base
        from pathlib import Path as _P
        parent = _P(base).parent / dataset
        parent.mkdir(parents=True, exist_ok=True)
        return str(parent / _P(base).name)

    def _resolve_vector_path(self, dataset: str | None) -> str:
        """Return vector persist path, optionally scoped by dataset."""
        base = self.config.storage.vector.persist_path
        if not dataset:
            return base
        from pathlib import Path as _P
        parent = _P(base).parent / dataset
        parent.mkdir(parents=True, exist_ok=True)
        return str(parent / _P(base).name)

    def _persist_fingerprint(self, dataset: str | None) -> str:
        """Hash the ingest-relevant config + dataset so a changed pipeline
        (chunker / extractor / pruner / embedding / dataset) invalidates a
        previously persisted graph instead of silently reusing a stale one.
        """
        import hashlib
        import json as _json

        cfg = self.config
        material = {
            "dataset": dataset or "",
            "chunker": {"active": cfg.ingestion.chunker.active, "params": cfg.ingestion.chunker.params},
            "extractor": {"active": cfg.ingestion.extractor.active, "params": cfg.ingestion.extractor.params},
            "pruner": {"active": cfg.ingestion.pruner.active, "params": cfg.ingestion.pruner.params},
            "embedding": {"model": cfg.embedding.model, "dim": cfg.embedding.dim},
        }
        blob = _json.dumps(material, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _chunks_path(self, dataset: str | None = None) -> str:
        return self._resolve_graph_path(dataset) + ".chunks.pkl"

    def _meta_path(self, dataset: str | None = None) -> str:
        return self._resolve_graph_path(dataset) + ".meta.json"

    def persist_graph(self, dataset: str | None = None) -> dict[str, str]:
        """Write graph + vector index + chunk_lookup + fingerprint to disk."""
        import json as _json
        import pickle
        from pathlib import Path as _Path

        graph_path = self._resolve_graph_path(dataset)
        vector_path = self._resolve_vector_path(dataset)

        _Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
        _Path(vector_path).parent.mkdir(parents=True, exist_ok=True)

        self.graph_store.persist(graph_path)
        self.vector_store.persist(vector_path)

        chunks_path = self._chunks_path(dataset)
        _Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.state.chunk_lookup, f)

        meta = {
            "fingerprint": self._persist_fingerprint(dataset),
            "dataset": dataset or "",
            "chunks": len(self.state.chunk_lookup),
            "triples": len(self.graph_store.all_triples()),
        }
        with open(self._meta_path(dataset), "w", encoding="utf-8") as f:
            _json.dump(meta, f, ensure_ascii=False, indent=2)

        return {"graph": graph_path, "vector": vector_path, "chunks": chunks_path, "meta": self._meta_path(dataset)}

    def try_load_graph(self, dataset: str | None = None) -> bool:
        """Load a persisted graph if present and fingerprint-compatible.

        Returns True on a successful load (pipelines now have a usable KG),
        False if nothing was loaded (caller should ingest). A fingerprint
        mismatch or any corrupted/partial artefact is treated as a miss.
        """
        import json as _json
        import pickle
        from pathlib import Path as _Path

        graph_path = self._resolve_graph_path(dataset)
        vector_path = self._resolve_vector_path(dataset)
        chunks_path = self._chunks_path(dataset)
        meta_path = self._meta_path(dataset)

        required = [_Path(graph_path), _Path(vector_path + ".faiss"), _Path(chunks_path), _Path(meta_path)]
        if not all(p.exists() for p in required):
            return False

        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = _json.load(f)
            if meta.get("fingerprint") != self._persist_fingerprint(dataset):
                logger_ = __import__("logging").getLogger(__name__)
                logger_.info("persisted graph fingerprint mismatch — will rebuild")
                return False

            self.graph_store.load(graph_path)
            self.vector_store.load(vector_path)
            with open(chunks_path, "rb") as f:
                self.state.chunk_lookup = pickle.load(f)
        except Exception as e:  # corrupted / incompatible artefact -> rebuild
            logger_ = __import__("logging").getLogger(__name__)
            logger_.warning("failed to load persisted graph (%s) — will rebuild", e)
            return False

        # Re-attach the freshly loaded chunk_lookup to the query retriever so
        # retrieval can resolve chunk ids back to text (mirrors the wiring in
        # _rebuild_pipelines_preserving_state).
        retr = self._query.retriever
        if hasattr(retr, "chunk_lookup"):
            retr.chunk_lookup = self.state.chunk_lookup
        return True

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
        elif name == "colla_rag":
            self.config.plugins.colla_rag.enabled = enabled
            if enabled:
                self.config.query.intent_classifier.active = "colla_rag.tree"
                self.config.query.retriever.active = "colla_rag.multi_agent"
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

        if plugins_cfg.colla_rag.enabled:
            try:
                importlib.import_module("chimera_rag.plugins.colla_rag")
            except ImportError as e:  # pragma: no cover - optional
                import logging

                logging.getLogger(__name__).warning("colla_rag plugin not available: %s", e)

    def _build_stores(self):
        from chimera_rag.stores.graph import NetworkXGraphStore
        from chimera_rag.stores.vector import FAISSVectorStore

        graph_cfg = self.config.storage.graph
        if graph_cfg.backend == "networkx":
            graph = NetworkXGraphStore()
        elif graph_cfg.backend == "neo4j":  # pragma: no cover - removed in cleanup
            raise ValueError("neo4j backend is not supported in this version")
        else:  # pragma: no cover - guarded by Pydantic
            raise ValueError(f"unknown graph backend: {graph_cfg.backend}")

        vec_cfg = self.config.storage.vector
        vec = FAISSVectorStore(
            dim=self.config.embedding.dim,
            index_type=vec_cfg.index_type,
            metric=vec_cfg.metric,
            hnsw=vec_cfg.hnsw.model_dump(),
            ivf=vec_cfg.ivf.model_dump(),
        )
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

    def _instantiate_chunker(self, cls: type, params: dict) -> BaseChunker:
        return cls(**self._filter_params(cls, params))

    def _instantiate_extractor(self, cls: type, params: dict) -> BaseTripleExtractor:
        call_params = self._filter_params(cls, params)
        # Strip labels like "default" that refer to the built-in template.
        if call_params.get("prompt_template") == "default":
            call_params.pop("prompt_template", None)
        return cls(llm=self.llm, **call_params)

    def _instantiate_pruner(self, cls: type, params: dict) -> BasePruner:
        return cls(**self._filter_params(cls, params))

    def _instantiate_intent(self, cls: type, params: dict) -> BaseIntentClassifier:
        import inspect

        sig = inspect.signature(cls.__init__)
        call_params = self._filter_params(cls, params)
        # If the class wants an LLM (e.g. LLMIntentClassifier), inject ours.
        if "llm" in sig.parameters and "llm" not in call_params:
            call_params["llm"] = self.llm
        return cls(**call_params)

    def _instantiate_retriever(self, cls: type, params: dict) -> BaseRetriever:
        call_params = self._filter_params(cls, params)
        # top_k is handled by QueryPipeline.
        call_params.pop("top_k", None)

        # Classic retrievers (DirectRetriever) want vector_store + graph_store + embedder
        # + chunk_lookup. Smart retrievers (CollaRAGRetriever) want a base_retriever
        # + classifier + optional hooks. We inspect the ctor and adapt.

        # Multi-agent retriever — wires the full multi-agent orchestrator.
        if cls.__name__ == "MultiAgentRetriever":
            wiring: dict = {
                "llm": self.llm,
                "vector_store": self.vector_store,
                "graph_store": self.graph_store,
                "embedder": self.embedder,
                "chunk_lookup": self.state.chunk_lookup,
                "reranker": self.reranker,
            }
            ma_cfg = self.config.plugins.colla_rag.multi_agent
            if ma_cfg.get("memory", {}).get("shared", {}).get("path"):
                wiring["memory_path"] = ma_cfg["memory"]["shared"]["path"]
            if ma_cfg.get("classifier"):
                wiring["classifier_config"] = ma_cfg["classifier"]
            agent_configs: dict = {}
            _LABEL_MAP = {
                "native_rag": "single_hop", "react": "multi_hop",
                "map_reduce": "summarization", "web_search": "other",
            }
            for key, label in _LABEL_MAP.items():
                if key in ma_cfg:
                    agent_configs[label] = ma_cfg[key]
            wiring["agent_configs"] = agent_configs
            return cls(**wiring)

        # Classic DirectRetriever signature.
        return cls(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
            chunk_lookup=self.state.chunk_lookup,
            **call_params,
        )

    def _instantiate_generator(self, cls: type, params: dict) -> BaseAnswerGenerator:
        call_params = self._filter_params(cls, params)
        if call_params.get("prompt_template") == "default":
            call_params.pop("prompt_template", None)
        return cls(llm=self.llm, **call_params)


__all__ = ["ChimeraRAG", "__version__"]

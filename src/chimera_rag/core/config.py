"""Configuration loader and Pydantic schema for ``config.yaml``.

A single :class:`AppConfig` is the typed representation of the yaml.
The loader :func:`load_config` is deliberately strict:

* missing / unparseable file       -> :class:`ConfigError`
* unknown / missing required field -> :class:`ConfigError` (wraps Pydantic)

Sub-models are intentionally lenient on ``params: dict[str, Any]`` because
plugin authors should be free to define their own parameters without
changing this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from chimera_rag.core.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Top-level building blocks
# ---------------------------------------------------------------------------
class AppMeta(BaseModel):
    name: str
    version: str
    workspace_dir: str


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["rich", "plain", "json"] = "rich"
    file: str | None = None


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    initial_wait: float = Field(default=1.0, ge=0)
    max_wait: float = Field(default=10.0, ge=0)


class LLMConfig(BaseModel):
    # OpenAI-compatible endpoint. All model/connection knobs live here;
    # .env only supplies the API key (via api_key_env, default LLM_API_KEY).
    model: str
    base_url: str | None = None
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 30.0
    retry: RetryConfig = Field(default_factory=RetryConfig)
    # Disable "thinking" / chain-of-thought output for reasoning models that
    # emit it by default (e.g. Qwen3 via ollama). When True the provider
    # routes through ollama's native /api/chat with think:false.
    disable_thinking: bool = False


class EmbeddingConfig(BaseModel):
    # OpenAI-compatible embeddings endpoint. model/connection live here;
    # .env only supplies the API key.
    model: str
    dim: int
    base_url: str | None = None
    api_key_env: str = "LLM_API_KEY"
    batch_size: int = 32
    normalize: bool = True


class RerankConfig(BaseModel):
    """Cross-encoder reranking via an OpenAI-compatible rerank endpoint.

    Optional — set ``enabled: true`` to activate.  The provider calls
    ``POST /rerank`` with ``{"model", "query", "documents", "top_n"}``
    (the de-facto contract used by vLLM / Xinference / SiliconFlow / Jina).
    """
    enabled: bool = False
    model: str = ""
    base_url: str | None = None
    api_key_env: str = "LLM_API_KEY"
    top_n: int = 5
    timeout: float = 30.0
    retry: RetryConfig = Field(default_factory=RetryConfig)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class GraphStoreConfig(BaseModel):
    backend: Literal["networkx", "neo4j"] = "networkx"
    persist_path: str
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password_env: str | None = None


class HNSWParams(BaseModel):
    """HNSW graph index hyper-parameters."""

    M: int = Field(default=32, ge=4, le=128, description="max edges per node")
    ef_construction: int = Field(default=200, ge=8, description="build-time search width")
    ef_search: int = Field(default=64, ge=1, description="query-time search width")


class IVFParams(BaseModel):
    """IVF coarse-quantiser index hyper-parameters."""

    nlist: int = Field(default=100, ge=1, description="number of Voronoi cells")
    nprobe: int = Field(default=8, ge=1, description="cells probed per query")
    train_size: int | None = Field(
        default=None, description="points used for k-means; null = all"
    )


class VectorStoreConfig(BaseModel):
    backend: Literal["faiss"] = "faiss"
    index_type: Literal["flat_ip", "flat_l2", "hnsw", "ivf_flat"] = "hnsw"
    metric: Literal["ip", "l2"] = "ip"
    persist_path: str
    top_k: int = 5
    hnsw: HNSWParams = Field(default_factory=HNSWParams)
    ivf: IVFParams = Field(default_factory=IVFParams)


class StorageConfig(BaseModel):
    graph: GraphStoreConfig
    vector: VectorStoreConfig


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetConfig(BaseModel):
    """Named dataset entry (declared in ``config.datasets``).

    Supports two path-declaration styles:

    * Legacy single path::

          path: ./data/hotpotqa.json

    * Explicit split paths (preferred for HotpotQA-like formats where
      corpus and QA are separate files)::

          corpus_path: ./data/hotpotqa/corpus.json
          qa_path:     ./data/hotpotqa/qa.json
          output_dir:  ./output/hotpotqa

    At least one of ``path`` / ``corpus_path`` / ``qa_path`` must be set.
    """

    loader: str
    path: str | None = None
    corpus_path: str | None = None
    qa_path: str | None = None
    output_dir: str | None = None
    split: str | None = None
    eval_field_mapping: dict[str, str] = Field(default_factory=dict)

    @field_validator("output_dir", mode="after")
    @classmethod
    def _empty_to_none(cls, v: str | None) -> str | None:
        return None if v == "" else v

    def model_post_init(self, _ctx: Any) -> None:
        if not any((self.path, self.corpus_path, self.qa_path)):
            raise ValueError(
                "dataset entry must declare at least one of "
                "'path' / 'corpus_path' / 'qa_path'"
            )

    # ------------------------------------------------------------------
    def output_for(self, config_name: str) -> Path:
        """Return ``<output_dir>/<config_name>.json`` (output_dir must be set)."""
        if not self.output_dir:
            raise ValueError("output_dir is not set for this dataset entry")
        return Path(self.output_dir) / f"{config_name}.json"


# ---------------------------------------------------------------------------
# Slot descriptor (active + params)
# ---------------------------------------------------------------------------
class SlotConfig(BaseModel):
    active: str
    params: dict[str, Any] = Field(default_factory=dict)


class IngestionConfig(BaseModel):
    chunker: SlotConfig
    extractor: SlotConfig
    pruner: SlotConfig


class QueryConfig(BaseModel):
    intent_classifier: SlotConfig
    retriever: SlotConfig
    generator: SlotConfig


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------
class AdaGraphConfig(BaseModel):
    enabled: bool = False
    # The following are free-form so the plugin can grow without breaking core.
    dynamic_chunker: dict[str, Any] = Field(default_factory=dict)
    layered_extractor: dict[str, Any] = Field(default_factory=dict)
    dual_pruner: dict[str, Any] = Field(default_factory=dict)


class CollaRAGConfig(BaseModel):
    enabled: bool = False
    memory: dict[str, Any] = Field(default_factory=dict)
    intent_classifier: dict[str, Any] = Field(default_factory=dict)
    strategies: dict[str, Any] = Field(default_factory=dict)
    hooks: dict[str, Any] = Field(default_factory=dict)
    multi_agent: dict[str, Any] = Field(default_factory=dict)


class PluginsConfig(BaseModel):
    adagraph: AdaGraphConfig = Field(default_factory=AdaGraphConfig)
    colla_rag: CollaRAGConfig = Field(default_factory=CollaRAGConfig)


# ---------------------------------------------------------------------------
# Web / Evaluation
# ---------------------------------------------------------------------------
class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)
    static_dir: str = "src/chimera_rag/web/static"


class EvaluationConfig(BaseModel):
    metrics: list[Literal["em", "f1", "rouge_l", "bertscore"]] = Field(
        default_factory=lambda: ["em", "f1"]
    )
    max_samples: int = 100
    parallel: int = 1
    bertscore_lang: str = "en"


# ---------------------------------------------------------------------------
# AppConfig — root model
# ---------------------------------------------------------------------------
class AppConfig(BaseModel):
    app: AppMeta
    logging: LoggingConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    storage: StorageConfig
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)
    ingestion: IngestionConfig
    query: QueryConfig
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> AppConfig:
    """Load ``path`` as yaml, validate, return :class:`AppConfig`.

    All errors are re-raised as :class:`ConfigError` with actionable messages.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {p}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level of {p} must be a mapping, got {type(raw).__name__}")

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration in {p}:\n{e}") from e


__all__ = [
    "AdaGraphConfig",
    "AppConfig",
    "AppMeta",
    "ConfigError",
    "DatasetConfig",
    "EmbeddingConfig",
    "EvaluationConfig",
    "GraphStoreConfig",
    "HNSWParams",
    "IVFParams",
    "IngestionConfig",
    "CollaRAGConfig",
    "LLMConfig",
    "LoggingConfig",
    "PluginsConfig",
    "QueryConfig",
    "RerankConfig",
    "RetryConfig",
    "SlotConfig",
    "StorageConfig",
    "VectorStoreConfig",
    "WebConfig",
    "load_config",
]

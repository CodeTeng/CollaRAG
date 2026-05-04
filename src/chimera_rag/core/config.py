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
from pydantic import BaseModel, Field, ValidationError

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
    provider: Literal["mock", "openai", "deepseek", "ollama"]
    model: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 30.0
    retry: RetryConfig = Field(default_factory=RetryConfig)


class EmbeddingConfig(BaseModel):
    provider: Literal["mock", "openai", "ollama", "sentence_transformer"]
    model: str
    dim: int
    batch_size: int = 32
    normalize: bool = True


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class GraphStoreConfig(BaseModel):
    backend: Literal["networkx", "neo4j"] = "networkx"
    persist_path: str
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password_env: str | None = None


class VectorStoreConfig(BaseModel):
    backend: Literal["faiss"] = "faiss"
    index_type: Literal["flat_ip", "flat_l2"] = "flat_ip"
    persist_path: str
    top_k: int = 5


class StorageConfig(BaseModel):
    graph: GraphStoreConfig
    vector: VectorStoreConfig


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetConfig(BaseModel):
    loader: str
    path: str
    split: str | None = None
    eval_field_mapping: dict[str, str] = Field(default_factory=dict)


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


class IntentAgentConfig(BaseModel):
    enabled: bool = False
    memory: dict[str, Any] = Field(default_factory=dict)
    intent_classifier: dict[str, Any] = Field(default_factory=dict)
    strategies: dict[str, Any] = Field(default_factory=dict)
    hooks: dict[str, Any] = Field(default_factory=dict)


class PluginsConfig(BaseModel):
    adagraph: AdaGraphConfig = Field(default_factory=AdaGraphConfig)
    intent_agent: IntentAgentConfig = Field(default_factory=IntentAgentConfig)


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
    "DatasetConfig",
    "EmbeddingConfig",
    "EvaluationConfig",
    "GraphStoreConfig",
    "IngestionConfig",
    "IntentAgentConfig",
    "LLMConfig",
    "LoggingConfig",
    "PluginsConfig",
    "QueryConfig",
    "RetryConfig",
    "SlotConfig",
    "StorageConfig",
    "VectorStoreConfig",
    "WebConfig",
    "load_config",
]

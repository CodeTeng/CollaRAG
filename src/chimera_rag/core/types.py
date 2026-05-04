"""Foundational data models shared across Chimera-RAG.

All models are Pydantic v2 ``BaseModel`` with frozen configs where it makes
sense; we prefer explicit validation over silent coercion so that bugs in
chunkers / extractors / retrievers surface early.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Allowed intent labels (keep in sync with IntentAgent plugin).
IntentLabel = Literal[
    "factual",
    "analytical",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
]


# ---------------------------------------------------------------------------
# Corpus: Document / Chunk
# ---------------------------------------------------------------------------
class Document(BaseModel):
    """A raw textual document fed into the ingestion pipeline."""

    doc_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id")
    @classmethod
    def _non_empty_doc_id(cls, v: str) -> str:
        if not v:
            raise ValueError("doc_id must be non-empty")
        return v


class Chunk(BaseModel):
    """A chunk produced by a Chunker and fed to the Extractor."""

    chunk_id: str
    doc_id: str
    index: int = Field(ge=0, description="0-based chunk order within its document")
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knowledge: Triple / Entity
# ---------------------------------------------------------------------------
class Triple(BaseModel):
    """A (subject, predicate, object) fact extracted from a chunk."""

    subject: str
    predicate: str
    object: str
    source_chunk_id: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # AdaGraph layered extractor labels: L1=explicit, L2=implicit, L3=schema, L4=cross-sentence.
    layer: Literal["L1", "L2", "L3", "L4"] = "L1"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A graph node representing a named entity / concept."""

    entity_id: str
    name: str
    entity_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query-side: Query / Intent / ReasoningPlan / RetrievalResult / Answer
# ---------------------------------------------------------------------------
class Query(BaseModel):
    """A user query plus an optional session id for multi-turn memory."""

    text: str
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Intent(BaseModel):
    """Classifier output; label taken from the 6-way IntentLabel union."""

    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class ReasoningPlan(BaseModel):
    """Bridge between Intent and Decision: which strategy to run and how."""

    intent: Intent
    strategy_name: str
    steps: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Output of any retriever / strategy."""

    chunks: list[Chunk] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityFeedback(BaseModel):
    """Output of a Reflector; consumed by the executor to decide next action."""

    quality: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    suggestion: str | None = None


class Answer(BaseModel):
    """Final response returned by the QueryPipeline."""

    text: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    intent_label: IntentLabel | None = None
    strategy_name: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    trace: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Eval: QAExample / EvalResult
# ---------------------------------------------------------------------------
class QAExample(BaseModel):
    """A single QA example loaded from a dataset."""

    qid: str
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Aggregated metrics produced by an Evaluator run."""

    config_name: str
    metrics: dict[str, float]
    per_example: list[dict[str, Any]] = Field(default_factory=list)
    # Optional run-level diagnostics populated by Evaluator.run when available.
    usage: dict[str, int] = Field(default_factory=dict)  # prompt/completion/total tokens + calls
    elapsed_s: float = 0.0  # wall-clock seconds of the evaluation loop
    # Optional one-shot ingest cost (graph building), populated by eval.py
    # when meta.json contains an "ingest" section. None means "not reported".
    ingest: dict[str, Any] | None = None


__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "Entity",
    "EvalResult",
    "Intent",
    "IntentLabel",
    "QAExample",
    "QualityFeedback",
    "Query",
    "ReasoningPlan",
    "RetrievalResult",
    "Triple",
]

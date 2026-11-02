"""Foundational data models shared across Chimera-RAG.

All models are Pydantic v2 ``BaseModel`` with frozen configs where it makes
sense; we prefer explicit validation over silent coercion so that bugs in
chunkers / extractors / retrievers surface early.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Allowed intent labels (keep in sync with CollaRAG plugin).
IntentLabel = Literal[
    "factual",
    "analytical",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
]

MultiAgentIntentLabel = Literal[
    "greeting",
    "single_hop",
    "multi_hop",
    "summarization",
    "other",
]

# PrismRAG (innovation #1) granularity layers. ``unified`` is the default and
# reproduces the behaviour of every non-PrismRAG chunker: the chunk is both
# extracted and indexed. ``fine`` chunks feed the extractor only; ``coarse``
# chunks feed the vector store only. See docs/innovation-1/design.md §4.1.
ChunkGranularity = Literal["unified", "fine", "coarse"]

# Which granularities each ingestion stage consumes.
EXTRACTED_GRANULARITIES: frozenset[str] = frozenset({"unified", "fine"})
INDEXED_GRANULARITIES: frozenset[str] = frozenset({"unified", "coarse"})

# Typed chunk defects (PrismRAG contribution 2). Each code carries exactly one
# repair action, so a detected defect is already a directed instruction —
# the ingestion-side mirror of CollaRAG's typed evidence gaps.
DefectCode = Literal["D1", "D2", "D3", "D4", "D5", "D6"]
RepairAction = Literal["split", "extend_left", "bind", "merge_up", "split_by_topic"]

DEFECT_REPAIR_ACTION: dict[str, RepairAction] = {
    "D1": "split",
    "D2": "extend_left",
    "D3": "bind",
    "D4": "merge_up",
    "D5": "split",
    "D6": "split_by_topic",
}


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
    """A chunk produced by a Chunker and fed to the Extractor.

    ``granularity`` decides which ingestion stages consume the chunk; see
    :data:`EXTRACTED_GRANULARITIES` / :data:`INDEXED_GRANULARITIES`.
    ``parent_chunk_id`` binds a ``fine`` extraction unit to the ``coarse``
    retrieval unit that covers it, so a retrieval hit on the coarse chunk can
    be walked back to the high-purity triples of all its fine children.
    """

    chunk_id: str
    doc_id: str
    index: int = Field(ge=0, description="0-based chunk order within its document")
    text: str
    granularity: ChunkGranularity = "unified"
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _parent_only_on_fine(self) -> Chunk:
        if self.parent_chunk_id is not None and self.granularity != "fine":
            raise ValueError(
                f"parent_chunk_id is only meaningful on granularity='fine'; "
                f"got granularity={self.granularity!r} with "
                f"parent_chunk_id={self.parent_chunk_id!r}"
            )
        return self


class ChunkDefect(BaseModel):
    """A typed chunking defect detected by auditing extraction output.

    PrismRAG contribution 2: chunk quality is formalised as a closed set of
    structurally detectable defect types, needing no human ground truth. Each
    defect carries its own directed repair action, mirroring the typed
    evidence gaps CollaRAG's Reflect stage emits on the query side.
    """

    code: DefectCode
    chunk_id: str
    action: RepairAction
    detail: str = ""
    # Other chunks implicated in the same defect (D3 straddles a pair).
    related_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _action_matches_code(self) -> ChunkDefect:
        expected = DEFECT_REPAIR_ACTION[self.code]
        if self.action != expected:
            raise ValueError(
                f"defect {self.code} must carry action {expected!r}, got {self.action!r}"
            )
        return self

    @property
    def repair_driving(self) -> bool:
        """Whether this defect may trigger a repair, or is record-only.

        Risk R3: a repair loop amplifies detector noise, so a detector that
        could not falsify its own candidate marks it ``confidence="low"`` and
        the pipeline must not act on it. Putting the rule here means every
        consumer honours it without knowing the confidence vocabulary.
        """
        return self.metadata.get("confidence") != "low"


class RepairOutcome(BaseModel):
    """Result of one repair round: a re-segmented chunk list plus bookkeeping.

    ``chunks`` replaces the round's input wholesale — repairs are *boundary
    changes*, so a split or a merge renumbers the layer rather than editing a
    chunk in place. ``defects_deferred`` counts defects that could not be acted
    on (a split on a single-sentence chunk, an ``extend_left`` at the start of a
    document), which is what lets the pipeline stop instead of looping forever.
    """

    chunks: list[Chunk] = Field(default_factory=list)
    defects_applied: int = Field(default=0, ge=0)
    defects_deferred: int = Field(default=0, ge=0)
    actions: dict[str, int] = Field(default_factory=dict)


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
    # Extraction tiers: L1=explicit, L2=implicit, L3=schema, L4=cross-sentence.
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
    """Classifier output; label from IntentLabel or MultiAgentIntentLabel."""

    label: IntentLabel | MultiAgentIntentLabel
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
    """Output of a Reflector; consumed by the executor to decide next action.

    ``reflection_text`` is a verbal (language) self-critique produced by
    LLM-backed reflectors (e.g. :class:`LLMReflectionHook`). When present,
    the reflection loop can inject it into the next planning round so the
    agent avoids repeating the same mistake — the core mechanism behind
    Reflexion-style self-correction (Shinn et al., 2023).
    """

    quality: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    suggestion: str | None = None
    reflection_text: str | None = None


class Answer(BaseModel):
    """Final response returned by the QueryPipeline."""

    text: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    intent_label: IntentLabel | MultiAgentIntentLabel | None = None
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
    # Optional one-shot ingest cost (graph building), populated by experiments/eval.py
    # when meta.json contains an "ingest" section. None means "not reported".
    ingest: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Multi-Agent architecture types (innovation 2 refactor)
# ---------------------------------------------------------------------------


class PreprocessResult(BaseModel):
    """Output of the Preprocessor Agent."""

    rewritten_query: str
    original_query: str
    is_followup: bool = False
    entities_mentioned: list[str] = Field(default_factory=list)
    language: str = "en"


class AgentExperienceEntry(BaseModel):
    """One distilled retrieval experience, bucketed by agent + intent."""

    agent_type: str
    intent: str
    query_text: str
    tool_sequence: list[str] = Field(default_factory=list)
    reflection_count: int = Field(default=0, ge=0)
    final_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome: Literal["success", "lesson"] = "success"
    lesson: str | None = None
    answer_text: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    hit_count: int = Field(default=0, ge=0)
    created_at: str = ""


__all__ = [
    "DEFECT_REPAIR_ACTION",
    "EXTRACTED_GRANULARITIES",
    "INDEXED_GRANULARITIES",
    "AgentExperienceEntry",
    "Answer",
    "Chunk",
    "ChunkDefect",
    "ChunkGranularity",
    "DefectCode",
    "Document",
    "Entity",
    "EvalResult",
    "Intent",
    "IntentLabel",
    "MultiAgentIntentLabel",
    "PreprocessResult",
    "QAExample",
    "QualityFeedback",
    "Query",
    "ReasoningPlan",
    "RepairAction",
    "RepairOutcome",
    "RetrievalResult",
    "Triple",
]

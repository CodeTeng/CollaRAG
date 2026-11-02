"""Typed chunk-defect detectors — PrismRAG contribution 2.

Chunk quality is formalised here as a closed set of defect types that are
**structurally detectable** from the extractor's own output plus the chunk
text and the graph built so far. No human ground truth is needed, which is
what makes the ``extract -> audit -> repair`` loop in Phase 4 affordable.

Every detector shares one signature::

    detect(chunk, triples, ctx) -> list[ChunkDefect]

and emits defects whose ``action`` is already fixed by the code (see
:data:`chimera_rag.core.types.DEFECT_REPAIR_ACTION`), so a detection is a
directed instruction rather than a score. Detectors ignore the parts of
:class:`AuditContext` they do not need.

Two conventions run through all six:

**Granularity gating.** Each detector declares the layers it is meaningful on
via ``audited_granularities``. D2 (antecedent loss) is a property of the
*retrieval* layer only — a ``fine`` chunk is deliberately a fragment whose
coarse parent carries the context, so auditing it would be pure noise. The
four triple-derived detectors run on the *extraction* layer instead. This
gating is what makes the audit projection-aware rather than generic.

**Two-tier confidence (risk R3).** A repair loop amplifies detector noise, so
a detector that cannot falsify its own candidate reports
``confidence="low"`` — recorded for measurement, never repair-driving.
``"high"`` means the candidate survived every falsification available.

==========  =========================================  ==============  =================
Defect      Signal                                     Action          Layer
==========  =========================================  ==============  =================
D1          surface form -> disconnected neighbourhood split           extraction
D2          opening anaphor with no in-chunk antecedent extend_left    retrieval
D3          2-hop path severed at an adjacent boundary bind            extraction
D4          entity present, zero triples extracted     merge_up        extraction
D5          extraction hit the triple budget           split           extraction
D6          two values for one functional predicate    split_by_topic  extraction
==========  =========================================  ==============  =================
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from chimera_rag.core.types import (
    EXTRACTED_GRANULARITIES,
    INDEXED_GRANULARITIES,
    Chunk,
    ChunkDefect,
    Triple,
)
from chimera_rag.interfaces.graph_store import BaseGraphStore
from chimera_rag.plugins.prism.features import find_entity_mentions, split_sentences


@dataclass(frozen=True)
class AuditContext:
    """Everything a detector may consult beyond the chunk it is auditing.

    ``graph`` is the store built so far. It is optional on purpose: D1 needs
    it to *falsify* a collision candidate, but must still run (at reduced
    confidence) when it is absent, so a first-ever ingest is auditable.
    """

    graph: BaseGraphStore | None = None
    chunks: tuple[Chunk, ...] = ()
    triples_by_chunk: Mapping[str, Sequence[Triple]] = field(
        default_factory=lambda: MappingProxyType({})
    )


class DefectDetector(Protocol):
    """The uniform contract every D1–D6 detector satisfies."""

    code: str
    #: Granularity layers this detector is meaningful on; chunks outside it are
    #: skipped, because the defect is a design property of the other layer.
    audited_granularities: frozenset[str]

    def detect(
        self,
        chunk: Chunk,
        triples: Sequence[Triple],
        ctx: AuditContext | None = None,
    ) -> list[ChunkDefect]: ...


# ---------------------------------------------------------------------------
# D5 — budget truncation
# ---------------------------------------------------------------------------
class D5BudgetDetector:
    """Flags a chunk whose extraction hit (or nearly hit) the triple budget.

    ``ratio`` turns the hard cap into an early warning: with ``budget=30`` and
    ``ratio=0.8`` a chunk yielding 24 triples is already reported. The default
    ``1.0`` reproduces the exact cap, i.e. only real truncation fires.
    """

    code = "D5"
    audited_granularities = EXTRACTED_GRANULARITIES

    def __init__(self, *, budget: int = 30, ratio: float = 1.0) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.budget = budget
        self.ratio = ratio
        # Round up: a fractional ratio must never collapse the threshold to 0,
        # which would flag every chunk including empty ones.
        self.threshold = max(1, math.ceil(budget * ratio))

    def detect(
        self,
        chunk: Chunk,
        triples: Sequence[Triple],
        ctx: AuditContext | None = None,
    ) -> list[ChunkDefect]:
        if chunk.granularity not in self.audited_granularities:
            return []
        n = len(triples)
        if n < self.threshold:
            return []
        return [
            ChunkDefect(
                code="D5",
                chunk_id=chunk.chunk_id,
                action="split",
                detail=(
                    f"chunk yielded {n} triples, at or above the {self.threshold} "
                    f"triple budget — the extractor is truncating, so the chunk "
                    f"holds more than one extraction unit"
                ),
                metadata={
                    "triple_count": n,
                    "budget": self.budget,
                    "ratio": self.ratio,
                    "threshold": self.threshold,
                },
            )
        ]


# ---------------------------------------------------------------------------
# D1 — entity collision
# ---------------------------------------------------------------------------
class D1CollisionDetector:
    """Flags a surface form whose in-chunk neighbourhood is disconnected.

    A local-only test is not enough: **every hub entity fails it.** Marie
    Curie's neighbours {Warsaw, Nobel Prize in Physics, Physics} have no edges
    among them within one chunk, yet she is plainly one entity. So a candidate
    is only confirmed when the existing graph cannot merge its components —
    and the merge path is computed with the ambiguous surface form itself
    removed, otherwise every candidate would trivially reconnect at hop 1.

    When no graph is available there is nothing to falsify with, so the
    candidate is reported at ``confidence="low"``. Risk R3 in the design doc:
    detector noise is amplified by a repair loop, so low-confidence defects are
    recorded for measurement and must not drive repairs.
    """

    code = "D1"
    audited_granularities = EXTRACTED_GRANULARITIES

    def __init__(self, *, min_neighbors: int = 3, min_components: int = 2, hops: int = 2) -> None:
        if min_neighbors < 1:
            raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")
        if min_components < 2:
            raise ValueError(f"min_components must be >= 2, got {min_components}")
        if hops < 1:
            raise ValueError(f"hops must be >= 1, got {hops}")
        self.min_neighbors = min_neighbors
        self.min_components = min_components
        self.hops = hops

    def detect(
        self,
        chunk: Chunk,
        triples: Sequence[Triple],
        ctx: AuditContext | None = None,
    ) -> list[ChunkDefect]:
        if chunk.granularity not in self.audited_granularities:
            return []
        ctx = ctx or AuditContext()
        local_edges = _undirected_edges(triples)
        neighbours = _adjacency(local_edges)

        defects: list[ChunkDefect] = []
        for entity in sorted(neighbours):
            hood = neighbours[entity]
            if len(hood) < self.min_neighbors:
                continue

            components = _components(hood, local_edges)
            if len(components) < self.min_components:
                continue

            metadata: dict[str, object] = {
                "entity": entity,
                "neighbor_count": len(hood),
                "components": len(components),
                "component_members": [sorted(c) for c in components],
            }

            if ctx.graph is None:
                metadata["confidence"] = "low"
            else:
                global_edges = _undirected_edges(ctx.graph.get_neighbors(entity, hops=self.hops))
                merged = _components(hood, local_edges | global_edges)
                metadata["global_components"] = len(merged)
                if len(merged) < self.min_components:
                    # Falsified: the neighbourhood is one entity in the graph
                    # we already trust.
                    continue
                metadata["confidence"] = "high"

            defects.append(
                ChunkDefect(
                    code="D1",
                    chunk_id=chunk.chunk_id,
                    action="split",
                    detail=(
                        f"{entity!r} maps to {len(components)} mutually unreachable "
                        f"neighbourhoods inside chunk {chunk.chunk_id}"
                    ),
                    metadata=metadata,
                )
            )
        return defects


# ---------------------------------------------------------------------------
# D2 — antecedent loss
# ---------------------------------------------------------------------------
# Unambiguous personal / possessive anaphora. "it" and "that" are deliberately
# absent: both are overwhelmingly dummy subjects and complementisers, and R3
# says a noisy detector costs more than a missed one.
_PRONOUN_RE = re.compile(r"(?i)\b(he|she|they|him|her|them|his|its|their)\b")
# Demonstratives count only at the head of the opening sentence.
_SENTENCE_HEAD_DEMONSTRATIVE_RE = re.compile(r"(?i)\b(it|this|that|these|those)\b")
# Anaphoric definite descriptions: a generic head noun standing in for an
# entity the chunk never names.
_DEFINITE_NP_RE = re.compile(
    r"(?i)\bthe (former|latter|above|same|company|corporation|city|town|country|"
    r"nation|university|college|team|government|president|author|book|film|"
    r"system|product|study|paper)\b"
)


def _find_anaphor(sentence: str) -> tuple[int, str] | None:
    """``(offset, text)`` of the earliest anaphoric expression in *sentence*."""
    candidates: list[tuple[int, str]] = []
    for regex in (_PRONOUN_RE, _DEFINITE_NP_RE):
        m = regex.search(sentence)
        if m:
            candidates.append((m.start(), m.group(0)))
    lead = len(sentence) - len(sentence.lstrip())
    head = _SENTENCE_HEAD_DEMONSTRATIVE_RE.match(sentence, lead)
    if head:
        candidates.append((head.start(), head.group(0)))
    return min(candidates, key=lambda c: c[0]) if candidates else None


class D2AntecedentDetector:
    """Flags a chunk whose opening sentence has no in-chunk antecedent.

    Audited on the *retrieval* layer only. A ``fine`` chunk is designed to be a
    fragment whose coarse parent supplies the surrounding context, so an
    unresolved pronoun there is the design working, not a defect.

    Confidence splits on cataphora: when the chunk does name an entity but only
    *after* the anaphor, a reader can still resolve it, so the candidate is
    recorded at ``"low"``. When the chunk names nothing at all there is no
    possible antecedent and the defect is ``"high"``.
    """

    code = "D2"
    audited_granularities = INDEXED_GRANULARITIES

    def detect(
        self,
        chunk: Chunk,
        triples: Sequence[Triple],
        ctx: AuditContext | None = None,
    ) -> list[ChunkDefect]:
        if chunk.granularity not in self.audited_granularities:
            return []
        ctx = ctx or AuditContext()

        opening = split_sentences(chunk.text)[0]
        found = _find_anaphor(opening)
        if found is None:
            return []
        anaphor_offset, anaphor = found

        mentions = find_entity_mentions(chunk.text)
        if any(offset < anaphor_offset for offset, _ in mentions):
            # The antecedent is inside the chunk; nothing was lost.
            return []

        return [
            ChunkDefect(
                code="D2",
                chunk_id=chunk.chunk_id,
                action="extend_left",
                detail=(
                    f"chunk opens with {anaphor!r} but no entity precedes it, so the "
                    f"referent lives in an earlier chunk"
                ),
                related_chunk_ids=_preceding_chunk_ids(chunk, ctx.chunks),
                metadata={
                    "anaphor": anaphor,
                    "anaphor_offset": anaphor_offset,
                    "entity_count": len({name for _, name in mentions}),
                    "confidence": "low" if mentions else "high",
                },
            )
        ]


# ---------------------------------------------------------------------------
# D4 — context starvation
# ---------------------------------------------------------------------------
class D4StarvationDetector:
    """Flags a chunk that visibly contains an entity yet yields no triples.

    The repair is ``merge_up``: stop extracting this unit on its own and let the
    coarse parent cover it, which is the *opposite* direction to D1/D5/D6 and
    the reason one segmentation layer cannot satisfy both.

    ``max_zero_rate`` guards against misreading an extractor outage as a
    chunking problem. When most of the audited batch is empty, no boundary
    change can help, so the defect is demoted to ``"low"``. The guard needs at
    least ``min_batch`` chunks to judge — a rate computed from one chunk is not
    evidence.
    """

    code = "D4"
    audited_granularities = EXTRACTED_GRANULARITIES

    def __init__(self, *, max_zero_rate: float = 0.5, min_batch: int = 2) -> None:
        if not 0.0 < max_zero_rate <= 1.0:
            raise ValueError(f"max_zero_rate must be in (0, 1], got {max_zero_rate}")
        if min_batch < 1:
            raise ValueError(f"min_batch must be >= 1, got {min_batch}")
        self.max_zero_rate = max_zero_rate
        self.min_batch = min_batch

    def detect(
        self,
        chunk: Chunk,
        triples: Sequence[Triple],
        ctx: AuditContext | None = None,
    ) -> list[ChunkDefect]:
        if chunk.granularity not in self.audited_granularities:
            return []
        if triples:
            return []

        entities = {name for _, name in find_entity_mentions(chunk.text)}
        if not entities:
            # Zero triples from prose with nothing to extract is not starvation.
            return []

        ctx = ctx or AuditContext()
        batch = ctx.triples_by_chunk
        confidence = "high"
        metadata: dict[str, object] = {
            "entity_count": len(entities),
            "entities": sorted(entities),
        }
        if len(batch) >= self.min_batch:
            zero_rate = sum(1 for ts in batch.values() if not ts) / len(batch)
            metadata["batch_zero_rate"] = zero_rate
            metadata["batch_size"] = len(batch)
            if zero_rate > self.max_zero_rate:
                confidence = "low"
        metadata["confidence"] = confidence

        return [
            ChunkDefect(
                code="D4",
                chunk_id=chunk.chunk_id,
                action="merge_up",
                detail=(
                    f"chunk names {len(entities)} entit"
                    f"{'y' if len(entities) == 1 else 'ies'} but produced no triples"
                ),
                metadata=metadata,
            )
        ]


# ---------------------------------------------------------------------------
# D3 — relation straddling a chunk boundary
# ---------------------------------------------------------------------------
class D3StraddleDetector:
    """Flags a 2-hop path severed between two adjacent extraction units.

    The pattern is ``a -> s`` in one chunk and ``s -> b`` in the next, with no
    ``a -> b`` edge anywhere: neither chunk holds the path, so the path
    projection loses it. Repair is ``bind`` — pull both under one coarse parent
    so the connective tissue survives.

    Two suppressions keep this precise:

    * the pair already shares a ``parent_chunk_id``, so the binding ``bind``
      would create already exists;
    * ``a`` and ``b`` are directly linked inside either chunk, so the path has
      a shortcut and binding adds nothing.

    This is the ingestion-side twin of CollaRAG's ``missing_path`` typed gap,
    which is what makes the cross-innovation experiment E5 measurable.
    """

    code = "D3"
    audited_granularities = EXTRACTED_GRANULARITIES

    def detect(
        self,
        chunk: Chunk,
        triples: Sequence[Triple],
        ctx: AuditContext | None = None,
    ) -> list[ChunkDefect]:
        if chunk.granularity not in self.audited_granularities or not triples:
            return []
        ctx = ctx or AuditContext()

        sibling = _following_chunk(chunk, ctx.chunks)
        if sibling is None or sibling.granularity not in self.audited_granularities:
            return []
        if (
            chunk.parent_chunk_id is not None
            and chunk.parent_chunk_id == sibling.parent_chunk_id
        ):
            return []

        sibling_triples = list(ctx.triples_by_chunk.get(sibling.chunk_id, ()))
        if not sibling_triples:
            return []

        here = _incidence(triples)
        there = _incidence(sibling_triples)
        joined = _undirected_edges([*triples, *sibling_triples])

        bridges: list[str] = []
        directed = False
        for entity in sorted(set(here) & set(there)):
            near = (here[entity]["in"] | here[entity]["out"]) - {entity}
            far = (there[entity]["in"] | there[entity]["out"]) - {entity}
            severs = any(
                frozenset((a, b)) not in joined for a in near for b in far if a != b
            )
            if not severs:
                continue
            bridges.append(entity)
            flows_in_then_out = bool(here[entity]["in"]) and bool(there[entity]["out"])
            flows_out_then_in = bool(here[entity]["out"]) and bool(there[entity]["in"])
            directed = directed or flows_in_then_out or flows_out_then_in

        if not bridges:
            return []

        return [
            ChunkDefect(
                code="D3",
                chunk_id=chunk.chunk_id,
                action="bind",
                detail=(
                    f"path through {', '.join(repr(b) for b in bridges)} is severed "
                    f"between {chunk.chunk_id} and {sibling.chunk_id}"
                ),
                related_chunk_ids=[sibling.chunk_id],
                metadata={
                    "bridges": bridges,
                    "sibling_chunk_id": sibling.chunk_id,
                    "directed": directed,
                    # A directed in->out flow is a real path being cut; two
                    # same-direction halves are weaker evidence, so record only.
                    "confidence": "high" if directed else "low",
                },
            )
        ]


# ---------------------------------------------------------------------------
# D6 — intra-chunk contradiction
# ---------------------------------------------------------------------------
#: Predicates known to be single-valued. A conflict here is reported at
#: ``"high"`` confidence; everything else has to go through the name heuristic.
DEFAULT_EXCLUSIVE_PREDICATES: frozenset[str] = frozenset(
    {
        "birth_date",
        "birth_year",
        "born_in",
        "born_on",
        "capital_of",
        "death_date",
        "death_year",
        "died_in",
        "died_on",
        "founded_year",
        "headquartered_in",
        "nationality",
    }
)

# Deliberately narrower than the explicit set's vocabulary: bare "founded"
# would sweep in "founded_by", which is legitimately multi-valued.
_FUNCTIONAL_PREDICATE_RE = re.compile(
    r"(?i)born|died|birth|death|nationality|headquarter|capital_of|"
    r"founded_(?:in|on|year)|founding_date|located_in|ceo_of|president_of|"
    r"isbn|population"
)


def _distinct_values(objects: Sequence[str]) -> list[str]:
    """Deduplicate object surface forms, folding containment variants.

    ``"Warsaw"`` and ``"Warsaw, Poland"`` are one value written two ways, not a
    contradiction; treating them as two would make D6 fire on normalisation
    noise alone.
    """
    kept: list[str] = []
    for obj in sorted(set(objects)):
        low = obj.strip().lower()
        if not low:
            continue
        if any(low in k.strip().lower() or k.strip().lower() in low for k in kept):
            continue
        kept.append(obj)
    return kept


class D6ContradictionDetector:
    """Flags one subject carrying two values for a functional predicate.

    Two topics got packed into one extraction unit, which corrupts the
    attribute projection. Repair is ``split_by_topic``.

    Without an ontology the detector cannot know which predicates are
    single-valued, so it is two-tiered: ``exclusive_predicates`` is curated and
    reports ``"high"``, while :data:`_FUNCTIONAL_PREDICATE_RE` guesses from the
    predicate's name and reports ``"low"``. ``enable_heuristic=False`` gives
    experiment E3 a precision-only mode.
    """

    code = "D6"
    audited_granularities = EXTRACTED_GRANULARITIES

    def __init__(
        self,
        *,
        exclusive_predicates: frozenset[str] | set[str] | None = None,
        enable_heuristic: bool = True,
    ) -> None:
        predicates = (
            DEFAULT_EXCLUSIVE_PREDICATES if exclusive_predicates is None else exclusive_predicates
        )
        if not predicates:
            raise ValueError("exclusive_predicates must not be empty")
        self.exclusive_predicates = frozenset(predicates)
        self.enable_heuristic = enable_heuristic

    def detect(
        self,
        chunk: Chunk,
        triples: Sequence[Triple],
        ctx: AuditContext | None = None,
    ) -> list[ChunkDefect]:
        if chunk.granularity not in self.audited_granularities:
            return []

        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for t in triples:
            grouped[(t.subject, t.predicate)].append(t.object)

        defects: list[ChunkDefect] = []
        for (subject, predicate), objects in sorted(grouped.items()):
            values = _distinct_values(objects)
            if len(values) < 2:
                continue

            if predicate in self.exclusive_predicates:
                source, confidence = "explicit", "high"
            elif self.enable_heuristic and _FUNCTIONAL_PREDICATE_RE.search(predicate):
                source, confidence = "heuristic", "low"
            else:
                # Legitimately multi-valued ("won" two prizes) — not a defect.
                continue

            defects.append(
                ChunkDefect(
                    code="D6",
                    chunk_id=chunk.chunk_id,
                    action="split_by_topic",
                    detail=(
                        f"{subject!r} carries {len(values)} conflicting values for "
                        f"{predicate!r} inside one chunk: {values!r}"
                    ),
                    metadata={
                        "subject": subject,
                        "predicate": predicate,
                        "objects": values,
                        "source": source,
                        "confidence": confidence,
                    },
                )
            )
        return defects


# ---------------------------------------------------------------------------
# Graph helpers (module-private)
# ---------------------------------------------------------------------------
def _undirected_edges(triples: Sequence[Triple]) -> set[frozenset[str]]:
    """Endpoint pairs as frozensets, dropping self-loops.

    A self-loop ``(X, same_as, X)`` would otherwise count X as its own
    neighbour and inflate every fan-out measurement.
    """
    edges: set[frozenset[str]] = set()
    for t in triples:
        if t.subject == t.object:
            continue
        edges.add(frozenset((t.subject, t.object)))
    return edges


def _adjacency(edges: set[frozenset[str]]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        u, v = tuple(edge)
        adj[u].add(v)
        adj[v].add(u)
    return dict(adj)


def _components(nodes: set[str], edges: set[frozenset[str]]) -> list[set[str]]:
    """Connected components of the subgraph *induced* on ``nodes``.

    Edges touching a node outside ``nodes`` are dropped by construction, which
    is how D1 keeps the confirmation path from running through the ambiguous
    entity itself.
    """
    induced: dict[str, set[str]] = {n: set() for n in nodes}
    for edge in edges:
        u, v = tuple(edge)
        if u in induced and v in induced:
            induced[u].add(v)
            induced[v].add(u)

    seen: set[str] = set()
    out: list[set[str]] = []
    for start in sorted(nodes):
        if start in seen:
            continue
        comp: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(induced[node] - comp)
        seen |= comp
        out.append(comp)
    return out


def _incidence(triples: Sequence[Triple]) -> dict[str, dict[str, set[str]]]:
    """``entity -> {"in": sources, "out": targets}`` for one triple list.

    Keeping the two directions apart is what lets D3 tell a genuinely severed
    ``a -> s -> b`` path from two same-direction halves that merely share a
    subject.
    """
    inc: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"in": set(), "out": set()})
    for t in triples:
        if t.subject == t.object:
            continue
        inc[t.subject]["out"].add(t.object)
        inc[t.object]["in"].add(t.subject)
    return inc


def _preceding_chunk_ids(chunk: Chunk, chunks: Sequence[Chunk]) -> list[str]:
    """The chunk ``extend_left`` would grow into, or ``[]`` at a document start.

    Scoped to the same granularity layer: under dual granularity the fine and
    coarse layers each number their chunks from 0, so ``index + 1`` alone would
    reach into the other layer.
    """
    for c in chunks:
        if (
            c.doc_id == chunk.doc_id
            and c.granularity == chunk.granularity
            and c.index == chunk.index - 1
        ):
            return [c.chunk_id]
    return []


def _following_chunk(chunk: Chunk, chunks: Sequence[Chunk]) -> Chunk | None:
    """The next chunk in the same document *and layer*, or ``None``."""
    for c in chunks:
        if (
            c.doc_id == chunk.doc_id
            and c.granularity == chunk.granularity
            and c.index == chunk.index + 1
        ):
            return c
    return None


__all__ = [
    "DEFAULT_EXCLUSIVE_PREDICATES",
    "AuditContext",
    "D1CollisionDetector",
    "D2AntecedentDetector",
    "D3StraddleDetector",
    "D4StarvationDetector",
    "D5BudgetDetector",
    "D6ContradictionDetector",
    "DefectDetector",
]

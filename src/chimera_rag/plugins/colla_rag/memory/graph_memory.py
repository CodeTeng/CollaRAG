"""Graph-Grounded Dual-Layer Memory (Contribution 3, graph-grounded).

The original dual-layer memory stores (a) a text-hash QA cache and (b) free-form
tool sequences + text lessons. Two problems follow (reviewer M7): on QA
benchmarks that do not repeat questions the text-hash cache hit rate is
suspicious and unexplained, and the "learning" contribution of private memory
is conflated with cache hits.

This module re-grounds both layers in the knowledge graph:

* :class:`EvidenceSubgraphStore` (shared layer, mechanism 3B) caches
  ``(entities, evidence triples, answer)`` and retrieves by
  **entity-neighborhood overlap**, not text hash. Overlap reuse is legitimate
  on non-repeating benchmarks because different questions about the same topic
  recur over the *same entity neighborhood* — so a hit reflects structural
  reuse, not question duplication.
* :class:`PlanTemplateStore` (private layer, mechanism 3A) stores successful
  graph plan templates keyed by compositionality + entity signature, so a new
  query retrieves a **plan prior** matched on structure rather than text.
* :class:`GapLessonStore` (private layer, mechanism 3C) stores the typed
  structural gaps emitted by G-PER's Reflect phase (``missing_edge`` etc.) as
  pre-emptive retrieval lessons, so future similar queries fetch the missing
  edge *before* reasoning begins. This composes Contribution 3 with the
  Reflect output of Contribution 2.

Every store degrades gracefully: with no entities or no graph evidence they
return empty, so the system falls back to the text-hash cache and text
lessons.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _serialize_triples(triples: list[Any]) -> list[dict]:
    out: list[dict] = []
    for t in triples:
        if isinstance(t, dict):
            out.append({
                "subject": t.get("subject", ""),
                "predicate": t.get("predicate", ""),
                "object": t.get("object", ""),
            })
        else:
            out.append({
                "subject": getattr(t, "subject", ""),
                "predicate": getattr(t, "predicate", ""),
                "object": getattr(t, "object", ""),
            })
    return out


def _entity_set(triples: list[dict]) -> set[str]:
    s: set[str] = set()
    for t in triples:
        if t.get("subject"):
            s.add(t["subject"])
        if t.get("object"):
            s.add(t["object"])
    return s


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class EvidenceSubgraph:
    """A cached (evidence subgraph, answer) entry, keyed by its entity set."""

    entities: list[str]
    triples: list[dict]
    answer: str
    quality: float = 0.0
    query_text: str = ""


class EvidenceSubgraphStore:
    """Shared cache retrieved by entity-neighborhood overlap (mechanism 3B)."""

    def __init__(self, path: str, min_overlap: float = 0.5) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_overlap = min_overlap
        self._entries: list[dict] = []
        self.load()

    def store(
        self,
        query_text: str,
        entities: list[str],
        triples: list[Any],
        answer: str,
        quality: float = 0.0,
    ) -> None:
        if not entities and not triples:
            return  # nothing structural to cache
        ser = _serialize_triples(triples)
        ent_set = set(entities) | _entity_set(ser)
        if not ent_set:
            return
        entry = {
            "entities": sorted(ent_set),
            "triples": ser,
            "answer": answer,
            "quality": round(float(quality), 4),
            "query_text": query_text,
        }
        # Dedup on the entity signature; keep the highest-quality answer.
        sig = tuple(entry["entities"])
        for existing in self._entries:
            if tuple(existing["entities"]) == sig:
                if entry["quality"] > existing.get("quality", 0.0):
                    existing.update(entry)
                return
        self._entries.append(entry)

    def lookup(self, entities: list[str]) -> EvidenceSubgraph | None:
        """Return the best overlapping cached entry, or None below threshold."""
        if not entities:
            return None
        q = set(entities)
        best: tuple[float, dict] = (0.0, {})
        for e in self._entries:
            ov = _jaccard(q, set(e.get("entities", [])))
            if ov > best[0]:
                best = (ov, e)
        if best[0] >= self.min_overlap and best[1]:
            d = best[1]
            return EvidenceSubgraph(
                entities=d.get("entities", []),
                triples=d.get("triples", []),
                answer=d.get("answer", ""),
                quality=d.get("quality", 0.0),
                query_text=d.get("query_text", ""),
            )
        return None

    def stats(self) -> dict[str, int]:
        return {"evidence_subgraphs": len(self._entries)}

    def persist(self) -> None:
        self.path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._entries = data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("failed to load %s: %s", self.path, e)


class PlanTemplateStore:
    """Private per-compositionality graph plan templates (mechanism 3A)."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, list[dict]] = {}
        self.load()

    @staticmethod
    def _sig(compositionality: str, entities: list[str]) -> str:
        ent = "|".join(sorted(entities[:4]))  # cap for a coarse signature
        return f"{compositionality}::{ent}"

    def recall(
        self, compositionality: str, entities: list[str],
    ) -> list[dict] | None:
        key = self._sig(compositionality, entities)
        entries = self._store.get(key)
        if not entries:
            # fall back to compositionality-only bucket
            entries = self._store.get(f"{compositionality}::")
        if not entries:
            return None
        ranked = sorted(
            entries,
            key=lambda e: (e.get("quality", 0), e.get("hit_count", 0)),
            reverse=True,
        )
        return ranked[0].get("plan")

    def store(
        self, compositionality: str, entities: list[str],
        plan: list[dict], quality: float = 0.0,
    ) -> None:
        if not compositionality or not plan:
            return
        key = self._sig(compositionality, entities)
        bucket = self._store.setdefault(key, [])
        for existing in bucket:
            if existing.get("plan") == plan:
                existing["hit_count"] = existing.get("hit_count", 0) + 1
                existing["quality"] = max(
                    existing.get("quality", 0), float(quality),
                )
                return
        bucket.append({
            "plan": plan, "quality": round(float(quality), 4), "hit_count": 0,
        })

    def stats(self) -> dict[str, int]:
        return {"plan_templates": sum(len(v) for v in self._store.values())}

    def persist(self) -> None:
        self.path.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._store = data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("failed to load %s: %s", self.path, e)


@dataclass
class GapLesson:
    """A typed structural gap from G-PER Reflect, stored as a lesson (3C)."""

    kind: str
    detail: str
    repair_query: str
    repair_entities: list[str] = field(default_factory=list)


class GapLessonStore:
    """Private structural-gap lessons for pre-emptive retrieval (mechanism 3C)."""

    def __init__(self, path: str, max_entries: int = 200) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._entries: list[dict] = []
        self.load()

    def store(self, entities: list[str], gaps: list[dict]) -> None:
        if not gaps:
            return
        ent = sorted({e for e in entities if e})
        for g in gaps:
            ge = sorted(set(g.get("repair_entities") or []) | set(ent))
            entry = {
                "kind": g.get("kind", "missing_relation"),
                "detail": g.get("detail", ""),
                "repair_query": g.get("repair_query", ""),
                "entities": ge,
            }
            # Dedup on (kind, repair_query); bump hit count.
            for existing in self._entries:
                if (existing.get("kind") == entry["kind"]
                        and existing.get("repair_query") == entry["repair_query"]):
                    existing["entities"] = sorted(
                        set(existing.get("entities", [])) | set(ge),
                    )
                    existing["hit_count"] = existing.get("hit_count", 0) + 1
                    break
            else:
                entry["hit_count"] = 0
                self._entries.append(entry)
        # Cold eviction: drop least-reused when over capacity.
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda e: e.get("hit_count", 0))
            self._entries = self._entries[-self.max_entries:]

    def recall(self, entities: list[str], min_overlap: float = 0.34) -> list[GapLesson]:
        """Return gap lessons whose entity set overlaps the query's."""
        if not entities or not self._entries:
            return []
        q = set(entities)
        hits: list[tuple[float, dict]] = []
        for e in self._entries:
            ov = _jaccard(q, set(e.get("entities", [])))
            if ov >= min_overlap:
                hits.append((ov, e))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [
            GapLesson(
                kind=h["kind"], detail=h.get("detail", ""),
                repair_query=h.get("repair_query", ""),
                repair_entities=h.get("entities", []),
            )
            for _, h in hits[:5]
        ]

    def stats(self) -> dict[str, int]:
        return {"gap_lessons": len(self._entries)}

    def persist(self) -> None:
        self.path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._entries = data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("failed to load %s: %s", self.path, e)


__all__ = [
    "EvidenceSubgraph",
    "EvidenceSubgraphStore",
    "GapLesson",
    "GapLessonStore",
    "PlanTemplateStore",
]

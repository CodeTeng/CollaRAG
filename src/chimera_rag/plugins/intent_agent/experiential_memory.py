"""FileExperientialMemory — innovation #2c's disk-backed experience store.

Claude-Code-style persistent memory: distilled retrieval experiences are
bucketed by intent and appended to per-intent JSONL files under a
configurable directory (default ``.chimera_memory/experience/``). A later
query — even across sessions / process restarts — recalls same-intent
experiences to seed the LangGraph agent's decision (see
``docs/innovation_2c_experiential_memory.md``).

Design notes:

* **Bucketed by intent** so recall reads exactly one small file (O(bucket))
  rather than scanning the whole store, aligning with the intent-as-index-key
  design and reusing the existing classifier output.
* **JSONL append** keeps writes cheap and crash-resilient (a corrupt tail
  line loses at most one entry, which we skip on load).
* **Bounded** via ``max_per_bucket`` with cold-and-old eviction so the store
  never grows unbounded.
* **De-duplicated**: an incoming entry with the same (tool_sequence, outcome)
  in its bucket updates the existing one (sliding quality + hit_count) instead
  of appending a duplicate.

Registered as ``('memory', 'intent_agent.experiential')``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from chimera_rag.core.registry import register
from chimera_rag.interfaces.experiential_memory import (
    BaseExperientialMemory,
    ExperienceEntry,
)
from chimera_rag.plugins.intent_agent.intent_classifier import CANONICAL_LABELS

logger = logging.getLogger(__name__)

_DEFAULT_DIR = ".chimera_memory"
_EXPERIENCE_SUBDIR = "experience"


@register("memory", "intent_agent.experiential")
class FileExperientialMemory(BaseExperientialMemory):
    """Disk-backed, intent-bucketed experience store."""

    def __init__(
        self,
        path: str = _DEFAULT_DIR,
        max_per_bucket: int = 200,
    ) -> None:
        if max_per_bucket <= 0:
            raise ValueError("max_per_bucket must be positive")
        self.root = Path(path)
        self.experience_dir = self.root / _EXPERIENCE_SUBDIR
        self.max_per_bucket = max_per_bucket
        self.experience_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _bucket_path(self, intent: str) -> Path:
        # Unknown intents fall into a shared 'factual' bucket (safest default,
        # matching the classifier's fallthrough label).
        safe_intent = intent if intent in CANONICAL_LABELS else "factual"
        return self.experience_dir / f"{safe_intent}.jsonl"

    # ------------------------------------------------------------------
    # Load / persist a single bucket
    # ------------------------------------------------------------------
    def _load_bucket(self, intent: str) -> list[ExperienceEntry]:
        path = self._bucket_path(intent)
        if not path.exists():
            return []
        entries: list[ExperienceEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(ExperienceEntry.model_validate_json(line))
            except Exception as e:  # corrupt / partial line -> skip, don't crash
                logger.debug("experiential_memory: skipping bad line in %s: %s", path, e)
        return entries

    def _write_bucket(self, intent: str, entries: list[ExperienceEntry]) -> None:
        path = self._bucket_path(intent)
        payload = "\n".join(e.model_dump_json() for e in entries)
        # Trailing newline keeps the file POSIX-friendly and append-ready.
        path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")

    # ------------------------------------------------------------------
    # Eviction: keep the hottest, freshest entries when over capacity.
    # ------------------------------------------------------------------
    def _evict_if_needed(self, entries: list[ExperienceEntry]) -> list[ExperienceEntry]:
        if len(entries) <= self.max_per_bucket:
            return entries
        # Cold-and-old first: sort by (hit_count, created_at) ascending and drop
        # the front; keep the tail (hot + recent).
        ordered = sorted(entries, key=lambda e: (e.hit_count, e.created_at))
        return ordered[len(entries) - self.max_per_bucket :]

    # ------------------------------------------------------------------
    # BaseExperientialMemory interface
    # ------------------------------------------------------------------
    def remember(self, entry: ExperienceEntry) -> None:
        entries = self._load_bucket(entry.intent)

        # De-dup: same tool path + same outcome -> merge instead of append.
        dedup_key = (tuple(entry.tool_sequence), entry.outcome)
        merged = False
        for existing in entries:
            if (tuple(existing.tool_sequence), existing.outcome) == dedup_key:
                # Sliding update of quality (favour the better run) + hit bump.
                existing.final_quality = max(existing.final_quality, entry.final_quality)
                existing.hit_count += 1
                if entry.answer_text and not existing.answer_text:
                    existing.answer_text = entry.answer_text
                    existing.evidence_chunk_ids = entry.evidence_chunk_ids
                if entry.lesson and not existing.lesson:
                    existing.lesson = entry.lesson
                merged = True
                break

        if not merged:
            entries.append(entry)

        entries = self._evict_if_needed(entries)
        self._write_bucket(entry.intent, entries)

    def recall(
        self, intent: str, query_text: str, top_n: int = 3
    ) -> list[ExperienceEntry]:
        if top_n <= 0:
            return []
        entries = self._load_bucket(intent)
        if not entries:
            return []
        # Rank by quality, then by reuse heat — surface proven, popular paths.
        ranked = sorted(
            entries,
            key=lambda e: (e.final_quality, e.hit_count),
            reverse=True,
        )
        return ranked[:top_n]

    def stats(self) -> dict:
        buckets: dict[str, int] = {}
        total = 0
        for label in sorted(CANONICAL_LABELS):
            count = len(self._load_bucket(label))
            if count:
                buckets[label] = count
                total += count
        return {"total": total, "buckets": buckets}


__all__ = ["FileExperientialMemory"]

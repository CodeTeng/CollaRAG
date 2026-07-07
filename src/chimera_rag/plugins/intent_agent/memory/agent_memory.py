"""AgentPrivateMemory — per-agent, per-intent JSONL experience store."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentPrivateMemory:
    def __init__(self, path: str, max_per_bucket: int = 200) -> None:
        self.root = Path(path)
        self.max_per_bucket = max_per_bucket

    def _bucket_path(self, agent_type: str, intent: str) -> Path:
        d = self.root / agent_type / "experience"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{intent}.jsonl"

    def _load_bucket(self, agent_type: str, intent: str) -> list[dict]:
        path = self._bucket_path(agent_type, intent)
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("skipping bad line in %s", path)
        return entries

    def _write_bucket(self, agent_type: str, intent: str, entries: list[dict]) -> None:
        path = self._bucket_path(agent_type, intent)
        payload = "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
        path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")

    def recall(self, agent_type: str, intent: str, top_n: int = 3) -> list[dict]:
        entries = self._load_bucket(agent_type, intent)
        ranked = sorted(
            entries,
            key=lambda e: (e.get("final_quality", 0), e.get("hit_count", 0)),
            reverse=True,
        )
        return ranked[:top_n]

    def remember(self, agent_type: str, entry: dict) -> None:
        intent = entry.get("intent", "unknown")
        entries = self._load_bucket(agent_type, intent)
        dedup_key = (tuple(entry.get("tool_sequence", [])), entry.get("outcome", "success"))
        merged = False
        for existing in entries:
            existing_key = (
                tuple(existing.get("tool_sequence", [])),
                existing.get("outcome", "success"),
            )
            if existing_key == dedup_key:
                existing["final_quality"] = max(
                    existing.get("final_quality", 0), entry.get("final_quality", 0)
                )
                existing["hit_count"] = existing.get("hit_count", 0) + 1
                merged = True
                break
        if not merged:
            entry.setdefault("hit_count", 0)
            entries.append(entry)
        if len(entries) > self.max_per_bucket:
            entries.sort(key=lambda e: (e.get("hit_count", 0), e.get("final_quality", 0)))
            entries = entries[len(entries) - self.max_per_bucket :]
        self._write_bucket(agent_type, intent, entries)

    def stats(self) -> dict:
        total = 0
        buckets: dict[str, int] = {}
        if self.root.exists():
            for agent_dir in self.root.iterdir():
                if agent_dir.is_dir() and (agent_dir / "experience").exists():
                    for f in (agent_dir / "experience").glob("*.jsonl"):
                        count = sum(1 for line in f.read_text().splitlines() if line.strip())
                        key = f"{agent_dir.name}/{f.stem}"
                        buckets[key] = count
                        total += count
        return {"total": total, "buckets": buckets}


__all__ = ["AgentPrivateMemory"]

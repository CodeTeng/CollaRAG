"""SharedMemory — JSON-backed global memory shared across all Agents."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SharedMemory:
    _TYPES = ("routing_stats", "qa_cache", "entity_aliases", "rewrite_templates")

    def __init__(self, path: str) -> None:
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict[str, Any]] = {t: {} for t in self._TYPES}
        self.load()

    def read(self, memory_type: str, key: str) -> Any | None:
        return self._store.get(memory_type, {}).get(key)

    def write(self, memory_type: str, key: str, value: Any) -> None:
        self._store.setdefault(memory_type, {})[key] = value

    def get_qa_cache(self, query_hash: str) -> dict | None:
        return self.read("qa_cache", query_hash)

    def persist(self) -> None:
        for mem_type, data in self._store.items():
            path = self.root / f"{mem_type}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        for mem_type in self._TYPES:
            path = self.root / f"{mem_type}.json"
            if path.exists():
                try:
                    self._store[mem_type] = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("failed to load %s: %s", path, e)

    def stats(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._store.items()}


__all__ = ["SharedMemory"]

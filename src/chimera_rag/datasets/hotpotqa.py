"""HotpotQA dataset loader.

Native format: JSON array where each entry has::

    {"_id": str, "question": str, "answer": str,
     "context": [[title, [sentence, sentence, ...]], ...],
     "type": "bridge" | "comparison", "level": "easy" | "medium" | "hard"}

We emit one :class:`Document` per unique context title (sentences joined)
and one :class:`QAExample` per entry with ``contexts`` populated from the
flattened sentences.
"""

from __future__ import annotations

import json
from pathlib import Path

from chimera_rag.core.exceptions import DatasetError
from chimera_rag.core.types import Document, QAExample
from chimera_rag.datasets.base import BaseDatasetLoader


class HotpotQALoader(BaseDatasetLoader):
    def __init__(self, path: str, **_ignored) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    def _read(self) -> list[dict]:
        if not self.path.is_file():
            raise DatasetError(f"file not found: {self.path}")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise DatasetError(f"invalid JSON in {self.path}: {e}") from e
        if not isinstance(data, list):
            raise DatasetError(f"expected a JSON array in {self.path}")
        return data

    # ------------------------------------------------------------------
    def load_examples(self, limit: int | None = None) -> list[QAExample]:
        raw = self._read()
        out: list[QAExample] = []
        for item in raw:
            if limit is not None and len(out) >= limit:
                break
            ctx_sentences: list[str] = []
            for ctx in item.get("context", []):
                if len(ctx) >= 2 and isinstance(ctx[1], list):
                    ctx_sentences.extend(s for s in ctx[1] if isinstance(s, str))
            out.append(
                QAExample(
                    qid=str(item.get("_id", f"hpq-{len(out)}")),
                    question=str(item["question"]),
                    answer=str(item["answer"]),
                    contexts=ctx_sentences,
                    metadata={
                        "type": item.get("type"),
                        "level": item.get("level"),
                    },
                )
            )
        return out

    # ------------------------------------------------------------------
    def load_documents(self, limit: int | None = None) -> list[Document]:
        raw = self._read()
        if limit is not None:
            raw = raw[:limit]
        by_title: dict[str, list[str]] = {}
        for item in raw:
            for ctx in item.get("context", []):
                if len(ctx) >= 2 and isinstance(ctx[1], list):
                    title = str(ctx[0])
                    by_title.setdefault(title, []).extend(
                        s for s in ctx[1] if isinstance(s, str)
                    )

        return [
            Document(
                doc_id=title,
                content=" ".join(sents).strip(),
                metadata={"source": "hotpotqa"},
            )
            for title, sents in by_title.items()
            if sents
        ]


__all__ = ["HotpotQALoader"]

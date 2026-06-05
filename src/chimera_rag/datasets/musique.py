"""MusiqueQA dataset loader (JSONL format).

Each line is a JSON object with at minimum ``id``, ``question``,
``answer`` keys. ``paragraphs`` (if present) is exposed to ingestion
as one Document per paragraph.
"""

from __future__ import annotations

import json
from pathlib import Path

from chimera_rag.core.exceptions import DatasetError
from chimera_rag.core.types import Document, QAExample
from chimera_rag.datasets.base import BaseDatasetLoader


class MusiqueLoader(BaseDatasetLoader):
    def __init__(self, path: str, **_ignored) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    def _iter_rows(self):
        if not self.path.is_file():
            raise DatasetError(f"file not found: {self.path}")
        with self.path.open(encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetError(
                        f"invalid JSON at {self.path}:{i}: {e}"
                    ) from e

    # ------------------------------------------------------------------
    def load_examples(self, limit: int | None = None) -> list[QAExample]:
        out: list[QAExample] = []
        for row in self._iter_rows():
            if limit is not None and len(out) >= limit:
                break
            out.append(
                QAExample(
                    qid=str(row.get("id", f"musique-{len(out)}")),
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    contexts=[
                        str(p.get("paragraph_text", ""))
                        for p in row.get("paragraphs", [])
                    ],
                )
            )
        return out

    # ------------------------------------------------------------------
    def load_documents(self, limit: int | None = None) -> list[Document]:
        docs: list[Document] = []
        n_rows = 0
        for row in self._iter_rows():
            if limit is not None and n_rows >= limit:
                break
            n_rows += 1
            rid = str(row.get("id", f"musique-{len(docs)}"))
            for i, para in enumerate(row.get("paragraphs", [])):
                title = str(para.get("title", ""))
                text = str(para.get("paragraph_text", ""))
                if not text.strip():
                    continue
                docs.append(
                    Document(
                        doc_id=f"{rid}::{i}" + (f"::{title}" if title else ""),
                        content=text,
                        metadata={"source": "musique", "title": title},
                    )
                )
        return docs


__all__ = ["MusiqueLoader"]

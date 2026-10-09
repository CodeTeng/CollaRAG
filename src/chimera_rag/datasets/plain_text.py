"""Plain-text corpus loader.

Treats the file as one document per *paragraph* (blank-line-separated).
Useful for the packaged examples/data/sample.txt demo.
"""

from __future__ import annotations

from pathlib import Path

from chimera_rag.core.exceptions import DatasetError
from chimera_rag.core.types import Document
from chimera_rag.datasets.base import BaseDatasetLoader


class PlainTextLoader(BaseDatasetLoader):
    """Split ``path`` on blank lines and emit one Document per paragraph."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load_documents(self, limit: int | None = None) -> list[Document]:
        if not self.path.is_file():
            raise DatasetError(f"file not found: {self.path}")
        text = self.path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        stem = self.path.stem
        return [
            Document(doc_id=f"{stem}::{i}", content=p, metadata={"source": str(self.path)})
            for i, p in enumerate(paragraphs)
        ]


__all__ = ["PlainTextLoader"]

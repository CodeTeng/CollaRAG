"""Combined corpus + QA loader for the synthetic wiki dataset.

Accepts either

* a directory ``path`` with default filenames inside::

    MockWikiLoader(path="data/mock_wiki")
    # expects data/mock_wiki/{corpus.txt, qa.jsonl}

* or explicit paths (preferred when ingestion and evaluation files
  don't sit next to each other)::

    MockWikiLoader(
        corpus_path="data/mock_wiki/corpus.txt",
        qa_path="data/mock_wiki/qa.jsonl",
    )

Keeping both artefacts under one loader means a single
``config.datasets.mock_wiki`` entry drives ingestion and evaluation
without the CLI having to juggle two named datasets.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from chimera_rag.core.exceptions import DatasetError
from chimera_rag.core.types import Document, QAExample
from chimera_rag.datasets.base import BaseDatasetLoader


class MockWikiLoader(BaseDatasetLoader):
    """Load paragraphs as :class:`Document` objects and JSONL rows as :class:`QAExample`."""

    def __init__(
        self,
        path: str | None = None,
        corpus_file: str = "corpus.txt",
        qa_file: str = "qa.jsonl",
        corpus_path: str | None = None,
        qa_path: str | None = None,
    ) -> None:
        # Explicit paths win.
        if corpus_path is not None:
            self.corpus_path = Path(corpus_path)
        elif path is not None:
            root = Path(path)
            self.corpus_path = root / corpus_file if root.is_dir() else Path(path)
        else:
            raise DatasetError("need at least one of 'path' / 'corpus_path'")

        if qa_path is not None:
            self.qa_path = Path(qa_path)
        elif path is not None:
            root = Path(path)
            self.qa_path = root / qa_file if root.is_dir() else Path(path).parent / qa_file
        else:
            # qa is optional at construction (ingest-only use cases)
            self.qa_path = Path("__missing__")

    # ------------------------------------------------------------------
    def load_documents(self, limit: int | None = None) -> list[Document]:
        if not self.corpus_path.is_file():
            raise DatasetError(f"corpus not found: {self.corpus_path}")
        text = self.corpus_path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        stem = self.corpus_path.stem
        return [
            Document(doc_id=f"{stem}::{i}", content=p, metadata={"source": str(self.corpus_path)})
            for i, p in enumerate(paragraphs)
        ]

    # ------------------------------------------------------------------
    def load_examples(self, limit: int | None = None) -> list[QAExample]:
        if not self.qa_path.is_file():
            raise DatasetError(f"qa file not found: {self.qa_path}")
        examples: list[QAExample] = []
        with self.qa_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if limit is not None and len(examples) >= limit:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetError(f"malformed JSON at line {i + 1}: {e}") from e
                try:
                    qid = str(obj.get("qid", f"auto-{uuid.uuid4().hex[:8]}"))
                    question = str(obj["question"])
                    answer = str(obj["answer"])
                except KeyError as e:
                    raise DatasetError(f"missing field at line {i + 1}: {e}") from e
                examples.append(QAExample(qid=qid, question=question, answer=answer))
        return examples


__all__ = ["MockWikiLoader"]

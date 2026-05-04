"""Generic JSONL QA loader.

Each line is a JSON object; the loader maps source field names to the
canonical ``qid / question / answer`` triple via the optional
``eval_field_mapping`` dict.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from chimera_rag.core.exceptions import DatasetError
from chimera_rag.core.types import QAExample
from chimera_rag.datasets.base import BaseDatasetLoader


class GenericJsonlLoader(BaseDatasetLoader):
    """Load QA pairs from a JSONL file."""

    def __init__(
        self,
        path: str,
        eval_field_mapping: dict[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.mapping = {
            "qid": "qid",
            "question": "question",
            "answer": "answer",
            **(eval_field_mapping or {}),
        }

    def load_examples(self, limit: int | None = None) -> list[QAExample]:
        if not self.path.is_file():
            raise DatasetError(f"file not found: {self.path}")
        examples: list[QAExample] = []
        with self.path.open(encoding="utf-8") as f:
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
                    qid = str(obj.get(self.mapping["qid"], f"auto-{uuid.uuid4().hex[:8]}"))
                    question = str(obj[self.mapping["question"]])
                    answer = str(obj[self.mapping["answer"]])
                except KeyError as e:
                    raise DatasetError(f"missing required field at line {i + 1}: {e}") from e
                examples.append(QAExample(qid=qid, question=question, answer=answer))
        return examples


__all__ = ["GenericJsonlLoader"]

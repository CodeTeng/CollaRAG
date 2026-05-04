"""Base classes and registry for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Document, QAExample


class BaseDatasetLoader(ABC):
    """Abstract parent for dataset loaders.

    Subclasses implement whichever of the two methods makes sense:

    * :meth:`load_documents` — raw corpus for ingestion
    * :meth:`load_examples` — (question, answer) pairs for evaluation
    """

    def load_documents(self) -> list[Document]:
        """Return the corpus as :class:`Document` objects."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support document loading"
        )

    def load_examples(self, limit: int | None = None) -> list[QAExample]:
        """Return QA examples. ``limit`` caps the count for smoke runs."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support example loading"
        )


__all__ = ["BaseDatasetLoader"]

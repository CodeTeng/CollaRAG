"""Dataset loaders and registry."""

from __future__ import annotations

from chimera_rag.core.exceptions import DatasetError
from chimera_rag.datasets.base import BaseDatasetLoader
from chimera_rag.datasets.generic_jsonl import GenericJsonlLoader
from chimera_rag.datasets.hotpotqa import HotpotQALoader
from chimera_rag.datasets.mock_wiki import MockWikiLoader
from chimera_rag.datasets.musique import MusiqueLoader
from chimera_rag.datasets.plain_text import PlainTextLoader
from chimera_rag.datasets.two_wiki import TwoWikiLoader

_LOADERS: dict[str, type[BaseDatasetLoader]] = {
    "plain_text": PlainTextLoader,
    "generic_jsonl": GenericJsonlLoader,
    "hotpotqa": HotpotQALoader,
    "two_wiki": TwoWikiLoader,
    "musique": MusiqueLoader,
    "mock_wiki": MockWikiLoader,
}


def get_dataset_loader(name: str) -> type[BaseDatasetLoader]:
    try:
        return _LOADERS[name]
    except KeyError as e:
        raise DatasetError(
            f"unknown dataset loader: {name!r}; "
            f"available={sorted(_LOADERS)}"
        ) from e


def register_dataset_loader(name: str, cls: type[BaseDatasetLoader]) -> None:
    """Allow plugins / tests to add custom loaders."""
    _LOADERS[name] = cls


__all__ = [
    "BaseDatasetLoader",
    "GenericJsonlLoader",
    "HotpotQALoader",
    "MockWikiLoader",
    "MusiqueLoader",
    "PlainTextLoader",
    "TwoWikiLoader",
    "get_dataset_loader",
    "register_dataset_loader",
]

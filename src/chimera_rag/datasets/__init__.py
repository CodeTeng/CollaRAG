"""Dataset loaders and registry."""

from __future__ import annotations

from typing import Type

from chimera_rag.core.exceptions import DatasetError
from chimera_rag.datasets.base import BaseDatasetLoader
from chimera_rag.datasets.generic_jsonl import GenericJsonlLoader
from chimera_rag.datasets.hotpotqa import HotpotQALoader
from chimera_rag.datasets.musique import MusiqueLoader
from chimera_rag.datasets.plain_text import PlainTextLoader
from chimera_rag.datasets.two_wiki import TwoWikiLoader

_LOADERS: dict[str, Type[BaseDatasetLoader]] = {
    "plain_text": PlainTextLoader,
    "generic_jsonl": GenericJsonlLoader,
    "hotpotqa": HotpotQALoader,
    "two_wiki": TwoWikiLoader,
    "musique": MusiqueLoader,
}


def get_dataset_loader(name: str) -> Type[BaseDatasetLoader]:
    try:
        return _LOADERS[name]
    except KeyError as e:
        raise DatasetError(
            f"unknown dataset loader: {name!r}; "
            f"available={sorted(_LOADERS)}"
        ) from e


def register_dataset_loader(name: str, cls: Type[BaseDatasetLoader]) -> None:
    """Allow plugins / tests to add custom loaders."""
    _LOADERS[name] = cls


__all__ = [
    "BaseDatasetLoader",
    "GenericJsonlLoader",
    "HotpotQALoader",
    "MusiqueLoader",
    "PlainTextLoader",
    "TwoWikiLoader",
    "get_dataset_loader",
    "register_dataset_loader",
]

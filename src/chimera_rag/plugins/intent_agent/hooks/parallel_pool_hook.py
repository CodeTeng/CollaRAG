"""Async semaphore-limited retrieval pool.

Small utility to fan out retrievals while capping concurrency.
``run(tasks)`` preserves input order in the returned list so callers can
zip results with inputs unambiguously.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


class ParallelRetrievalPool:
    """Simple ``asyncio.gather`` + semaphore pool."""

    def __init__(self, max_workers: int = 4) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

    async def run(self, tasks: list[Awaitable[T]]) -> list[T]:
        if not tasks:
            return []
        sem = asyncio.Semaphore(self.max_workers)

        async def _wrap(coro: Awaitable[T]) -> T:
            async with sem:
                return await coro

        return list(await asyncio.gather(*[_wrap(t) for t in tasks]))


__all__ = ["ParallelRetrievalPool"]

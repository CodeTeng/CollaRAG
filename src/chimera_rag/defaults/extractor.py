"""Simple LLM-based triple extractor — the vanilla baseline.

Pipeline: build a prompt containing the chunk text -> call the LLM ->
parse JSON list of triples. Robust to common LLM quirks:

* Markdown code fences (``` / ```json) stripped before parsing
* Leading prose (e.g. "Sure, here is the JSON") tolerated by greedy
  first/last bracket match
* Any parse failure yields ``[]`` rather than crashing the ingest job.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Triple
from chimera_rag.interfaces.extractor import BaseTripleExtractor
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


DEFAULT_PROMPT = """# Role
You are an expert knowledge-graph builder that extracts structured factual triples from text.

# Task
Extract all factual (subject, predicate, object) triples from the text below.

# Context
Text:
{text}

# Few-Shot
Text: "Marie Curie was born in Warsaw and won the Nobel Prize in Physics in 1903."
Output: [{{"subject": "Marie Curie", "predicate": "born_in", "object": "Warsaw"}}, {{"subject": "Marie Curie", "predicate": "won", "object": "Nobel Prize in Physics"}}, {{"subject": "Marie Curie", "predicate": "won_year", "object": "1903"}}]

# Format
Return a JSON array of objects with exactly the keys "subject", "predicate", "object". Do not include any explanations outside the JSON array."""


@register("extractor", "defaults.simple_llm")
class SimpleLLMExtractor(BaseTripleExtractor):
    """One-shot LLM extractor that returns explicit triples only (layer='L1')."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        prompt_template: str = DEFAULT_PROMPT,
        max_triples_per_chunk: int = 30,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_triples_per_chunk = max_triples_per_chunk

    # ------------------------------------------------------------------
    def extract(self, chunk: Chunk) -> list[Triple]:
        prompt = self.prompt_template.format(text=chunk.text)

        try:
            raw = _run_async(self.llm.complete(prompt))
        except Exception as e:  # pragma: no cover - provider errors
            logger.warning("extractor: LLM call failed for chunk %s: %s", chunk.chunk_id, e)
            return []

        parsed = _safe_parse_triples(raw)
        if not parsed:
            logger.debug("extractor: no triples parsed from chunk %s", chunk.chunk_id)
            return []

        triples: list[Triple] = []
        for item in parsed[: self.max_triples_per_chunk]:
            try:
                triples.append(
                    Triple(
                        subject=str(item["subject"]),
                        predicate=str(item["predicate"]),
                        object=str(item["object"]),
                        source_chunk_id=chunk.chunk_id,
                        layer="L1",
                    )
                )
            except (KeyError, TypeError) as e:
                logger.debug("extractor: skipping malformed triple %r (%s)", item, e)
        return triples


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _safe_parse_triples(text: str) -> list[dict]:
    """Try hard to recover a JSON list from ``text``. Return [] on failure."""
    # 1. strip code fences
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. attempt direct json.loads
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 3. greedy fallback: slice the first '[' to last ']'.
        lbr = text.find("[")
        rbr = text.rfind("]")
        if lbr == -1 or rbr == -1 or rbr <= lbr:
            return []
        try:
            data = json.loads(text[lbr : rbr + 1])
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _run_async(coro):
    """Run ``coro`` from sync code, safe whether or not a loop is running.

    When invoked inside an already-running loop (pytest-asyncio with
    ``asyncio_mode=auto``) we use a fresh loop in a worker thread to avoid
    the "asyncio.run() cannot be called from a running event loop" error.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Inside a running loop: execute coro synchronously on a private loop.
    import concurrent.futures

    def _worker():
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result()


__all__ = ["DEFAULT_PROMPT", "SimpleLLMExtractor"]

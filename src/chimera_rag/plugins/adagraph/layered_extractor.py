"""AdaGraph's layered triple extractor (L1 implemented; L2/L3/L4 reserved).

The design decomposes extraction into four complementary passes so each
one is specialised and easy to evaluate in ablation studies:

================ ===============================================================
Layer            Purpose
================ ===============================================================
L1 (explicit)    Literal (s, p, o) triples stated in the text — reuses the
                 vanilla ``SimpleLLMExtractor`` prompt under the hood.
L2 (implicit)    Coreferential, causal, or temporal inferences that aren't
                 literally stated. Reserved for a future TDD cycle.
L3 (schema)      Triples constrained by an entity-type schema (e.g. PERSON +
                 relation PREDICATE + ORG). Reserved.
L4 (cross-sent)  Facts that require fusing evidence across multiple sentences
                 within the same chunk. Reserved.
================ ===============================================================

The layer that produced each triple is recorded in ``Triple.layer`` so
downstream analysis and web UI can filter by source.
"""

from __future__ import annotations

import logging

from chimera_rag.core.registry import register
from chimera_rag.core.types import Chunk, Triple
from chimera_rag.defaults.extractor import DEFAULT_PROMPT, SimpleLLMExtractor
from chimera_rag.interfaces.extractor import BaseTripleExtractor
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


ALL_LAYERS = ("L1", "L2", "L3", "L4")


@register("extractor", "adagraph.layered")
class LayeredTripleExtractor(BaseTripleExtractor):
    """Run each enabled layer; concatenate and tag outputs by layer."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        layers_enabled: list[str] | None = None,
        prompt_template: str = DEFAULT_PROMPT,
        max_triples_per_chunk: int = 30,
    ) -> None:
        layers_enabled = list(layers_enabled) if layers_enabled else ["L1"]
        for lyr in layers_enabled:
            if lyr not in ALL_LAYERS:
                raise ValueError(f"unknown layer {lyr!r}; supported: {ALL_LAYERS}")
        self.llm = llm
        self.layers_enabled = layers_enabled
        self.max_triples_per_chunk = max_triples_per_chunk
        # L1 delegates to the vanilla extractor.
        self._l1 = SimpleLLMExtractor(
            llm=llm,
            prompt_template=prompt_template,
            max_triples_per_chunk=max_triples_per_chunk,
        )

    # ------------------------------------------------------------------
    def extract(self, chunk: Chunk) -> list[Triple]:
        out: list[Triple] = []
        for layer in self.layers_enabled:
            if layer == "L1":
                out.extend(self._l1.extract(chunk))
            elif layer == "L2":
                out.extend(self._extract_l2(chunk))
            elif layer == "L3":
                out.extend(self._extract_l3(chunk))
            elif layer == "L4":
                out.extend(self._extract_l4(chunk))
        return out[: self.max_triples_per_chunk]

    # ------------------------------------------------------------------
    # Reserved layers: will be implemented in follow-up TDD cycles.
    # ------------------------------------------------------------------
    def _extract_l2(self, chunk: Chunk) -> list[Triple]:
        logger.debug("L2 (implicit) reserved — skipping chunk %s", chunk.chunk_id)
        return []

    def _extract_l3(self, chunk: Chunk) -> list[Triple]:
        logger.debug("L3 (schema) reserved — skipping chunk %s", chunk.chunk_id)
        return []

    def _extract_l4(self, chunk: Chunk) -> list[Triple]:
        logger.debug("L4 (cross-sentence) reserved — skipping chunk %s", chunk.chunk_id)
        return []


__all__ = ["ALL_LAYERS", "LayeredTripleExtractor"]

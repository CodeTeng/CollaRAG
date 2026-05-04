"""Prompt-based answer generator — vanilla baseline.

Builds a context window from retrieved chunks and triples, plugs it into
the prompt template, and asks the LLM for a final answer.
"""

from __future__ import annotations

import logging

from chimera_rag.core.registry import register
from chimera_rag.core.types import Answer, Query, RetrievalResult
from chimera_rag.defaults.llm_extractor import _run_async
from chimera_rag.interfaces.generator import BaseAnswerGenerator
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


DEFAULT_PROMPT = """Use the CONTEXT to answer the QUESTION concisely.
If the answer is not in the context, say "I do not know."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


@register("generator", "defaults.prompt")
class PromptBasedGenerator(BaseAnswerGenerator):
    """Pack retrieval into a prompt and return the LLM's response."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    # ------------------------------------------------------------------
    def generate(self, query: Query, retrieval: RetrievalResult) -> Answer:
        context = self._build_context(retrieval)
        prompt = self.prompt_template.format(context=context, question=query.text)

        try:
            text = _run_async(self.llm.complete(prompt))
        except Exception as e:  # pragma: no cover
            logger.warning("generator: LLM call failed: %s", e)
            text = "I do not know."

        return Answer(
            text=text.strip(),
            evidence_chunk_ids=[c.chunk_id for c in retrieval.chunks],
            intent_label=None,
            strategy_name=None,
            confidence=1.0,
            trace={"prompt_len": len(prompt)},
        )

    # ------------------------------------------------------------------
    def _build_context(self, retrieval: RetrievalResult) -> str:
        parts: list[str] = []
        for i, ch in enumerate(retrieval.chunks, 1):
            parts.append(f"[chunk #{i}] {ch.text}")
        for t in retrieval.triples:
            parts.append(f"[triple] ({t.subject}, {t.predicate}, {t.object})")
        return "\n".join(parts) if parts else "(no context)"


__all__ = ["DEFAULT_PROMPT", "PromptBasedGenerator"]

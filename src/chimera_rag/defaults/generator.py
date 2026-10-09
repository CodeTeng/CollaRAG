"""Prompt-based answer generator — vanilla baseline.

Builds a context window from retrieved chunks and triples, plugs it into
the prompt template, and asks the LLM for a final answer.
"""

from __future__ import annotations

import logging

from chimera_rag.core.registry import register
from chimera_rag.core.types import Answer, Query, RetrievalResult
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.generator import BaseAnswerGenerator
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


DEFAULT_PROMPT = """# Role
You are a precise question-answering assistant for a Retrieval-Augmented Generation system.

# Task
Answer the QUESTION using ONLY the CONTEXT provided. Extract the shortest accurate answer span.

# Context
{context}

# Constraints
- Give the SHORTEST possible answer span: a name, entity, number, year, or phrase copied verbatim from the context
- Do NOT write a full sentence or repeat the question
- Do NOT add explanations, prefixes, punctuation, or quotes
- If the answer is not in the context, output exactly: I do not know.

# Few-Shot
QUESTION: In what year was Acme Corp founded?
ANSWER: 1998

QUESTION: Where is Jane Doe based?
ANSWER: Berlin

QUESTION: Which company does John work at?
ANSWER: Lumen Labs

# Format
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

# Multi-Agent RAG Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing CollaRAG (M-R-D-E + 6 strategies + LangGraph) with a multi-Agent architecture that routes queries to specialized Agents (Preprocessor → TreeIntent Classifier → SingleHop / MultiHop / Summarization / Other), backed by a shared tool layer and dual-layer memory system.

**Architecture:** A `MultiAgentRetriever` facade registers as `colla_rag.multi_agent` in the existing retriever slot. Inside, an Orchestrator runs: Preprocessor Agent → Tree-Intent Classifier → dispatch to one of four specialized Agents. Each Agent inherits from `BaseAgent` (Template Method pattern) and receives its tool subset from `ToolSetBuilder`. A dual-layer memory (shared + agent-private) persists across sessions.

**Tech Stack:** Python 3.11+, Pydantic v2, LangGraph ≥1.0, langchain-core ≥1.0, rank-bm25, duckduckgo-search, FAISS, NetworkX, sentence-transformers (optional reranker).

## Global Constraints

- Python ≥3.11, Pydantic v2 (strict validation, frozen where appropriate)
- All new code under `src/chimera_rag/plugins/colla_rag/` — never import plugins from `core/` or `defaults/`
- `@register` decorator for all slot implementations
- Tests use `MockProvider` — zero API keys in CI
- AdaGraph (innovation 1) is untouched — only query-side changes
- LangGraph ≥1.0 / langchain-core ≥1.0 (already in `[agent]` extra)
- New optional deps: `rank-bm25>=0.2.2`, `duckduckgo-search>=7.0`

## File Structure

```
src/chimera_rag/
├── core/
│   └── types.py                          # MODIFY: add new types
├── interfaces/
│   └── base_agent.py                     # CREATE: BaseAgent ABC
├── plugins/colla_rag/
│   ├── __init__.py                       # MODIFY: re-export new modules
│   ├── orchestrator.py                   # CREATE: MultiAgentRetriever facade
│   ├── classifier.py                     # CREATE: TreeIntentClassifier
│   ├── agent_factory.py                  # CREATE: AgentFactory + ToolSetBuilder
│   ├── agents/
│   │   ├── __init__.py                   # CREATE
│   │   ├── preprocessor.py              # CREATE: PreprocessorAgent
│   │   ├── single_hop.py                # CREATE: SingleHopAgent
│   │   ├── multi_hop.py               # CREATE: MultiHopAgent (LangGraph)
│   │   ├── summarization.py                # CREATE: SummarizationAgent
│   │   └── other.py                     # CREATE: OtherAgent
│   ├── tools/
│   │   ├── __init__.py                   # CREATE: build_all_tools + TOOL_MATRIX
│   │   ├── retrieval_tools.py           # CREATE: vector/bm25/hybrid/graph tools
│   │   ├── reasoning_tools.py           # CREATE: sub_query/assess/rerank tools
│   │   ├── preprocess_tools.py          # CREATE: query_rewrite/coref/followup tools
│   │   ├── memory_tools.py             # CREATE: read/write memory tools
│   │   ├── web_tools.py                 # CREATE: web_search/web_fetch tools
│   │   └── io_tools.py                  # CREATE: save/load file tools
│   └── memory/
│       ├── __init__.py                   # CREATE
│       ├── shared_memory.py             # CREATE: SharedMemory
│       ├── agent_memory.py              # CREATE: AgentPrivateMemory
│       └── memory_manager.py            # CREATE: MemoryManager
tests/
├── test_multi_agent_types.py             # CREATE
├── test_base_agent.py                    # CREATE
├── test_tree_classifier.py               # CREATE
├── test_retrieval_tools.py               # CREATE
├── test_bm25_tool.py                     # CREATE
├── test_preprocess_tools.py              # CREATE
├── test_memory_tools.py                  # CREATE
├── test_shared_memory.py                 # CREATE
├── test_agent_memory.py                  # CREATE
├── test_memory_manager.py                # CREATE
├── test_preprocessor_agent.py            # CREATE
├── test_native_rag_agent.py              # CREATE
├── test_multi_hop.py                   # CREATE
├── test_map_reduce_agent.py              # CREATE
├── test_web_search_agent.py              # CREATE
├── test_agent_factory.py                 # CREATE
├── test_orchestrator.py                  # CREATE
└── test_pipeline_multi_agent.py          # CREATE
```

---

### Task 1: Core Types & BaseAgent Interface

**Files:**
- Modify: `src/chimera_rag/core/types.py`
- Create: `src/chimera_rag/interfaces/base_agent.py`
- Test: `tests/test_multi_agent_types.py`
- Test: `tests/test_base_agent.py`

**Interfaces:**
- Consumes: existing `Query`, `Answer`, `RetrievalResult`, `Intent`, `QualityFeedback` from `core/types.py`
- Produces:
  - `MultiAgentIntentLabel = Literal["greeting", "single_hop", "multi_hop", "summarization", "other"]`
  - `PreprocessResult(rewritten_query: str, original_query: str, is_followup: bool, entities_mentioned: list[str], language: str)`
  - `AgentContext` dataclass with `session_id`, `shared_memory`, `agent_memory`, `tool_context`, `trace`
  - `AgentExperienceEntry` Pydantic model
  - `BaseAgent` ABC with `execute(query, context) -> Answer` template method

- [ ] **Step 1: Write failing tests for new types**

```python
# tests/test_multi_agent_types.py
"""Tests for multi-agent type extensions."""
import pytest
from chimera_rag.core.types import (
    AgentExperienceEntry,
    MultiAgentIntentLabel,
    PreprocessResult,
)


def test_preprocess_result_valid():
    r = PreprocessResult(
        rewritten_query="What is Einstein's birthplace?",
        original_query="Where was he born?",
        is_followup=True,
        entities_mentioned=["Einstein"],
        language="en",
    )
    assert r.rewritten_query == "What is Einstein's birthplace?"
    assert r.is_followup is True


def test_preprocess_result_defaults():
    r = PreprocessResult(
        rewritten_query="hello",
        original_query="hello",
    )
    assert r.is_followup is False
    assert r.entities_mentioned == []
    assert r.language == "en"


def test_agent_experience_entry_valid():
    e = AgentExperienceEntry(
        agent_type="react",
        intent="multi_hop",
        query_text="What is X?",
        tool_sequence=["hybrid_search", "graph_neighbors"],
        reflection_count=1,
        final_quality=0.8,
        outcome="success",
    )
    assert e.agent_type == "react"
    assert e.outcome == "success"
    assert e.hit_count == 0


def test_agent_experience_entry_lesson():
    e = AgentExperienceEntry(
        agent_type="react",
        intent="multi_hop",
        query_text="What is X?",
        tool_sequence=["vector_search"],
        reflection_count=2,
        final_quality=0.3,
        outcome="lesson",
        lesson="vector_search alone is insufficient for multi-hop",
    )
    assert e.outcome == "lesson"
    assert e.lesson is not None


def test_agent_experience_entry_quality_bounds():
    with pytest.raises(Exception):
        AgentExperienceEntry(
            agent_type="react",
            intent="multi_hop",
            query_text="x",
            tool_sequence=[],
            reflection_count=0,
            final_quality=1.5,  # out of bounds
            outcome="success",
        )


def test_multi_agent_intent_label_values():
    valid: list[MultiAgentIntentLabel] = [
        "greeting", "single_hop", "multi_hop", "summarization", "other",
    ]
    assert len(valid) == 5
```

- [ ] **Step 2: Write failing test for BaseAgent**

```python
# tests/test_base_agent.py
"""Tests for BaseAgent ABC and AgentContext."""
import pytest
from chimera_rag.interfaces.base_agent import BaseAgent, AgentContext
from chimera_rag.core.types import Answer, Query


class DummyAgent(BaseAgent):
    """Minimal concrete agent for testing the template method."""

    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        return Answer(text=f"answer for: {query.text}")


def test_agent_context_creation():
    ctx = AgentContext(session_id="s1")
    assert ctx.session_id == "s1"
    assert ctx.trace == {}


def test_base_agent_template_method():
    agent = DummyAgent(agent_type="test", tools=[], config={})
    ctx = AgentContext(session_id="s1")
    answer = agent.execute(Query(text="hello"), ctx)
    assert answer.text == "answer for: hello"


def test_base_agent_trace_populated():
    agent = DummyAgent(agent_type="test", tools=[], config={})
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="test"), ctx)
    assert "agent_type" in answer.trace
    assert answer.trace["agent_type"] == "test"


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent(agent_type="x", tools=[], config={})  # type: ignore
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_multi_agent_types.py tests/test_base_agent.py -v`
Expected: FAIL — `ImportError` for missing types and classes

- [ ] **Step 4: Add new types to `core/types.py`**

Append the following to `src/chimera_rag/core/types.py`:

```python
# ---------------------------------------------------------------------------
# Multi-Agent architecture types (innovation 2 refactor)
# ---------------------------------------------------------------------------
MultiAgentIntentLabel = Literal[
    "greeting",
    "single_hop",
    "multi_hop",
    "summarization",
    "other",
]


class PreprocessResult(BaseModel):
    """Output of the Preprocessor Agent."""

    rewritten_query: str
    original_query: str
    is_followup: bool = False
    entities_mentioned: list[str] = Field(default_factory=list)
    language: str = "en"


class AgentExperienceEntry(BaseModel):
    """One distilled retrieval experience, bucketed by agent + intent."""

    agent_type: str
    intent: str
    query_text: str
    tool_sequence: list[str] = Field(default_factory=list)
    reflection_count: int = Field(default=0, ge=0)
    final_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome: Literal["success", "lesson"] = "success"
    lesson: str | None = None
    answer_text: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    hit_count: int = Field(default=0, ge=0)
    created_at: str = ""
```

Also add `MultiAgentIntentLabel`, `PreprocessResult`, `AgentExperienceEntry` to the `__all__` list.

- [ ] **Step 5: Create `BaseAgent` ABC**

```python
# src/chimera_rag/interfaces/base_agent.py
"""ABC for specialized retrieval agents (Template Method pattern)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chimera_rag.core.types import Answer, Query


@dataclass
class AgentContext:
    """Mutable per-query context threaded through agent execution."""

    session_id: str | None = None
    shared_memory: Any = None
    agent_memory: Any = None
    tool_context: Any = None
    trace: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all specialized agents. Template Method pattern:
    subclasses override do_execute(), not execute().
    """

    def __init__(self, agent_type: str, tools: list, config: dict) -> None:
        self.agent_type = agent_type
        self.tools = tools
        self.config = config

    def execute(self, query: Query, context: AgentContext) -> Answer:
        self._pre_execute(query, context)
        answer = self.do_execute(query, context)
        self._post_execute(query, answer, context)
        return answer

    def _pre_execute(self, query: Query, context: AgentContext) -> None:
        context.trace["agent_type"] = self.agent_type

    @abstractmethod
    def do_execute(self, query: Query, context: AgentContext) -> Answer: ...

    def _post_execute(self, query: Query, answer: Answer, context: AgentContext) -> None:
        answer.trace.update(context.trace)


__all__ = ["AgentContext", "BaseAgent"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_multi_agent_types.py tests/test_base_agent.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/chimera_rag/core/types.py src/chimera_rag/interfaces/base_agent.py tests/test_multi_agent_types.py tests/test_base_agent.py
git commit -m "feat: add multi-agent types and BaseAgent ABC (Task 1)"
```

---

### Task 2: Tree-Intent Classifier

**Files:**
- Create: `src/chimera_rag/plugins/colla_rag/classifier.py`
- Test: `tests/test_tree_classifier.py`

**Interfaces:**
- Consumes: `Query` from `core/types.py`, `BaseIntentClassifier` from `interfaces/intent_classifier.py`, `BaseLLMProvider` from `interfaces/llm_provider.py`
- Produces: `TreeIntentClassifier.classify(query: Query) -> Intent` — returns `Intent` with `label` set to one of `MultiAgentIntentLabel` values. Registered as `("intent_classifier", "colla_rag.tree")`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tree_classifier.py
"""Tests for TreeIntentClassifier."""
import pytest
from chimera_rag.core.types import Intent, Query
from chimera_rag.plugins.colla_rag.classifier import TreeIntentClassifier


class FakeLLM:
    """Mock LLM that returns a predefined label."""

    def __init__(self, response: str = "single_hop"):
        self.response = response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.response


def _make_classifier(llm_response: str = "single_hop") -> TreeIntentClassifier:
    return TreeIntentClassifier(llm=FakeLLM(llm_response))


def test_greeting_detected_by_rules():
    c = _make_classifier()
    result = c.classify(Query(text="你好"))
    assert result.label == "greeting"


def test_greeting_english():
    c = _make_classifier()
    result = c.classify(Query(text="Hello, how are you?"))
    assert result.label == "greeting"


def test_single_hop_from_llm():
    c = _make_classifier("single_hop")
    result = c.classify(Query(text="What is the capital of France?"))
    assert result.label == "single_hop"


def test_multi_hop_from_llm():
    c = _make_classifier("multi_hop")
    result = c.classify(Query(text="Compare Einstein and Bohr contributions"))
    assert result.label == "multi_hop"


def test_summarization_from_llm():
    c = _make_classifier("summarization")
    result = c.classify(Query(text="Summarize the key findings of this paper"))
    assert result.label == "summarization"


def test_other_from_llm():
    c = _make_classifier("other")
    result = c.classify(Query(text="What's the weather in Tokyo?"))
    assert result.label == "other"


def test_invalid_llm_output_falls_back_to_rule():
    c = _make_classifier("gibberish_not_a_label")
    result = c.classify(Query(text="What is quantum computing?"))
    assert result.label in ("single_hop", "multi_hop", "summarization", "other")


def test_confidence_range():
    c = _make_classifier("single_hop")
    result = c.classify(Query(text="What is X?"))
    assert 0.0 <= result.confidence <= 1.0


def test_rule_fallback_multi_hop_keywords():
    c = _make_classifier("gibberish")
    q = Query(text="Compare A and B and explain the relationship between them")
    result = c.classify(q)
    assert result.label == "multi_hop"


def test_rule_fallback_summarization_keywords():
    c = _make_classifier("gibberish")
    q = Query(text="Please summarize the following document for me")
    result = c.classify(q)
    assert result.label == "summarization"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_tree_classifier.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement TreeIntentClassifier**

```python
# src/chimera_rag/plugins/colla_rag/classifier.py
"""Tree-shaped intent classifier: Level-1 greeting filter + Level-2 LLM classification."""
from __future__ import annotations

import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Intent, Query
from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier

logger = logging.getLogger(__name__)

_VALID_LABELS = {"single_hop", "multi_hop", "summarization", "other"}

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|你好|嗨|哈喽|good\s*(morning|afternoon|evening)|howdy|greetings)"
    r"[!?.,;：！？。，\s]*$",
    re.IGNORECASE,
)
_GREETING_CONTAINS = re.compile(
    r"^(hi|hello|hey|你好|嗨)\b.*\b(how are you|how's it going|what's up|怎么样|吗)\b",
    re.IGNORECASE,
)

_MULTI_HOP_KW = re.compile(
    r"\b(compar[ei]|contrast|relationship|differ|versus|vs\.?|between .+ and"
    r"|how .+ relate|why .+ and .+|explore|investigat)"
    r"\b",
    re.IGNORECASE,
)
_SUMMARIZE_KW = re.compile(
    r"\b(summariz|总结|概括|归纳|overview|synopsis|recap|key (findings|points|takeaways))\b",
    re.IGNORECASE,
)

_CLASSIFY_PROMPT = """Classify the following query into exactly one category.
Categories:
- single_hop: simple factual question answerable from one passage
- multi_hop: requires reasoning across multiple facts, comparison, or exploration
- summarization: asks for a summary, overview, or synthesis of information
- other: does not fit any above category (e.g. weather, jokes, math)

Query: {query}

Respond with ONLY the category name, nothing else."""


@register("intent_classifier", "colla_rag.tree")
class TreeIntentClassifier(BaseIntentClassifier):
    def __init__(self, llm: object, confidence_threshold: float = 0.7,
                 rule_fallback: bool = True) -> None:
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.rule_fallback = rule_fallback

    def classify(self, query: Query) -> Intent:
        text = query.text.strip()

        # Level-1: greeting detection (rules only, no LLM cost)
        if self._is_greeting(text):
            return Intent(label="greeting", confidence=0.95, rationale="rule:greeting")

        # Level-2: LLM classification
        label = self._llm_classify(text)
        if label in _VALID_LABELS:
            return Intent(label=label, confidence=self.confidence_threshold,
                          rationale="llm")

        # Fallback: rule-based heuristics
        if self.rule_fallback:
            label = self._rule_classify(text)
            return Intent(label=label, confidence=0.5, rationale="rule:fallback")

        return Intent(label="single_hop", confidence=0.3, rationale="default")

    @staticmethod
    def _is_greeting(text: str) -> bool:
        if _GREETING_PATTERNS.match(text):
            return True
        if _GREETING_CONTAINS.match(text):
            return True
        return False

    def _llm_classify(self, text: str) -> str:
        from chimera_rag.defaults.extractor import _run_async

        prompt = _CLASSIFY_PROMPT.format(query=text)
        try:
            raw = _run_async(self.llm.generate(prompt, max_tokens=20))
            label = raw.strip().lower().replace('"', "").replace("'", "")
            # Extract first valid label if LLM returned extra text
            for candidate in _VALID_LABELS:
                if candidate in label:
                    return candidate
            return label
        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
            return "unknown"

    @staticmethod
    def _rule_classify(text: str) -> str:
        if _SUMMARIZE_KW.search(text):
            return "summarization"
        if _MULTI_HOP_KW.search(text):
            return "multi_hop"
        return "single_hop"


__all__ = ["TreeIntentClassifier"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_tree_classifier.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/chimera_rag/plugins/colla_rag/classifier.py tests/test_tree_classifier.py
git commit -m "feat: add TreeIntentClassifier with greeting filter + LLM + rule fallback (Task 2)"
```

---

### Task 3: Retrieval Tools (vector / BM25 / hybrid / graph)

**Files:**
- Create: `src/chimera_rag/plugins/colla_rag/tools/__init__.py`
- Create: `src/chimera_rag/plugins/colla_rag/tools/retrieval_tools.py`
- Modify: `pyproject.toml` (add `rank-bm25` dependency)
- Test: `tests/test_retrieval_tools.py`
- Test: `tests/test_bm25_tool.py`

**Interfaces:**
- Consumes: `BaseVectorStore.search()`, `BaseGraphStore.get_neighbors()`, `BaseEmbeddingProvider.encode()`, `Chunk`, `RetrievalResult` from existing interfaces
- Produces: `build_retrieval_tools(vector_store, graph_store, embedder, chunk_lookup, bm25_index) -> list[StructuredTool]` returning tools named `vector_search`, `bm25_search`, `hybrid_search`, `graph_neighbors`, `graph_path_search`, `graph_community_search`. Also produces `BM25Index` wrapper class and `ToolAccumulator` dataclass.

- [ ] **Step 1: Write failing tests for BM25Index and ToolAccumulator**

```python
# tests/test_bm25_tool.py
"""Tests for BM25Index wrapper and ToolAccumulator."""
from chimera_rag.core.types import Chunk
from chimera_rag.plugins.colla_rag.tools.retrieval_tools import (
    BM25Index,
    ToolAccumulator,
)


def _make_chunks() -> dict[str, Chunk]:
    return {
        "c1": Chunk(chunk_id="c1", doc_id="d1", index=0, text="Einstein was born in Ulm Germany"),
        "c2": Chunk(chunk_id="c2", doc_id="d1", index=1, text="Bohr developed the atomic model"),
        "c3": Chunk(chunk_id="c3", doc_id="d1", index=2, text="Quantum mechanics revolutionized physics"),
    }


def test_bm25_index_build_and_search():
    chunks = _make_chunks()
    idx = BM25Index.from_chunks(chunks)
    results = idx.search("Einstein born", top_k=2)
    assert len(results) <= 2
    assert results[0][0] == "c1"  # best match
    assert results[0][1] > 0.0


def test_bm25_index_empty():
    idx = BM25Index.from_chunks({})
    results = idx.search("anything", top_k=5)
    assert results == []


def test_tool_accumulator_add_and_merge():
    acc = ToolAccumulator()
    chunks = _make_chunks()
    acc.add_chunks([chunks["c1"], chunks["c2"]], [0.9, 0.7])
    acc.add_chunks([chunks["c1"], chunks["c3"]], [0.8, 0.6])  # c1 duplicate
    merged = acc.merged_result(top_k=3)
    assert len(merged.chunks) == 3
    # c1 should keep highest score
    c1_idx = next(i for i, c in enumerate(merged.chunks) if c.chunk_id == "c1")
    assert merged.scores[c1_idx] == 0.9


def test_tool_accumulator_top_k_limit():
    acc = ToolAccumulator()
    chunks = _make_chunks()
    acc.add_chunks(list(chunks.values()), [0.9, 0.7, 0.5])
    merged = acc.merged_result(top_k=2)
    assert len(merged.chunks) == 2
```

- [ ] **Step 2: Write failing tests for retrieval tool functions**

```python
# tests/test_retrieval_tools.py
"""Tests for retrieval tools (vector, bm25, hybrid, graph)."""
import numpy as np
import pytest
from chimera_rag.core.types import Chunk, Triple
from chimera_rag.plugins.colla_rag.tools.retrieval_tools import (
    BM25Index,
    ToolAccumulator,
    build_retrieval_tools,
)


class MockVectorStore:
    def search(self, query_vector, top_k=5):
        return [("c1", 0.95), ("c2", 0.80)]

    def add(self, ids, vectors):
        pass


class MockGraphStore:
    def get_neighbors(self, entity, hops=1):
        if entity == "Einstein":
            return [Triple(subject="Einstein", predicate="born_in", object="Ulm")]
        return []

    def all_triples(self):
        return [
            Triple(subject="Einstein", predicate="born_in", object="Ulm"),
            Triple(subject="Bohr", predicate="developed", object="atomic model"),
        ]


class MockEmbedder:
    def encode(self, texts):
        return np.random.randn(len(texts), 64).astype(np.float32)


def _setup():
    chunks = {
        "c1": Chunk(chunk_id="c1", doc_id="d1", index=0, text="Einstein was born in Ulm"),
        "c2": Chunk(chunk_id="c2", doc_id="d1", index=1, text="Bohr developed atomic model"),
    }
    bm25 = BM25Index.from_chunks(chunks)
    acc = ToolAccumulator()
    return chunks, bm25, acc


def test_build_retrieval_tools_returns_six():
    chunks, bm25, acc = _setup()
    tools = build_retrieval_tools(
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        bm25_index=bm25,
        accumulator=acc,
    )
    names = {t.name for t in tools}
    assert "vector_search" in names
    assert "bm25_search" in names
    assert "hybrid_search" in names
    assert "graph_neighbors" in names
    assert "graph_path_search" in names
    assert "graph_community_search" in names


def test_vector_search_tool_invocation():
    chunks, bm25, acc = _setup()
    tools = build_retrieval_tools(
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        bm25_index=bm25,
        accumulator=acc,
    )
    vs_tool = next(t for t in tools if t.name == "vector_search")
    result = vs_tool.invoke({"query": "Einstein", "top_k": 2})
    assert result["tool"] == "vector_search"
    assert result["num_chunks"] >= 0


def test_bm25_search_tool_invocation():
    chunks, bm25, acc = _setup()
    tools = build_retrieval_tools(
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        bm25_index=bm25,
        accumulator=acc,
    )
    bm_tool = next(t for t in tools if t.name == "bm25_search")
    result = bm_tool.invoke({"query": "Einstein born", "top_k": 2})
    assert result["tool"] == "bm25_search"


def test_graph_neighbors_tool():
    chunks, bm25, acc = _setup()
    tools = build_retrieval_tools(
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        bm25_index=bm25,
        accumulator=acc,
    )
    gn_tool = next(t for t in tools if t.name == "graph_neighbors")
    result = gn_tool.invoke({"entity": "Einstein", "hops": 1})
    assert result["tool"] == "graph_neighbors"
    assert result["num_triples"] >= 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_bm25_tool.py tests/test_retrieval_tools.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Add `rank-bm25` to `pyproject.toml`**

In `pyproject.toml`, modify the `agent` extra:

```toml
agent = [
    "langgraph>=1.0,<2",
    "langchain-core>=1.0,<2",
    "rank-bm25>=0.2.2",
]
```

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv sync --extra agent`

- [ ] **Step 5: Create `tools/__init__.py`**

```python
# src/chimera_rag/plugins/colla_rag/tools/__init__.py
"""Shared tool layer for multi-agent architecture."""
```

- [ ] **Step 6: Implement retrieval_tools.py**

```python
# src/chimera_rag/plugins/colla_rag/tools/retrieval_tools.py
"""Retrieval tools: vector, BM25, hybrid, graph neighbors, graph path, graph community."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.core.types import Chunk, RetrievalResult, Triple

_PREVIEW_CHARS = 160


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"


# ---------------------------------------------------------------------------
# BM25 Index wrapper
# ---------------------------------------------------------------------------
class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi for chunk retrieval."""

    def __init__(self, chunk_ids: list[str], bm25: Any) -> None:
        self._chunk_ids = chunk_ids
        self._bm25 = bm25

    @classmethod
    def from_chunks(cls, chunk_lookup: dict[str, Chunk]) -> BM25Index:
        if not chunk_lookup:
            return cls([], None)
        from rank_bm25 import BM25Okapi

        ids = list(chunk_lookup.keys())
        tokenized = [re.findall(r"\w+", chunk_lookup[cid].text.lower()) for cid in ids]
        bm25 = BM25Okapi(tokenized)
        return cls(ids, bm25)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._chunk_ids[i], float(s)) for i, s in ranked if s > 0]


# ---------------------------------------------------------------------------
# Tool Accumulator
# ---------------------------------------------------------------------------
@dataclass
class ToolAccumulator:
    """Mutable per-query evidence accumulator shared across tool calls."""

    _chunks: dict[str, Chunk] = field(default_factory=dict)
    _scores: dict[str, float] = field(default_factory=dict)
    _triples: list[Triple] = field(default_factory=list)
    _seen_triple_keys: set[tuple[str, str, str]] = field(default_factory=set)

    def add_chunks(self, chunks: list[Chunk], scores: list[float]) -> None:
        for chunk, score in zip(chunks, scores):
            self._chunks.setdefault(chunk.chunk_id, chunk)
            prev = self._scores.get(chunk.chunk_id)
            if prev is None or score > prev:
                self._scores[chunk.chunk_id] = float(score)

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            key = (t.subject, t.predicate, t.object)
            if key not in self._seen_triple_keys:
                self._seen_triple_keys.add(key)
                self._triples.append(t)

    def merged_result(self, top_k: int | None = None) -> RetrievalResult:
        ordered = sorted(self._scores.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            ordered = ordered[:top_k]
        chunks = [self._chunks[cid] for cid, _ in ordered]
        scores = [s for _, s in ordered]
        return RetrievalResult(chunks=chunks, triples=list(self._triples), scores=scores)


# ---------------------------------------------------------------------------
# Pydantic arg schemas
# ---------------------------------------------------------------------------
class _QueryTopKArgs(BaseModel):
    query: str = Field(description="Natural-language query.")
    top_k: int = Field(default=5, ge=1, le=50)


class _HybridArgs(BaseModel):
    query: str = Field(description="Natural-language query.")
    top_k: int = Field(default=5, ge=1, le=50)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class _GraphNeighborsArgs(BaseModel):
    entity: str = Field(description="Entity name to look up.")
    hops: int = Field(default=1, ge=1, le=4)


class _GraphPathArgs(BaseModel):
    entity_a: str = Field(description="Start entity.")
    entity_b: str = Field(description="End entity.")
    max_depth: int = Field(default=4, ge=1, le=8)


class _GraphCommunityArgs(BaseModel):
    query: str = Field(description="Query for community search.")
    top_k: int = Field(default=3, ge=1, le=20)


def _summarise(chunks: list[Chunk], scores: list[float], label: str) -> dict:
    return {
        "tool": label,
        "num_chunks": len(chunks),
        "chunks": [
            {"chunk_id": c.chunk_id, "score": round(float(s), 4), "preview": _preview(c.text)}
            for c, s in zip(chunks, scores)
        ],
    }


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------
def build_retrieval_tools(
    *,
    vector_store,
    graph_store,
    embedder,
    chunk_lookup: dict[str, Chunk],
    bm25_index: BM25Index,
    accumulator: ToolAccumulator,
) -> list[StructuredTool]:

    def _vector_search(query: str, top_k: int = 5) -> dict:
        qvec = embedder.encode([query])[0]
        hits = vector_store.search(qvec, top_k=top_k)
        chunks, scores = [], []
        for cid, score in hits:
            ch = chunk_lookup.get(cid)
            if ch:
                chunks.append(ch)
                scores.append(score)
        accumulator.add_chunks(chunks, scores)
        return _summarise(chunks, scores, "vector_search")

    def _bm25_search(query: str, top_k: int = 5) -> dict:
        hits = bm25_index.search(query, top_k=top_k)
        chunks, scores = [], []
        for cid, score in hits:
            ch = chunk_lookup.get(cid)
            if ch:
                chunks.append(ch)
                scores.append(score)
        accumulator.add_chunks(chunks, scores)
        return _summarise(chunks, scores, "bm25_search")

    def _hybrid_search(query: str, top_k: int = 5, vector_weight: float = 0.5) -> dict:
        qvec = embedder.encode([query])[0]
        vec_hits = vector_store.search(qvec, top_k=top_k * 2)
        bm25_hits = bm25_index.search(query, top_k=top_k * 2)
        # RRF (Reciprocal Rank Fusion)
        rrf: dict[str, float] = {}
        k = 60
        for rank, (cid, _) in enumerate(vec_hits):
            rrf[cid] = rrf.get(cid, 0.0) + vector_weight / (k + rank + 1)
        for rank, (cid, _) in enumerate(bm25_hits):
            rrf[cid] = rrf.get(cid, 0.0) + (1 - vector_weight) / (k + rank + 1)
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        chunks, scores = [], []
        for cid, score in ranked:
            ch = chunk_lookup.get(cid)
            if ch:
                chunks.append(ch)
                scores.append(score)
        accumulator.add_chunks(chunks, scores)
        return _summarise(chunks, scores, "hybrid_search")

    def _graph_neighbors(entity: str, hops: int = 1) -> dict:
        triples = graph_store.get_neighbors(entity, hops=hops)
        accumulator.add_triples(triples)
        return {
            "tool": "graph_neighbors",
            "entity": entity,
            "hops": hops,
            "num_triples": len(triples),
            "triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in triples[:20]
            ],
        }

    def _graph_path_search(entity_a: str, entity_b: str, max_depth: int = 4) -> dict:
        import networkx as nx

        g = getattr(graph_store, "_graph", None)
        if g is None:
            return {"tool": "graph_path_search", "paths": [], "error": "graph not accessible"}
        ug = g.to_undirected()
        paths = []
        try:
            for path in nx.all_simple_paths(ug, entity_a, entity_b, cutoff=max_depth):
                paths.append(path)
                if len(paths) >= 5:
                    break
        except (nx.NodeNotFound, nx.NetworkXError):
            pass
        return {"tool": "graph_path_search", "entity_a": entity_a, "entity_b": entity_b, "num_paths": len(paths), "paths": paths}

    def _graph_community_search(query: str, top_k: int = 3) -> dict:
        all_triples = graph_store.all_triples()
        query_lower = query.lower()
        scored = []
        for t in all_triples:
            text = f"{t.subject} {t.predicate} {t.object}".lower()
            overlap = len(set(re.findall(r"\w+", query_lower)) & set(re.findall(r"\w+", text)))
            if overlap > 0:
                scored.append((t, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_triples = [t for t, _ in scored[:top_k * 3]]
        accumulator.add_triples(top_triples)
        return {
            "tool": "graph_community_search",
            "num_triples": len(top_triples),
            "triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in top_triples[:20]
            ],
        }

    return [
        StructuredTool.from_function(_vector_search, name="vector_search",
            description="Semantic vector search. Use for direct factual lookups.",
            args_schema=_QueryTopKArgs),
        StructuredTool.from_function(_bm25_search, name="bm25_search",
            description="BM25 keyword search. Use when exact term matching matters.",
            args_schema=_QueryTopKArgs),
        StructuredTool.from_function(_hybrid_search, name="hybrid_search",
            description="RRF fusion of vector + BM25. Best general-purpose retrieval.",
            args_schema=_HybridArgs),
        StructuredTool.from_function(_graph_neighbors, name="graph_neighbors",
            description="Expand knowledge graph around an entity by N hops.",
            args_schema=_GraphNeighborsArgs),
        StructuredTool.from_function(_graph_path_search, name="graph_path_search",
            description="Find paths between two entities in the knowledge graph.",
            args_schema=_GraphPathArgs),
        StructuredTool.from_function(_graph_community_search, name="graph_community_search",
            description="Find triples related to a query topic in the knowledge graph.",
            args_schema=_GraphCommunityArgs),
    ]


__all__ = ["BM25Index", "ToolAccumulator", "build_retrieval_tools"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_bm25_tool.py tests/test_retrieval_tools.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/chimera_rag/plugins/colla_rag/tools/ tests/test_bm25_tool.py tests/test_retrieval_tools.py
git commit -m "feat: add retrieval tools — vector, BM25, hybrid RRF, graph (Task 3)"
```

---

### Task 4: Preprocessing, Reasoning, Memory, Web, and IO Tools

**Files:**
- Create: `src/chimera_rag/plugins/colla_rag/tools/preprocess_tools.py`
- Create: `src/chimera_rag/plugins/colla_rag/tools/reasoning_tools.py`
- Create: `src/chimera_rag/plugins/colla_rag/tools/memory_tools.py`
- Create: `src/chimera_rag/plugins/colla_rag/tools/web_tools.py`
- Create: `src/chimera_rag/plugins/colla_rag/tools/io_tools.py`
- Modify: `pyproject.toml` (add `duckduckgo-search` dependency)
- Test: `tests/test_preprocess_tools.py`
- Test: `tests/test_memory_tools.py`

**Interfaces:**
- Consumes: `ToolAccumulator` from Task 3, `BaseLLMProvider`, `BaseReflector`, `BaseReranker`
- Produces:
  - `build_preprocess_tools(llm, session_memory) -> list[StructuredTool]` — tools: `query_rewrite`, `coreference_resolve`, `followup_merge`
  - `build_reasoning_tools(llm, reflector, reranker, accumulator, chunk_lookup) -> list[StructuredTool]` — tools: `sub_query_decompose`, `assess_evidence`, `rerank`
  - `build_memory_tools(shared_memory, agent_memory) -> list[StructuredTool]` — tools: `read_session_context`, `write_session_context`, `read_shared_memory`, `write_shared_memory`, `read_agent_memory`, `write_agent_memory`
  - `build_web_tools() -> list[StructuredTool]` — tools: `web_search`, `web_fetch`
  - `build_io_tools() -> list[StructuredTool]` — tools: `save_to_file`, `load_from_file`

- [ ] **Step 1: Write failing tests for preprocess tools**

```python
# tests/test_preprocess_tools.py
"""Tests for preprocess tools (query_rewrite, coreference_resolve, followup_merge)."""
from chimera_rag.plugins.colla_rag.tools.preprocess_tools import build_preprocess_tools


class FakeLLM:
    async def generate(self, prompt: str, **kw) -> str:
        if "rewrite" in prompt.lower() or "改写" in prompt.lower():
            return "What is Albert Einstein's birthplace?"
        if "coreference" in prompt.lower() or "指代" in prompt.lower():
            return "What is Einstein's birthplace?"
        return prompt


class FakeSessionMemory:
    def __init__(self):
        self._store: dict[str, list[str]] = {}

    def get_history(self, sid: str) -> list[str]:
        return self._store.get(sid, [])

    def append(self, sid: str, turn: str):
        self._store.setdefault(sid, []).append(turn)


def test_build_preprocess_tools_count():
    tools = build_preprocess_tools(llm=FakeLLM(), session_memory=FakeSessionMemory())
    names = {t.name for t in tools}
    assert "query_rewrite" in names
    assert "coreference_resolve" in names
    assert "followup_merge" in names
    assert len(tools) == 3


def test_query_rewrite_invocation():
    tools = build_preprocess_tools(llm=FakeLLM(), session_memory=FakeSessionMemory())
    rw = next(t for t in tools if t.name == "query_rewrite")
    result = rw.invoke({"query": "Where was he born?", "context": "Discussing Einstein"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_followup_merge_invocation():
    tools = build_preprocess_tools(llm=FakeLLM(), session_memory=FakeSessionMemory())
    fm = next(t for t in tools if t.name == "followup_merge")
    result = fm.invoke({
        "query": "What about his awards?",
        "prev_query": "Tell me about Einstein",
        "prev_answer": "Einstein was a physicist",
    })
    assert isinstance(result, str)
```

- [ ] **Step 2: Write failing tests for memory tools**

```python
# tests/test_memory_tools.py
"""Tests for memory tools."""
from chimera_rag.plugins.colla_rag.tools.memory_tools import build_memory_tools


class FakeSharedMemory:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def read(self, memory_type: str, key: str):
        return self._store.get(memory_type, {}).get(key)

    def write(self, memory_type: str, key: str, value):
        self._store.setdefault(memory_type, {})[key] = value


class FakeAgentMemory:
    def __init__(self):
        self._store: dict[str, list] = {}

    def recall(self, agent_type: str, intent: str, top_n: int = 3):
        return self._store.get(f"{agent_type}:{intent}", [])[:top_n]

    def remember(self, agent_type: str, entry: dict):
        key = f"{agent_type}:{entry.get('intent', 'unknown')}"
        self._store.setdefault(key, []).append(entry)


class FakeSessionMemory:
    def __init__(self):
        self._h: dict[str, list[str]] = {}

    def get_history(self, sid):
        return self._h.get(sid, [])

    def append(self, sid, turn):
        self._h.setdefault(sid, []).append(turn)


def test_build_memory_tools_count():
    tools = build_memory_tools(
        shared_memory=FakeSharedMemory(),
        agent_memory=FakeAgentMemory(),
        session_memory=FakeSessionMemory(),
    )
    names = {t.name for t in tools}
    assert "read_session_context" in names
    assert "write_session_context" in names
    assert "read_shared_memory" in names
    assert "write_shared_memory" in names
    assert "read_agent_memory" in names
    assert "write_agent_memory" in names


def test_read_write_shared_memory():
    sm = FakeSharedMemory()
    tools = build_memory_tools(shared_memory=sm, agent_memory=FakeAgentMemory(), session_memory=FakeSessionMemory())
    write_tool = next(t for t in tools if t.name == "write_shared_memory")
    read_tool = next(t for t in tools if t.name == "read_shared_memory")
    write_tool.invoke({"memory_type": "qa_cache", "key": "q1", "value": "answer1"})
    result = read_tool.invoke({"memory_type": "qa_cache", "key": "q1"})
    assert result == "answer1"


def test_read_write_session_context():
    sess = FakeSessionMemory()
    tools = build_memory_tools(shared_memory=FakeSharedMemory(), agent_memory=FakeAgentMemory(), session_memory=sess)
    write_tool = next(t for t in tools if t.name == "write_session_context")
    read_tool = next(t for t in tools if t.name == "read_session_context")
    write_tool.invoke({"session_id": "s1", "query": "hi", "answer": "hello"})
    result = read_tool.invoke({"session_id": "s1", "last_n": 5})
    assert len(result) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_preprocess_tools.py tests/test_memory_tools.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Add `duckduckgo-search` to `pyproject.toml`**

In `pyproject.toml`, add to the `agent` extra:

```toml
agent = [
    "langgraph>=1.0,<2",
    "langchain-core>=1.0,<2",
    "rank-bm25>=0.2.2",
    "duckduckgo-search>=7.0",
]
```

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv sync --extra agent`

- [ ] **Step 5: Implement all five tool modules**

Create `src/chimera_rag/plugins/colla_rag/tools/preprocess_tools.py`:

```python
"""Preprocessing tools: query rewrite, coreference resolution, follow-up merge."""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.defaults.extractor import _run_async

_REWRITE_PROMPT = """Rewrite the following query to be clearer, more specific, and self-contained.
Context: {context}
Original query: {query}
Rewritten query:"""

_COREF_PROMPT = """Resolve all pronouns and references in the query using the conversation history.
History: {history}
Query: {query}
Resolved query (replace pronouns with explicit entities):"""

class _RewriteArgs(BaseModel):
    query: str = Field(description="The query to rewrite.")
    context: str = Field(default="", description="Optional context for rewriting.")

class _CorefArgs(BaseModel):
    query: str = Field(description="Query with potential pronouns.")
    history: list[str] = Field(default_factory=list, description="Previous conversation turns.")

class _FollowupArgs(BaseModel):
    query: str = Field(description="The follow-up query.")
    prev_query: str = Field(default="", description="Previous query.")
    prev_answer: str = Field(default="", description="Previous answer.")


def build_preprocess_tools(*, llm, session_memory) -> list[StructuredTool]:
    def _query_rewrite(query: str, context: str = "") -> str:
        prompt = _REWRITE_PROMPT.format(query=query, context=context)
        return _run_async(llm.generate(prompt, max_tokens=200)).strip()

    def _coreference_resolve(query: str, history: list[str] | None = None) -> str:
        hist_str = " | ".join(history or [])
        prompt = _COREF_PROMPT.format(history=hist_str, query=query)
        return _run_async(llm.generate(prompt, max_tokens=200)).strip()

    def _followup_merge(query: str, prev_query: str = "", prev_answer: str = "") -> str:
        merged = f"{prev_query} {prev_answer} {query}".strip()
        prompt = _REWRITE_PROMPT.format(query=merged, context="Follow-up question")
        return _run_async(llm.generate(prompt, max_tokens=200)).strip()

    return [
        StructuredTool.from_function(_query_rewrite, name="query_rewrite",
            description="Rewrite a query for clarity and specificity.", args_schema=_RewriteArgs),
        StructuredTool.from_function(_coreference_resolve, name="coreference_resolve",
            description="Resolve pronouns using conversation history.", args_schema=_CorefArgs),
        StructuredTool.from_function(_followup_merge, name="followup_merge",
            description="Merge a follow-up with previous QA into a self-contained query.", args_schema=_FollowupArgs),
    ]

__all__ = ["build_preprocess_tools"]
```

Create `src/chimera_rag/plugins/colla_rag/tools/reasoning_tools.py`:

```python
"""Reasoning tools: sub-query decomposition, evidence assessment, reranking."""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async

_DECOMPOSE_PROMPT = """Decompose the following complex query into simpler sub-queries.
Return a JSON list of strings, e.g. ["sub-query 1", "sub-query 2"].
Query: {query}
Sub-queries:"""


class _DecomposeArgs(BaseModel):
    query: str = Field(description="Complex query to decompose.")
    max_sub: int = Field(default=4, ge=1, le=8)

class _AssessArgs(BaseModel):
    query: str = Field(description="Original query to judge relevance against.")

class _RerankArgs(BaseModel):
    query: str = Field(description="Query for reranking.")
    chunk_ids: list[str] = Field(description="Chunk IDs to rerank.")
    top_k: int = Field(default=5, ge=1, le=50)


def build_reasoning_tools(*, llm, reflector, reranker, accumulator, chunk_lookup) -> list[StructuredTool]:

    def _sub_query_decompose(query: str, max_sub: int = 4) -> list[str]:
        prompt = _DECOMPOSE_PROMPT.format(query=query)
        raw = _run_async(llm.generate(prompt, max_tokens=300))
        try:
            subs = json.loads(raw.strip())
            if isinstance(subs, list):
                return [str(s) for s in subs[:max_sub]]
        except (json.JSONDecodeError, TypeError):
            pass
        return [query]

    def _assess_evidence(query: str) -> dict:
        merged = accumulator.merged_result()
        draft = Answer(
            text=" ".join(c.text for c in merged.chunks[:10]),
            evidence_chunk_ids=[c.chunk_id for c in merged.chunks],
        )
        feedback = _run_async(reflector.evaluate(Query(text=query), draft))
        return {
            "tool": "assess_evidence",
            "quality": round(float(feedback.quality), 4),
            "relevance": round(float(feedback.relevance), 4),
            "completeness": round(float(feedback.completeness), 4),
            "num_evidence_chunks": len(merged.chunks),
            "suggestion": feedback.suggestion,
        }

    def _rerank(query: str, chunk_ids: list[str], top_k: int = 5) -> dict:
        from chimera_rag.core.types import RetrievalResult
        chunks = [chunk_lookup[cid] for cid in chunk_ids if cid in chunk_lookup]
        if not chunks:
            return {"tool": "rerank", "num_chunks": 0, "chunks": []}
        rr = RetrievalResult(chunks=chunks, scores=[1.0] * len(chunks))
        if reranker is not None:
            rr = _run_async(reranker.rerank(Query(text=query), rr))
            rr_chunks = rr.chunks[:top_k]
        else:
            rr_chunks = chunks[:top_k]
        return {
            "tool": "rerank",
            "num_chunks": len(rr_chunks),
            "chunks": [{"chunk_id": c.chunk_id, "preview": c.text[:160]} for c in rr_chunks],
        }

    return [
        StructuredTool.from_function(_sub_query_decompose, name="sub_query_decompose",
            description="Decompose a complex query into simpler sub-queries.", args_schema=_DecomposeArgs),
        StructuredTool.from_function(_assess_evidence, name="assess_evidence",
            description="Score quality/relevance/completeness of gathered evidence.", args_schema=_AssessArgs),
        StructuredTool.from_function(_rerank, name="rerank",
            description="Rerank chunks by semantic relevance using a cross-encoder.", args_schema=_RerankArgs),
    ]

__all__ = ["build_reasoning_tools"]
```

Create `src/chimera_rag/plugins/colla_rag/tools/memory_tools.py`:

```python
"""Memory and context tools: read/write session, shared, and agent memory."""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class _ReadSessionArgs(BaseModel):
    session_id: str
    last_n: int = Field(default=5, ge=1, le=50)

class _WriteSessionArgs(BaseModel):
    session_id: str
    query: str
    answer: str

class _ReadSharedArgs(BaseModel):
    memory_type: str
    key: str

class _WriteSharedArgs(BaseModel):
    memory_type: str
    key: str
    value: Any

class _ReadAgentArgs(BaseModel):
    agent_type: str
    intent: str
    top_n: int = Field(default=3, ge=1, le=20)

class _WriteAgentArgs(BaseModel):
    agent_type: str
    entry: dict


def build_memory_tools(*, shared_memory, agent_memory, session_memory) -> list[StructuredTool]:

    def _read_session_context(session_id: str, last_n: int = 5) -> list[str]:
        return session_memory.get_history(session_id)[-last_n:]

    def _write_session_context(session_id: str, query: str, answer: str) -> str:
        session_memory.append(session_id, f"Q: {query} | A: {answer}")
        return "ok"

    def _read_shared_memory(memory_type: str, key: str) -> Any:
        return shared_memory.read(memory_type, key)

    def _write_shared_memory(memory_type: str, key: str, value: Any) -> str:
        shared_memory.write(memory_type, key, value)
        return "ok"

    def _read_agent_memory(agent_type: str, intent: str, top_n: int = 3) -> list:
        return agent_memory.recall(agent_type, intent, top_n=top_n)

    def _write_agent_memory(agent_type: str, entry: dict) -> str:
        agent_memory.remember(agent_type, entry)
        return "ok"

    return [
        StructuredTool.from_function(_read_session_context, name="read_session_context",
            description="Read recent conversation history.", args_schema=_ReadSessionArgs),
        StructuredTool.from_function(_write_session_context, name="write_session_context",
            description="Write a QA turn to session history.", args_schema=_WriteSessionArgs),
        StructuredTool.from_function(_read_shared_memory, name="read_shared_memory",
            description="Read from shared memory (QA cache, entity aliases, etc).", args_schema=_ReadSharedArgs),
        StructuredTool.from_function(_write_shared_memory, name="write_shared_memory",
            description="Write to shared memory.", args_schema=_WriteSharedArgs),
        StructuredTool.from_function(_read_agent_memory, name="read_agent_memory",
            description="Read agent-private experience memory.", args_schema=_ReadAgentArgs),
        StructuredTool.from_function(_write_agent_memory, name="write_agent_memory",
            description="Write to agent-private experience memory.", args_schema=_WriteAgentArgs),
    ]

__all__ = ["build_memory_tools"]
```

Create `src/chimera_rag/plugins/colla_rag/tools/web_tools.py`:

```python
"""Web tools: search and fetch."""
from __future__ import annotations

import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _WebSearchArgs(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)

class _WebFetchArgs(BaseModel):
    url: str
    max_chars: int = Field(default=3000, ge=100, le=50000)


def build_web_tools() -> list[StructuredTool]:

    def _web_search(query: str, max_results: int = 5) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")} for r in results]
        except ImportError:
            logger.warning("duckduckgo-search not installed; returning empty results")
            return []
        except Exception as e:
            logger.warning("web_search failed: %s", e)
            return []

    def _web_fetch(url: str, max_chars: int = 3000) -> str:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            import re
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception as e:
            return f"fetch failed: {e}"

    return [
        StructuredTool.from_function(_web_search, name="web_search",
            description="Search the web via DuckDuckGo.", args_schema=_WebSearchArgs),
        StructuredTool.from_function(_web_fetch, name="web_fetch",
            description="Fetch and extract text content from a URL.", args_schema=_WebFetchArgs),
    ]

__all__ = ["build_web_tools"]
```

Create `src/chimera_rag/plugins/colla_rag/tools/io_tools.py`:

```python
"""IO tools: save and load files."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class _SaveArgs(BaseModel):
    path: str
    content: str
    format: str = Field(default="text", description="'json' or 'text'")

class _LoadArgs(BaseModel):
    path: str


def build_io_tools() -> list[StructuredTool]:

    def _save_to_file(path: str, content: str, format: str = "text") -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            try:
                data = json.loads(content)
                p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except json.JSONDecodeError:
                p.write_text(content, encoding="utf-8")
        else:
            p.write_text(content, encoding="utf-8")
        return f"saved to {path}"

    def _load_from_file(path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"file not found: {path}"
        return p.read_text(encoding="utf-8")

    return [
        StructuredTool.from_function(_save_to_file, name="save_to_file",
            description="Save content to a file.", args_schema=_SaveArgs),
        StructuredTool.from_function(_load_from_file, name="load_from_file",
            description="Load content from a file.", args_schema=_LoadArgs),
    ]

__all__ = ["build_io_tools"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_preprocess_tools.py tests/test_memory_tools.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/chimera_rag/plugins/colla_rag/tools/ tests/test_preprocess_tools.py tests/test_memory_tools.py
git commit -m "feat: add preprocess, reasoning, memory, web, IO tools (Task 4)"
```

---

### Task 5: Dual-Layer Memory System

**Files:**
- Create: `src/chimera_rag/plugins/colla_rag/memory/__init__.py`
- Create: `src/chimera_rag/plugins/colla_rag/memory/shared_memory.py`
- Create: `src/chimera_rag/plugins/colla_rag/memory/agent_memory.py`
- Create: `src/chimera_rag/plugins/colla_rag/memory/memory_manager.py`
- Test: `tests/test_shared_memory.py`
- Test: `tests/test_agent_memory.py`
- Test: `tests/test_memory_manager.py`

**Interfaces:**
- Consumes: `AgentExperienceEntry` from Task 1
- Produces:
  - `SharedMemory.read(memory_type, key)`, `.write(memory_type, key, value)`, `.get_qa_cache(query)`, `.stats()`
  - `AgentPrivateMemory.recall(agent_type, intent, top_n)`, `.remember(agent_type, entry)`, `.stats()`
  - `MemoryManager(shared, agent_private, session)` — unified facade

- [ ] **Step 1: Write failing tests for SharedMemory**

```python
# tests/test_shared_memory.py
"""Tests for SharedMemory (routing stats, QA cache, entity aliases, rewrite templates)."""
import tempfile
from pathlib import Path

from chimera_rag.plugins.colla_rag.memory.shared_memory import SharedMemory


def test_shared_memory_read_write():
    with tempfile.TemporaryDirectory() as d:
        sm = SharedMemory(path=d)
        sm.write("qa_cache", "q1_hash", {"answer": "42", "quality": 0.9})
        result = sm.read("qa_cache", "q1_hash")
        assert result["answer"] == "42"


def test_shared_memory_read_missing():
    with tempfile.TemporaryDirectory() as d:
        sm = SharedMemory(path=d)
        assert sm.read("qa_cache", "nonexistent") is None


def test_shared_memory_routing_stats():
    with tempfile.TemporaryDirectory() as d:
        sm = SharedMemory(path=d)
        sm.write("routing_stats", "q1", {"intent": "multi_hop", "agent": "react", "quality": 0.8})
        result = sm.read("routing_stats", "q1")
        assert result["intent"] == "multi_hop"


def test_shared_memory_persistence():
    with tempfile.TemporaryDirectory() as d:
        sm1 = SharedMemory(path=d)
        sm1.write("entity_aliases", "Einstein", "Albert Einstein")
        sm1.persist()
        sm2 = SharedMemory(path=d)
        sm2.load()
        assert sm2.read("entity_aliases", "Einstein") == "Albert Einstein"


def test_shared_memory_stats():
    with tempfile.TemporaryDirectory() as d:
        sm = SharedMemory(path=d)
        sm.write("qa_cache", "q1", {"answer": "x"})
        sm.write("qa_cache", "q2", {"answer": "y"})
        s = sm.stats()
        assert s["qa_cache"] == 2
```

- [ ] **Step 2: Write failing tests for AgentPrivateMemory**

```python
# tests/test_agent_memory.py
"""Tests for AgentPrivateMemory."""
import tempfile

from chimera_rag.plugins.colla_rag.memory.agent_memory import AgentPrivateMemory


def test_remember_and_recall():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        entry = {
            "intent": "multi_hop",
            "query_text": "What is X?",
            "tool_sequence": ["hybrid_search", "graph_neighbors"],
            "final_quality": 0.8,
            "outcome": "success",
        }
        am.remember("react", entry)
        recalled = am.recall("react", "multi_hop", top_n=3)
        assert len(recalled) == 1
        assert recalled[0]["tool_sequence"] == ["hybrid_search", "graph_neighbors"]


def test_recall_empty():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        assert am.recall("react", "single_hop") == []


def test_dedup_on_same_tool_sequence():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        entry = {"intent": "multi_hop", "query_text": "Q", "tool_sequence": ["a", "b"],
                 "final_quality": 0.5, "outcome": "success"}
        am.remember("react", entry)
        entry2 = {**entry, "final_quality": 0.9}
        am.remember("react", entry2)
        recalled = am.recall("react", "multi_hop")
        assert len(recalled) == 1
        assert recalled[0]["final_quality"] == 0.9  # kept the better one


def test_cross_agent_isolation():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        am.remember("react", {"intent": "multi_hop", "query_text": "Q", "tool_sequence": ["a"],
                               "final_quality": 0.8, "outcome": "success"})
        am.remember("native_rag", {"intent": "single_hop", "query_text": "Q2", "tool_sequence": ["b"],
                                    "final_quality": 0.7, "outcome": "success"})
        assert len(am.recall("react", "multi_hop")) == 1
        assert len(am.recall("native_rag", "multi_hop")) == 0
        assert len(am.recall("native_rag", "single_hop")) == 1


def test_stats():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        am.remember("react", {"intent": "multi_hop", "query_text": "Q", "tool_sequence": [],
                               "final_quality": 0.8, "outcome": "success"})
        s = am.stats()
        assert s["total"] >= 1
```

- [ ] **Step 3: Write failing tests for MemoryManager**

```python
# tests/test_memory_manager.py
"""Tests for MemoryManager (dual-layer facade)."""
import tempfile

from chimera_rag.plugins.colla_rag.memory.memory_manager import MemoryManager


def test_memory_manager_creation():
    with tempfile.TemporaryDirectory() as d:
        mm = MemoryManager(base_path=d)
        assert mm.shared is not None
        assert mm.agent_private is not None
        assert mm.session is not None


def test_memory_manager_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        mm = MemoryManager(base_path=d)
        mm.shared.write("qa_cache", "q1", {"answer": "test"})
        assert mm.shared.read("qa_cache", "q1")["answer"] == "test"

        mm.agent_private.remember("react", {
            "intent": "multi_hop", "query_text": "Q",
            "tool_sequence": ["a"], "final_quality": 0.9, "outcome": "success",
        })
        assert len(mm.agent_private.recall("react", "multi_hop")) == 1
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_shared_memory.py tests/test_agent_memory.py tests/test_memory_manager.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 5: Implement memory modules**

Create `src/chimera_rag/plugins/colla_rag/memory/__init__.py`:

```python
"""Dual-layer memory system for multi-agent architecture."""
```

Create `src/chimera_rag/plugins/colla_rag/memory/shared_memory.py`:

```python
"""SharedMemory — JSON-backed global memory shared across all Agents."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SharedMemory:
    _TYPES = ("routing_stats", "qa_cache", "entity_aliases", "rewrite_templates")

    def __init__(self, path: str) -> None:
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict[str, Any]] = {t: {} for t in self._TYPES}
        self.load()

    def read(self, memory_type: str, key: str) -> Any | None:
        return self._store.get(memory_type, {}).get(key)

    def write(self, memory_type: str, key: str, value: Any) -> None:
        self._store.setdefault(memory_type, {})[key] = value

    def get_qa_cache(self, query_hash: str) -> dict | None:
        return self.read("qa_cache", query_hash)

    def persist(self) -> None:
        for mem_type, data in self._store.items():
            path = self.root / f"{mem_type}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        for mem_type in self._TYPES:
            path = self.root / f"{mem_type}.json"
            if path.exists():
                try:
                    self._store[mem_type] = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("failed to load %s: %s", path, e)

    def stats(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._store.items()}


__all__ = ["SharedMemory"]
```

Create `src/chimera_rag/plugins/colla_rag/memory/agent_memory.py`:

```python
"""AgentPrivateMemory — per-agent, per-intent JSONL experience store."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AgentPrivateMemory:
    def __init__(self, path: str, max_per_bucket: int = 200) -> None:
        self.root = Path(path)
        self.max_per_bucket = max_per_bucket

    def _bucket_path(self, agent_type: str, intent: str) -> Path:
        d = self.root / agent_type / "experience"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{intent}.jsonl"

    def _load_bucket(self, agent_type: str, intent: str) -> list[dict]:
        path = self._bucket_path(agent_type, intent)
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("skipping bad line in %s", path)
        return entries

    def _write_bucket(self, agent_type: str, intent: str, entries: list[dict]) -> None:
        path = self._bucket_path(agent_type, intent)
        payload = "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
        path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")

    def recall(self, agent_type: str, intent: str, top_n: int = 3) -> list[dict]:
        entries = self._load_bucket(agent_type, intent)
        ranked = sorted(entries, key=lambda e: (e.get("final_quality", 0), e.get("hit_count", 0)), reverse=True)
        return ranked[:top_n]

    def remember(self, agent_type: str, entry: dict) -> None:
        intent = entry.get("intent", "unknown")
        entries = self._load_bucket(agent_type, intent)
        dedup_key = (tuple(entry.get("tool_sequence", [])), entry.get("outcome", "success"))
        merged = False
        for existing in entries:
            existing_key = (tuple(existing.get("tool_sequence", [])), existing.get("outcome", "success"))
            if existing_key == dedup_key:
                existing["final_quality"] = max(existing.get("final_quality", 0), entry.get("final_quality", 0))
                existing["hit_count"] = existing.get("hit_count", 0) + 1
                merged = True
                break
        if not merged:
            entry.setdefault("hit_count", 0)
            entries.append(entry)
        if len(entries) > self.max_per_bucket:
            entries.sort(key=lambda e: (e.get("hit_count", 0), e.get("final_quality", 0)))
            entries = entries[len(entries) - self.max_per_bucket:]
        self._write_bucket(agent_type, intent, entries)

    def stats(self) -> dict:
        total = 0
        buckets: dict[str, int] = {}
        if self.root.exists():
            for agent_dir in self.root.iterdir():
                if agent_dir.is_dir() and (agent_dir / "experience").exists():
                    for f in (agent_dir / "experience").glob("*.jsonl"):
                        count = sum(1 for line in f.read_text().splitlines() if line.strip())
                        key = f"{agent_dir.name}/{f.stem}"
                        buckets[key] = count
                        total += count
        return {"total": total, "buckets": buckets}


__all__ = ["AgentPrivateMemory"]
```

Create `src/chimera_rag/plugins/colla_rag/memory/memory_manager.py`:

```python
"""MemoryManager — unified facade for dual-layer memory."""
from __future__ import annotations

from pathlib import Path

from chimera_rag.plugins.colla_rag.memory.agent_memory import AgentPrivateMemory
from chimera_rag.plugins.colla_rag.memory.shared_memory import SharedMemory


class SessionMemory:
    """Simple in-memory session history (same interface as BaseMemory)."""

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    def append(self, session_id: str, turn: str) -> None:
        self._store.setdefault(session_id, []).append(turn)

    def get_history(self, session_id: str) -> list[str]:
        return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class MemoryManager:
    """Dual-layer memory facade: shared + agent-private + session."""

    def __init__(self, base_path: str, max_per_bucket: int = 200) -> None:
        root = Path(base_path)
        self.shared = SharedMemory(path=str(root / "shared"))
        self.agent_private = AgentPrivateMemory(path=str(root), max_per_bucket=max_per_bucket)
        self.session = SessionMemory()


__all__ = ["MemoryManager", "SessionMemory"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_shared_memory.py tests/test_agent_memory.py tests/test_memory_manager.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/chimera_rag/plugins/colla_rag/memory/ tests/test_shared_memory.py tests/test_agent_memory.py tests/test_memory_manager.py
git commit -m "feat: add dual-layer memory system — shared + agent-private (Task 5)"
```

---

### Task 6: Agent Factory & ToolSetBuilder

**Files:**
- Create: `src/chimera_rag/plugins/colla_rag/agent_factory.py`
- Test: `tests/test_agent_factory.py`

**Interfaces:**
- Consumes: `BaseAgent` from Task 1, all `build_*_tools` from Tasks 3-4, `MemoryManager` from Task 5
- Produces:
  - `TOOL_MATRIX: dict[str, list[str]]` — agent_type → allowed tool names
  - `ToolSetBuilder.for_agent(agent_type, all_tools) -> list[StructuredTool]`
  - `AgentFactory.create(intent, all_tools, memory_manager, config) -> BaseAgent`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent_factory.py
"""Tests for AgentFactory and ToolSetBuilder."""
from chimera_rag.plugins.colla_rag.agent_factory import (
    TOOL_MATRIX,
    AgentFactory,
    ToolSetBuilder,
)
from langchain_core.tools import StructuredTool


def _dummy_tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(lambda: "ok", name=name, description=f"dummy {name}")


def test_tool_matrix_has_all_agents():
    assert "preprocessor" in TOOL_MATRIX
    assert "single_hop" in TOOL_MATRIX
    assert "multi_hop" in TOOL_MATRIX
    assert "summarization" in TOOL_MATRIX
    assert "other" in TOOL_MATRIX


def test_tool_set_builder_filters():
    all_tools = [_dummy_tool("vector_search"), _dummy_tool("web_search"), _dummy_tool("bm25_search")]
    filtered = ToolSetBuilder.for_agent("single_hop", all_tools)
    names = {t.name for t in filtered}
    assert "vector_search" in names
    assert "bm25_search" in names
    assert "web_search" not in names  # not in single_hop's tool set


def test_tool_set_builder_unknown_agent():
    all_tools = [_dummy_tool("vector_search")]
    filtered = ToolSetBuilder.for_agent("nonexistent_agent", all_tools)
    assert filtered == []  # unknown agent gets nothing


def test_agent_factory_register_and_create():
    from chimera_rag.interfaces.base_agent import BaseAgent, AgentContext
    from chimera_rag.core.types import Answer, Query

    @AgentFactory.register("single_hop")
    class TestAgent(BaseAgent):
        def do_execute(self, query: Query, context: AgentContext) -> Answer:
            return Answer(text="test")

    all_tools = [_dummy_tool("vector_search"), _dummy_tool("hybrid_search")]
    agent = AgentFactory.create("single_hop", all_tools=all_tools, memory_manager=None, config={})
    assert agent.agent_type == "single_hop"
    assert len(agent.tools) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_agent_factory.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement agent_factory.py**

```python
# src/chimera_rag/plugins/colla_rag/agent_factory.py
"""AgentFactory (Factory pattern) + ToolSetBuilder (Builder pattern)."""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from chimera_rag.interfaces.base_agent import BaseAgent

TOOL_MATRIX: dict[str, list[str]] = {
    "preprocessor": [
        "query_rewrite", "coreference_resolve", "followup_merge",
        "read_session_context", "read_shared_memory",
    ],
    "single_hop": [
        "vector_search", "bm25_search", "hybrid_search", "graph_neighbors",
        "assess_evidence", "rerank",
        "read_shared_memory", "read_agent_memory",
        "save_to_file", "load_from_file",
    ],
    "multi_hop": [
        "vector_search", "bm25_search", "hybrid_search",
        "graph_neighbors", "graph_path_search", "graph_community_search",
        "sub_query_decompose", "assess_evidence", "rerank",
        "read_session_context", "read_shared_memory", "read_agent_memory",
        "save_to_file", "load_from_file",
    ],
    "summarization": [
        "vector_search", "bm25_search", "hybrid_search",
        "graph_community_search", "rerank",
        "read_shared_memory", "read_agent_memory",
        "save_to_file", "load_from_file",
    ],
    "other": [
        "hybrid_search", "web_search", "web_fetch",
        "read_shared_memory", "read_agent_memory",
        "save_to_file", "load_from_file",
    ],
}


class ToolSetBuilder:
    @staticmethod
    def for_agent(agent_type: str, all_tools: list[StructuredTool]) -> list[StructuredTool]:
        allowed = set(TOOL_MATRIX.get(agent_type, []))
        return [t for t in all_tools if t.name in allowed]


class AgentFactory:
    _registry: dict[str, type[BaseAgent]] = {}

    @classmethod
    def register(cls, intent: str):
        def decorator(agent_cls: type[BaseAgent]) -> type[BaseAgent]:
            cls._registry[intent] = agent_cls
            return agent_cls
        return decorator

    @classmethod
    def create(cls, intent: str, *, all_tools: list[StructuredTool],
               memory_manager: Any, config: dict) -> BaseAgent:
        agent_cls = cls._registry.get(intent)
        if agent_cls is None:
            raise ValueError(f"No agent registered for intent {intent!r}")
        tools = ToolSetBuilder.for_agent(intent, all_tools)
        return agent_cls(agent_type=intent, tools=tools, config=config)


__all__ = ["AgentFactory", "TOOL_MATRIX", "ToolSetBuilder"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_agent_factory.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/chimera_rag/plugins/colla_rag/agent_factory.py tests/test_agent_factory.py
git commit -m "feat: add AgentFactory + ToolSetBuilder with TOOL_MATRIX (Task 6)"
```

---

### Task 7: Specialized Agents (Preprocessor + SingleHop + ReAct + Map-Reduce + WebSearch)

**Files:**
- Create: `src/chimera_rag/plugins/colla_rag/agents/__init__.py`
- Create: `src/chimera_rag/plugins/colla_rag/agents/preprocessor.py`
- Create: `src/chimera_rag/plugins/colla_rag/agents/single_hop.py`
- Create: `src/chimera_rag/plugins/colla_rag/agents/multi_hop.py`
- Create: `src/chimera_rag/plugins/colla_rag/agents/summarization.py`
- Create: `src/chimera_rag/plugins/colla_rag/agents/other.py`
- Test: `tests/test_preprocessor_agent.py`
- Test: `tests/test_native_rag_agent.py`
- Test: `tests/test_multi_hop.py`
- Test: `tests/test_map_reduce_agent.py`
- Test: `tests/test_web_search_agent.py`

**Interfaces:**
- Consumes: `BaseAgent` / `AgentContext` from Task 1, `ToolAccumulator` / `BM25Index` from Task 3, all tools from Tasks 3-4, `AgentFactory.register` from Task 6
- Produces: Five `BaseAgent` subclasses, each registered via `@AgentFactory.register(intent)`:
  - `PreprocessorAgent.do_execute(query, context) -> Answer` (returns dummy Answer; real output is side-effect on context)
  - `SingleHopAgent.do_execute(query, context) -> Answer`
  - `MultiHopAgent.do_execute(query, context) -> Answer`
  - `SummarizationAgent.do_execute(query, context) -> Answer`
  - `OtherAgent.do_execute(query, context) -> Answer`

This is the largest task. Each agent is a focused file. Tests use mock tools — no real LLM or vector store calls.

- [ ] **Step 1: Write failing tests for all five agents**

```python
# tests/test_preprocessor_agent.py
"""Tests for PreprocessorAgent."""
from chimera_rag.core.types import Query, PreprocessResult
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.preprocessor import PreprocessorAgent
from langchain_core.tools import StructuredTool


def _mock_rewrite_tool():
    return StructuredTool.from_function(
        lambda query, context="": f"rewritten: {query}",
        name="query_rewrite", description="mock rewrite",
    )

def _mock_coref_tool():
    return StructuredTool.from_function(
        lambda query, history=None: query,
        name="coreference_resolve", description="mock coref",
    )

def _mock_followup_tool():
    return StructuredTool.from_function(
        lambda query, prev_query="", prev_answer="": f"{prev_query} {query}".strip(),
        name="followup_merge", description="mock followup",
    )


def test_preprocessor_rewrites_query():
    agent = PreprocessorAgent(
        agent_type="preprocessor",
        tools=[_mock_rewrite_tool(), _mock_coref_tool(), _mock_followup_tool()],
        config={},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="Where was he born?"), ctx)
    assert "rewritten" in answer.text or "rewritten" in ctx.trace.get("rewritten_query", "")


def test_preprocessor_with_session():
    agent = PreprocessorAgent(
        agent_type="preprocessor",
        tools=[_mock_rewrite_tool(), _mock_coref_tool(), _mock_followup_tool()],
        config={},
    )
    ctx = AgentContext(session_id="s1")
    answer = agent.execute(Query(text="What about his awards?"), ctx)
    assert answer.text  # should produce something
```

```python
# tests/test_native_rag_agent.py
"""Tests for SingleHopAgent."""
from chimera_rag.core.types import Answer, Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.single_hop import SingleHopAgent
from langchain_core.tools import StructuredTool


def _mock_hybrid():
    return StructuredTool.from_function(
        lambda query, top_k=5, vector_weight=0.5: {"tool": "hybrid_search", "num_chunks": 2, "chunks": [
            {"chunk_id": "c1", "score": 0.9, "preview": "Einstein was born in Ulm"},
            {"chunk_id": "c2", "score": 0.7, "preview": "He won the Nobel Prize"},
        ]},
        name="hybrid_search", description="mock hybrid",
    )

def _mock_assess():
    return StructuredTool.from_function(
        lambda query: {"tool": "assess_evidence", "quality": 0.8, "relevance": 0.9,
                       "completeness": 0.7, "num_evidence_chunks": 2, "suggestion": None},
        name="assess_evidence", description="mock assess",
    )


def test_native_rag_produces_answer():
    class FakeLLM:
        async def generate(self, prompt, **kw):
            return "Einstein was born in Ulm, Germany."

    agent = SingleHopAgent(
        agent_type="single_hop",
        tools=[_mock_hybrid(), _mock_assess()],
        config={"llm": FakeLLM(), "top_k": 5},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="Where was Einstein born?"), ctx)
    assert len(answer.text) > 0
    assert answer.trace.get("agent_type") == "single_hop"
```

```python
# tests/test_multi_hop.py
"""Tests for MultiHopAgent (simplified — no real LangGraph, mock tools)."""
from chimera_rag.core.types import Answer, Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.multi_hop import MultiHopAgent
from langchain_core.tools import StructuredTool


def _mock_hybrid():
    return StructuredTool.from_function(
        lambda query, top_k=5, vector_weight=0.5: {"tool": "hybrid_search", "num_chunks": 1, "chunks": []},
        name="hybrid_search", description="mock",
    )

def _mock_assess():
    return StructuredTool.from_function(
        lambda query: {"tool": "assess_evidence", "quality": 0.8, "relevance": 0.8,
                       "completeness": 0.8, "num_evidence_chunks": 1, "suggestion": None},
        name="assess_evidence", description="mock",
    )


def test_react_agent_creates():
    class FakeLLM:
        async def generate(self, prompt, **kw):
            return "Final Answer: The answer is 42."

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[_mock_hybrid(), _mock_assess()],
        config={"llm": FakeLLM(), "max_iterations": 2, "quality_threshold": 0.6},
    )
    assert agent.agent_type == "multi_hop"


def test_react_agent_executes():
    class FakeLLM:
        async def generate(self, prompt, **kw):
            return "Final Answer: The answer is 42."

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[_mock_hybrid(), _mock_assess()],
        config={"llm": FakeLLM(), "max_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="What is 6*7?"), ctx)
    assert len(answer.text) > 0
```

```python
# tests/test_map_reduce_agent.py
"""Tests for SummarizationAgent."""
from chimera_rag.core.types import Answer, Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.summarization import SummarizationAgent
from langchain_core.tools import StructuredTool


def _mock_hybrid():
    return StructuredTool.from_function(
        lambda query, top_k=5, vector_weight=0.5: {"tool": "hybrid_search", "num_chunks": 2, "chunks": [
            {"chunk_id": "c1", "score": 0.9, "preview": "Point A about topic"},
            {"chunk_id": "c2", "score": 0.7, "preview": "Point B about topic"},
        ]},
        name="hybrid_search", description="mock",
    )


def test_map_reduce_produces_summary():
    class FakeLLM:
        async def generate(self, prompt, **kw):
            if "extract" in prompt.lower() or "提取" in prompt.lower():
                return "Key point from this chunk."
            return "Summary: The topic has two main points."

    agent = SummarizationAgent(
        agent_type="summarization",
        tools=[_mock_hybrid()],
        config={"llm": FakeLLM(), "retrieve_top_k": 10},
    )
    ctx = AgentContext(session_id=None)
    ctx.tool_context = {"chunk_lookup": {
        "c1": type("C", (), {"text": "Point A about topic", "chunk_id": "c1"})(),
        "c2": type("C", (), {"text": "Point B about topic", "chunk_id": "c2"})(),
    }}
    answer = agent.execute(Query(text="Summarize the topic"), ctx)
    assert len(answer.text) > 0
```

```python
# tests/test_web_search_agent.py
"""Tests for OtherAgent."""
from chimera_rag.core.types import Answer, Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.web_search import OtherAgent
from langchain_core.tools import StructuredTool


def _mock_web_search():
    return StructuredTool.from_function(
        lambda query, max_results=5: [{"title": "Result", "href": "http://example.com", "body": "Some info"}],
        name="web_search", description="mock",
    )


def test_web_search_agent():
    class FakeLLM:
        async def generate(self, prompt, **kw):
            return "Based on web results, the answer is X."

    agent = OtherAgent(
        agent_type="other",
        tools=[_mock_web_search()],
        config={"llm": FakeLLM()},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="What is the weather in Tokyo?"), ctx)
    assert len(answer.text) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_preprocessor_agent.py tests/test_native_rag_agent.py tests/test_multi_hop.py tests/test_map_reduce_agent.py tests/test_web_search_agent.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Create `agents/__init__.py`**

```python
# src/chimera_rag/plugins/colla_rag/agents/__init__.py
"""Specialized agents for multi-agent RAG architecture."""
from chimera_rag.plugins.colla_rag.agents.preprocessor import PreprocessorAgent
from chimera_rag.plugins.colla_rag.agents.single_hop import SingleHopAgent
from chimera_rag.plugins.colla_rag.agents.multi_hop import MultiHopAgent
from chimera_rag.plugins.colla_rag.agents.summarization import SummarizationAgent
from chimera_rag.plugins.colla_rag.agents.web_search import OtherAgent

__all__ = ["PreprocessorAgent", __all__ = ["MultiHopAgent", "OtherAgent", "PreprocessorAgent", "SingleHopAgent", "SummarizationAgent"]]
```

- [ ] **Step 4: Implement PreprocessorAgent**

```python
# src/chimera_rag/plugins/colla_rag/agents/preprocessor.py
"""PreprocessorAgent — query rewriting, coreference resolution, follow-up merging."""
from __future__ import annotations

from chimera_rag.core.types import Answer, Query
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory


@AgentFactory.register("preprocessor")
class PreprocessorAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        tool_map = {t.name: t for t in self.tools}
        rewritten = query.text

        if "query_rewrite" in tool_map:
            rewritten = tool_map["query_rewrite"].invoke({"query": rewritten, "context": ""})

        context.trace["rewritten_query"] = rewritten
        context.trace["original_query"] = query.text
        return Answer(text=rewritten)


__all__ = ["PreprocessorAgent"]
```

- [ ] **Step 5: Implement SingleHopAgent**

```python
# src/chimera_rag/plugins/colla_rag/agents/single_hop.py
"""SingleHopAgent — single-round retrieval for simple factual queries."""
from __future__ import annotations

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory

_GENERATE_PROMPT = """Answer the question based on the following evidence.
Evidence:
{evidence}

Question: {query}
Answer:"""


@AgentFactory.register("single_hop")
class SingleHopAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        top_k = self.config.get("top_k", 10)
        tool_map = {t.name: t for t in self.tools}

        # 1. Retrieve
        search_tool = tool_map.get("hybrid_search") or tool_map.get("vector_search")
        result = search_tool.invoke({"query": query.text, "top_k": top_k})
        chunks = result.get("chunks", [])
        evidence = "\n".join(c.get("preview", "") for c in chunks)
        chunk_ids = [c.get("chunk_id", "") for c in chunks]

        # 2. Lightweight quality check
        if "assess_evidence" in tool_map:
            assessment = tool_map["assess_evidence"].invoke({"query": query.text})
            quality = assessment.get("quality", 1.0)
            context.trace["quality"] = quality
            context.trace["low_confidence"] = quality < self.config.get("quality_threshold", 0.4)

        # 3. Generate
        prompt = _GENERATE_PROMPT.format(evidence=evidence, query=query.text)
        answer_text = _run_async(llm.generate(prompt, max_tokens=512))

        return Answer(
            text=answer_text.strip(),
            evidence_chunk_ids=chunk_ids,
            strategy_name="native_rag",
        )


__all__ = ["SingleHopAgent"]
```

- [ ] **Step 6: Implement MultiHopAgent**

```python
# src/chimera_rag/plugins/colla_rag/agents/multi_hop.py
"""MultiHopAgent — multi-step reasoning with tool selection and reflection loop."""
from __future__ import annotations

import json
import logging
import re

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a retrieval agent. You have access to these tools:
{tool_descriptions}

Use the following format:
Thought: reason about what to do next
Action: tool_name
Action Input: {{"param": "value"}}
... (repeat Thought/Action/Action Input as needed)
Final Answer: your final answer based on gathered evidence

Begin!

Question: {query}"""

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


@AgentFactory.register("multi_hop")
class MultiHopAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        max_iters = self.config.get("max_iterations", 3)
        quality_threshold = self.config.get("quality_threshold", 0.6)
        tool_map = {t.name: t for t in self.tools}

        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in self.tools)
        conversation = _SYSTEM_PROMPT.format(tool_descriptions=tool_descs, query=query.text)
        tool_call_log: list[str] = []

        for iteration in range(max_iters):
            raw = _run_async(llm.generate(conversation, max_tokens=1024))
            conversation += "\n" + raw

            # Check for final answer
            final_match = _FINAL_RE.search(raw)
            if final_match and not _ACTION_RE.search(raw):
                context.trace["iterations"] = iteration + 1
                context.trace["tool_call_log"] = tool_call_log
                return Answer(text=final_match.group(1).strip(), strategy_name="react")

            # Parse and execute tool call
            action_match = _ACTION_RE.search(raw)
            if action_match:
                tool_name = action_match.group(1)
                try:
                    tool_input = json.loads(action_match.group(2))
                except json.JSONDecodeError:
                    tool_input = {"query": query.text}

                tool = tool_map.get(tool_name)
                if tool:
                    tool_call_log.append(tool_name)
                    observation = tool.invoke(tool_input)
                    conversation += f"\nObservation: {json.dumps(observation, default=str)[:2000]}\n"
                else:
                    conversation += f"\nObservation: Tool '{tool_name}' not available.\n"
            else:
                # No action and no final answer — treat as final answer
                context.trace["iterations"] = iteration + 1
                context.trace["tool_call_log"] = tool_call_log
                return Answer(text=raw.strip(), strategy_name="react")

            # Reflection check after tool call
            if "assess_evidence" in tool_map:
                assessment = tool_map["assess_evidence"].invoke({"query": query.text})
                quality = assessment.get("quality", 0.0)
                if quality >= quality_threshold:
                    conversation += f"\nThought: Evidence quality ({quality:.2f}) is sufficient. I should give the final answer.\n"

        context.trace["iterations"] = max_iters
        context.trace["tool_call_log"] = tool_call_log
        # Max iterations reached — generate final answer from what we have
        final_prompt = f"{conversation}\nThought: I've gathered enough evidence. Let me provide a final answer.\nFinal Answer:"
        final_text = _run_async(llm.generate(final_prompt, max_tokens=512))
        return Answer(text=final_text.strip(), strategy_name="react")


__all__ = ["MultiHopAgent"]
```

- [ ] **Step 7: Implement SummarizationAgent**

```python
# src/chimera_rag/plugins/colla_rag/agents/summarization.py
"""SummarizationAgent — retrieve top-N chunks, map each to a summary, reduce into final answer."""
from __future__ import annotations

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory

_MAP_PROMPT = """Extract key information relevant to the query from this passage.
Query: {query}
Passage: {text}
Key information:"""

_REDUCE_PROMPT = """Based on the following extracted information, provide a comprehensive summary for the query.
Query: {query}
Extracted information:
{summaries}
Summary:"""


@AgentFactory.register("summarization")
class SummarizationAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        retrieve_top_k = self.config.get("retrieve_top_k", 50)
        tool_map = {t.name: t for t in self.tools}

        # 1. Retrieve
        search_tool = tool_map.get("hybrid_search") or tool_map.get("vector_search")
        result = search_tool.invoke({"query": query.text, "top_k": retrieve_top_k})
        chunks_data = result.get("chunks", [])

        # Resolve full chunk text from context if available
        chunk_lookup = {}
        if context.tool_context and isinstance(context.tool_context, dict):
            chunk_lookup = context.tool_context.get("chunk_lookup", {})

        texts = []
        chunk_ids = []
        for c in chunks_data:
            cid = c.get("chunk_id", "")
            chunk_ids.append(cid)
            ch = chunk_lookup.get(cid)
            if ch and hasattr(ch, "text"):
                texts.append(ch.text)
            else:
                texts.append(c.get("preview", ""))

        if not texts:
            return Answer(text="No relevant information found.", strategy_name="map_reduce")

        # 2. Map
        summaries = []
        for text in texts:
            prompt = _MAP_PROMPT.format(query=query.text, text=text[:1000])
            summary = _run_async(llm.generate(prompt, max_tokens=200))
            summaries.append(summary.strip())

        # 3. Reduce
        combined = "\n---\n".join(f"[{i+1}] {s}" for i, s in enumerate(summaries) if s)
        reduce_prompt = _REDUCE_PROMPT.format(query=query.text, summaries=combined)
        final = _run_async(llm.generate(reduce_prompt, max_tokens=1024))

        context.trace["map_count"] = len(summaries)
        return Answer(
            text=final.strip(),
            evidence_chunk_ids=chunk_ids,
            strategy_name="map_reduce",
        )


__all__ = ["SummarizationAgent"]
```

- [ ] **Step 8: Implement OtherAgent**

```python
# src/chimera_rag/plugins/colla_rag/agents/other.py
"""OtherAgent — search the web + optional local knowledge, then generate."""
from __future__ import annotations

from chimera_rag.core.types import Answer, Query
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory

_GENERATE_PROMPT = """Answer the question based on the following web search results and local knowledge.
Web results:
{web_results}

Local knowledge:
{local_knowledge}

Question: {query}
Answer:"""


@AgentFactory.register("other")
class OtherAgent(BaseAgent):
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        llm = self.config.get("llm")
        tool_map = {t.name: t for t in self.tools}

        # 1. Web search
        web_results_text = ""
        if "web_search" in tool_map:
            results = tool_map["web_search"].invoke({"query": query.text, "max_results": 5})
            if isinstance(results, list):
                web_results_text = "\n".join(
                    f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
                )

        # 2. Optional local hybrid search
        local_text = ""
        if "hybrid_search" in tool_map:
            local = tool_map["hybrid_search"].invoke({"query": query.text, "top_k": 5})
            chunks = local.get("chunks", [])
            local_text = "\n".join(c.get("preview", "") for c in chunks)

        # 3. Generate
        prompt = _GENERATE_PROMPT.format(
            web_results=web_results_text or "(none)",
            local_knowledge=local_text or "(none)",
            query=query.text,
        )
        answer_text = _run_async(llm.generate(prompt, max_tokens=1024))

        context.trace["web_results_count"] = len(web_results_text.split("\n")) if web_results_text else 0
        return Answer(text=answer_text.strip(), strategy_name="web_search")


__all__ = ["OtherAgent"]
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_preprocessor_agent.py tests/test_native_rag_agent.py tests/test_multi_hop.py tests/test_map_reduce_agent.py tests/test_web_search_agent.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add src/chimera_rag/plugins/colla_rag/agents/ tests/test_preprocessor_agent.py tests/test_native_rag_agent.py tests/test_multi_hop.py tests/test_map_reduce_agent.py tests/test_web_search_agent.py
git commit -m "feat: add 5 specialized agents — Preprocessor, SingleHop, MultiHop, Summarization, Other (Task 7)"
```

---

### Task 8: Orchestrator & MultiAgentRetriever Facade

**Files:**
- Create: `src/chimera_rag/plugins/colla_rag/orchestrator.py`
- Modify: `src/chimera_rag/plugins/colla_rag/__init__.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `TreeIntentClassifier` (Task 2), `AgentFactory` (Task 6), all agents (Task 7), `MemoryManager` (Task 5), `BM25Index` + `ToolAccumulator` (Task 3), all `build_*_tools` functions (Tasks 3-4)
- Produces: `MultiAgentRetriever.retrieve(query, top_k) -> RetrievalResult` registered as `("retriever", "colla_rag.multi_agent")`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
"""Tests for MultiAgentRetriever orchestrator."""
import numpy as np
import pytest
from chimera_rag.core.types import Answer, Chunk, Query, RetrievalResult, Triple


class MockLLM:
    async def generate(self, prompt: str, **kw) -> str:
        if "classify" in prompt.lower() or "category" in prompt.lower():
            return "single_hop"
        return "Mock answer based on evidence."


class MockVectorStore:
    def search(self, qvec, top_k=5):
        return [("c1", 0.9)]

    def add(self, ids, vecs):
        pass


class MockGraphStore:
    def get_neighbors(self, entity, hops=1):
        return []

    def all_triples(self):
        return []


class MockEmbedder:
    def encode(self, texts):
        return np.random.randn(len(texts), 64).astype(np.float32)


def test_orchestrator_creation():
    from chimera_rag.plugins.colla_rag.orchestrator import MultiAgentRetriever
    chunks = {"c1": Chunk(chunk_id="c1", doc_id="d1", index=0, text="Test chunk")}
    r = MultiAgentRetriever(
        llm=MockLLM(),
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
    )
    assert r is not None


def test_orchestrator_greeting():
    from chimera_rag.plugins.colla_rag.orchestrator import MultiAgentRetriever
    chunks = {"c1": Chunk(chunk_id="c1", doc_id="d1", index=0, text="Test")}
    r = MultiAgentRetriever(
        llm=MockLLM(),
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
    )
    result = r.retrieve(Query(text="你好"), top_k=5)
    assert isinstance(result, RetrievalResult)


def test_orchestrator_single_hop():
    from chimera_rag.plugins.colla_rag.orchestrator import MultiAgentRetriever
    chunks = {"c1": Chunk(chunk_id="c1", doc_id="d1", index=0, text="Einstein was born in Ulm")}
    r = MultiAgentRetriever(
        llm=MockLLM(),
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
    )
    result = r.retrieve(Query(text="Where was Einstein born?"), top_k=5)
    assert isinstance(result, RetrievalResult)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement orchestrator.py**

```python
# src/chimera_rag/plugins/colla_rag/orchestrator.py
"""MultiAgentRetriever — top-level facade (Facade pattern)."""
from __future__ import annotations

import hashlib
import logging

from chimera_rag.core.registry import register
from chimera_rag.core.types import Answer, Chunk, Query, RetrievalResult
from chimera_rag.defaults.extractor import _run_async
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.interfaces.retriever import BaseRetriever
from chimera_rag.plugins.colla_rag.agent_factory import AgentFactory
from chimera_rag.plugins.colla_rag.classifier import TreeIntentClassifier
from chimera_rag.plugins.colla_rag.memory.memory_manager import MemoryManager
from chimera_rag.plugins.colla_rag.tools.retrieval_tools import (
    BM25Index,
    ToolAccumulator,
    build_retrieval_tools,
)

logger = logging.getLogger(__name__)


@register("retriever", "colla_rag.multi_agent")
class MultiAgentRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        llm,
        vector_store,
        graph_store,
        embedder,
        chunk_lookup: dict[str, Chunk],
        memory_path: str = ".chimera_memory",
        classifier_config: dict | None = None,
        agent_configs: dict | None = None,
        reflector=None,
        reranker=None,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.chunk_lookup = chunk_lookup
        self.reflector = reflector
        self.reranker = reranker
        self.agent_configs = agent_configs or {}

        self.classifier = TreeIntentClassifier(llm=llm, **(classifier_config or {}))
        self.memory = MemoryManager(base_path=memory_path)
        self.bm25_index = BM25Index.from_chunks(chunk_lookup)

        # Import all agents to trigger @AgentFactory.register side effects
        import chimera_rag.plugins.colla_rag.agents  # noqa: F401

    def _build_tools(self, accumulator: ToolAccumulator) -> list:
        from chimera_rag.plugins.colla_rag.tools.io_tools import build_io_tools
        from chimera_rag.plugins.colla_rag.tools.memory_tools import build_memory_tools
        from chimera_rag.plugins.colla_rag.tools.preprocess_tools import build_preprocess_tools
        from chimera_rag.plugins.colla_rag.tools.reasoning_tools import build_reasoning_tools
        from chimera_rag.plugins.colla_rag.tools.web_tools import build_web_tools

        all_tools = []
        all_tools.extend(build_retrieval_tools(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
            chunk_lookup=self.chunk_lookup,
            bm25_index=self.bm25_index,
            accumulator=accumulator,
        ))
        all_tools.extend(build_preprocess_tools(llm=self.llm, session_memory=self.memory.session))
        all_tools.extend(build_reasoning_tools(
            llm=self.llm,
            reflector=self.reflector or self._default_reflector(),
            reranker=self.reranker,
            accumulator=accumulator,
            chunk_lookup=self.chunk_lookup,
        ))
        all_tools.extend(build_memory_tools(
            shared_memory=self.memory.shared,
            agent_memory=self.memory.agent_private,
            session_memory=self.memory.session,
        ))
        all_tools.extend(build_web_tools())
        all_tools.extend(build_io_tools())
        return all_tools

    @staticmethod
    def _default_reflector():
        from chimera_rag.plugins.colla_rag.hooks.reflection_hook import HeuristicReflectionHook
        return HeuristicReflectionHook()

    def retrieve(self, query: Query, top_k: int = 5) -> RetrievalResult:
        # 1. Classify
        intent = self.classifier.classify(query)
        label = intent.label
        logger.info("query classified as %s (conf=%.2f)", label, intent.confidence)

        # 2. Greeting short-circuit
        if label == "greeting":
            greeting_text = _run_async(self.llm.generate(
                f"Respond warmly to this greeting: {query.text}", max_tokens=100,
            ))
            return RetrievalResult(
                metadata={"agent": "greeting", "intent": label, "answer_text": greeting_text.strip()},
            )

        # 3. QA cache check
        q_hash = hashlib.md5(query.text.encode()).hexdigest()
        cached = self.memory.shared.get_qa_cache(q_hash)
        if cached:
            return RetrievalResult(metadata={"agent": "cache", "intent": label, "answer_text": cached.get("answer", "")})

        # 4. Preprocess
        accumulator = ToolAccumulator()
        all_tools = self._build_tools(accumulator)
        preprocess_agent = AgentFactory.create(
            "preprocessor", all_tools=all_tools, memory_manager=self.memory, config={},
        )
        ctx = AgentContext(session_id=query.session_id, shared_memory=self.memory.shared,
                           agent_memory=self.memory.agent_private)
        pre_answer = preprocess_agent.execute(query, ctx)
        rewritten = pre_answer.text or query.text

        # 5. Dispatch to specialized agent
        agent_config = {"llm": self.llm, **self.agent_configs.get(label, {})}
        if label == "summarization":
            ctx.tool_context = {"chunk_lookup": self.chunk_lookup}
        agent = AgentFactory.create(label, all_tools=all_tools, memory_manager=self.memory, config=agent_config)
        answer = agent.execute(Query(text=rewritten, session_id=query.session_id), ctx)

        # 6. Post-process
        merged = accumulator.merged_result(top_k=top_k)
        merged.metadata["agent"] = label
        merged.metadata["intent"] = label
        merged.metadata["answer_text"] = answer.text
        merged.metadata["strategy"] = answer.strategy_name
        merged.metadata.update(answer.trace)

        # Write to memory
        self.memory.shared.write("routing_stats", q_hash, {"intent": label, "agent": label})
        if answer.trace.get("quality", 1.0) >= 0.6:
            self.memory.shared.write("qa_cache", q_hash, {"answer": answer.text})
        self.memory.shared.persist()

        self.memory.agent_private.remember(label, {
            "intent": label,
            "query_text": query.text,
            "tool_sequence": answer.trace.get("tool_call_log", []),
            "final_quality": answer.trace.get("quality", 0.0),
            "outcome": "success" if answer.trace.get("quality", 0.5) >= 0.5 else "lesson",
        })

        # Write session context
        if query.session_id:
            self.memory.session.append(query.session_id, f"Q: {query.text} | A: {answer.text[:200]}")

        return merged


__all__ = ["MultiAgentRetriever"]
```

- [ ] **Step 4: Update `plugins/colla_rag/__init__.py`**

```python
# src/chimera_rag/plugins/colla_rag/__init__.py
"""CollaRAG plugin — multi-agent architecture for intent-aware retrieval."""
# Trigger @register side effects for all slot implementations.
from chimera_rag.plugins.colla_rag import classifier as _classifier  # noqa: F401
from chimera_rag.plugins.colla_rag import orchestrator as _orchestrator  # noqa: F401
# Legacy registrations (kept for backward compatibility with existing configs)
from chimera_rag.plugins.colla_rag import intent_classifier as _intent_classifier  # noqa: F401
from chimera_rag.plugins.colla_rag import memory as _memory  # noqa: F401
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_orchestrator.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/chimera_rag/plugins/colla_rag/orchestrator.py src/chimera_rag/plugins/colla_rag/__init__.py tests/test_orchestrator.py
git commit -m "feat: add MultiAgentRetriever orchestrator facade (Task 8)"
```

---

### Task 9: Config Integration & End-to-End Pipeline Test

**Files:**
- Modify: `src/chimera_rag/core/config.py` (extend `CollaRAGConfig` for multi-agent params)
- Modify: `src/chimera_rag/__init__.py` (wire `MultiAgentRetriever` in `_instantiate_retriever`)
- Create: `configs/deepseek_multi_agent.yaml`
- Test: `tests/test_pipeline_multi_agent.py`

**Interfaces:**
- Consumes: `MultiAgentRetriever` from Task 8, `ChimeraRAG` facade, `AppConfig`
- Produces: End-to-end pipeline test proving `ChimeraRAG.query()` works with `colla_rag.multi_agent` retriever

- [ ] **Step 1: Write failing end-to-end test**

```python
# tests/test_pipeline_multi_agent.py
"""End-to-end test: ChimeraRAG with multi-agent retriever (MockProvider)."""
from chimera_rag import ChimeraRAG
from chimera_rag.core.config import load_config
from chimera_rag.core.types import Document, Query


def test_multi_agent_ingest_and_query(tmp_path):
    # Create a minimal config
    config_text = f"""
app:
  name: test
  version: "0.1"
  workspace_dir: "{tmp_path}"
logging:
  level: DEBUG
llm:
  provider: mock
  model: mock
embedding:
  provider: mock
  model: mock
  dim: 64
storage:
  graph:
    persist_path: "{tmp_path}/graph.gpickle"
  vector:
    persist_path: "{tmp_path}/vectors"
    index_type: flat_ip
ingestion:
  chunker:
    active: defaults.fixed
    params:
      chunk_size: 100
  extractor:
    active: defaults.simple_llm
    params: {{}}
  pruner:
    active: defaults.noop
    params: {{}}
query:
  intent_classifier:
    active: colla_rag.tree
    params: {{}}
  retriever:
    active: colla_rag.multi_agent
    params: {{}}
  generator:
    active: defaults.prompt
    params: {{}}
plugins:
  colla_rag:
    enabled: true
"""
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(config_text)

    crag = ChimeraRAG.from_config(str(cfg_path))
    docs = [
        Document(doc_id="d1", content="Albert Einstein was born in Ulm, Germany in 1879."),
        Document(doc_id="d2", content="Einstein developed the theory of relativity."),
    ]
    stats = crag.ingest(docs)
    assert stats["documents"] == 2

    answer = crag.query("Where was Einstein born?")
    assert answer.text  # non-empty
    assert answer.intent_label is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_pipeline_multi_agent.py -v`
Expected: FAIL — config or wiring errors

- [ ] **Step 3: Extend `CollaRAGConfig` in config.py**

Add `multi_agent` field to `CollaRAGConfig` in `src/chimera_rag/core/config.py`:

```python
class CollaRAGConfig(BaseModel):
    enabled: bool = False
    memory: dict[str, Any] = Field(default_factory=dict)
    intent_classifier: dict[str, Any] = Field(default_factory=dict)
    strategies: dict[str, Any] = Field(default_factory=dict)
    hooks: dict[str, Any] = Field(default_factory=dict)
    multi_agent: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Wire `MultiAgentRetriever` in `ChimeraRAG._instantiate_retriever`**

In `src/chimera_rag/__init__.py`, add handling for `MultiAgentRetriever` inside `_instantiate_retriever`. Add the following condition check before the existing `base_retriever` check:

```python
        # Multi-agent retriever — wires the full multi-agent orchestrator.
        if "orchestrator" in cls.__module__ or cls.__name__ == "MultiAgentRetriever":
            wiring = {
                "llm": self.llm,
                "vector_store": self.vector_store,
                "graph_store": self.graph_store,
                "embedder": self.embedder,
                "chunk_lookup": self.state.chunk_lookup,
            }
            ma_cfg = self.config.plugins.colla_rag.multi_agent
            if ma_cfg.get("memory", {}).get("shared", {}).get("path"):
                wiring["memory_path"] = ma_cfg["memory"]["shared"]["path"]
            if ma_cfg.get("classifier"):
                wiring["classifier_config"] = ma_cfg["classifier"]
            agent_configs = {}
            for key in ("native_rag", "react", "map_reduce", "web_search"):
                if key in ma_cfg:
                    agent_configs[{"native_rag": "single_hop", "react": "multi_hop",
                                   "map_reduce": "summarization", "web_search": "other"}[key]] = ma_cfg[key]
            wiring["agent_configs"] = agent_configs
            self._wire_reranker_into(wiring)
            self._wire_reflector_into(wiring)
            return cls(**wiring)
```

- [ ] **Step 5: Create `configs/deepseek_multi_agent.yaml`**

```yaml
app:
  name: chimera-rag
  version: "0.1.0"
  workspace_dir: .

logging:
  level: INFO
  format: rich

llm:
  provider: deepseek
  model: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
  base_url_env: DEEPSEEK_BASE_URL
  temperature: 0.1
  max_tokens: 1024
  timeout: 60.0

embedding:
  provider: ollama
  model: nomic-embed-text
  dim: 768
  batch_size: 32

storage:
  graph:
    backend: networkx
    persist_path: storage/graph.gpickle
  vector:
    backend: faiss
    index_type: hnsw
    persist_path: storage/vectors
    top_k: 10

ingestion:
  chunker:
    active: adagraph.dynamic
    params:
      min_chunk_size: 120
      max_chunk_size: 500
      base_chunk_size: 300
  extractor:
    active: adagraph.layered
    params:
      layers_enabled: [L1]
  pruner:
    active: adagraph.dual_redundancy
    params:
      similarity_threshold: 0.85

query:
  intent_classifier:
    active: colla_rag.tree
    params: {}
  retriever:
    active: colla_rag.multi_agent
    params: {}
  generator:
    active: defaults.prompt
    params: {}

plugins:
  adagraph:
    enabled: true
  colla_rag:
    enabled: true
    multi_agent:
      classifier:
        confidence_threshold: 0.7
        rule_fallback: true
      native_rag:
        top_k: 10
        quality_threshold: 0.4
      react:
        max_iterations: 3
        quality_threshold: 0.6
      map_reduce:
        retrieve_top_k: 50
      web_search:
        search_backend: duckduckgo
        max_results: 5
      memory:
        shared:
          path: .chimera_memory/shared
        agent_private:
          max_per_bucket: 200
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/test_pipeline_multi_agent.py -v`
Expected: PASS

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/ -v --timeout=60`
Expected: All existing tests still pass, no regressions

- [ ] **Step 8: Commit**

```bash
git add src/chimera_rag/core/config.py src/chimera_rag/__init__.py configs/deepseek_multi_agent.yaml tests/test_pipeline_multi_agent.py
git commit -m "feat: wire MultiAgentRetriever into ChimeraRAG facade + e2e test (Task 9)"
```

---

### Task 10: Clean Up Old Code & Update Documentation

**Files:**
- Delete: `src/chimera_rag/plugins/colla_rag/strategies/` (entire directory)
- Delete: `src/chimera_rag/plugins/colla_rag/langgraph/` (entire directory)
- Delete: `src/chimera_rag/plugins/colla_rag/agent.py`
- Delete: `src/chimera_rag/plugins/colla_rag/reasoning.py`
- Modify: `src/chimera_rag/plugins/colla_rag/__init__.py` (remove old imports)
- Modify: `docs/architecture_overview.md` (update to reflect multi-agent architecture)
- Modify: `README.md` (update usage instructions)

**Interfaces:**
- Consumes: none (cleanup task)
- Produces: clean codebase with no dead code from the old architecture

- [ ] **Step 1: Run full test suite to baseline**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/ -v --timeout=60`
Expected: All PASS — this is our baseline before deletion

- [ ] **Step 2: Delete old strategy files**

```bash
cd /Users/teng/Projects/my/Chimera-RAG
rm -rf src/chimera_rag/plugins/colla_rag/strategies/
rm -rf src/chimera_rag/plugins/colla_rag/langgraph/
rm -f src/chimera_rag/plugins/colla_rag/agent.py
rm -f src/chimera_rag/plugins/colla_rag/reasoning.py
```

- [ ] **Step 3: Update `__init__.py` to remove old imports**

Replace `src/chimera_rag/plugins/colla_rag/__init__.py` with:

```python
# src/chimera_rag/plugins/colla_rag/__init__.py
"""CollaRAG plugin — multi-agent architecture for intent-aware retrieval."""
from chimera_rag.plugins.colla_rag import classifier as _classifier  # noqa: F401
from chimera_rag.plugins.colla_rag import orchestrator as _orchestrator  # noqa: F401
from chimera_rag.plugins.colla_rag import intent_classifier as _intent_classifier  # noqa: F401
from chimera_rag.plugins.colla_rag import memory as _memory  # noqa: F401
from chimera_rag.plugins.colla_rag import experiential_memory as _exp_memory  # noqa: F401
```

- [ ] **Step 4: Remove or update tests that reference deleted code**

Find and remove tests that import from deleted modules:

```bash
cd /Users/teng/Projects/my/Chimera-RAG
grep -rl "from chimera_rag.plugins.colla_rag.strategies" tests/ | head -10
grep -rl "from chimera_rag.plugins.colla_rag.langgraph" tests/ | head -10
grep -rl "from chimera_rag.plugins.colla_rag.agent import" tests/ | head -10
grep -rl "from chimera_rag.plugins.colla_rag.reasoning" tests/ | head -10
```

For each file found: either delete the test file (if it only tests deleted code) or update imports to point to new equivalents.

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/ -v --timeout=60`
Expected: All PASS — no broken imports, no regressions

- [ ] **Step 6: Update architecture_overview.md**

Add a new §9 section to `docs/architecture_overview.md` describing the multi-agent architecture (replace the old single-retriever description). Keep the update brief — reference the design spec at `docs/innovation-2/design.md` for full details.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove old strategies/langgraph, update docs for multi-agent (Task 10)"
```

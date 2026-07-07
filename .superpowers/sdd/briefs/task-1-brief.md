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


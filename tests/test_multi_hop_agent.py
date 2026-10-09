"""Tests for MultiHopAgent (Plan-Execute-Reflect)."""

from chimera_rag.core.types import Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.multi_hop import MultiHopAgent


def test_multi_hop_agent_creates():
    class FakeLLM:
        async def complete(self, prompt, **kw):
            return "Step Result: The answer is 42."

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_plan_steps": 3, "quality_threshold": 0.6},
    )
    assert agent.agent_type == "multi_hop"


def test_multi_hop_agent_executes():
    call_count = 0

    class FakeLLM:
        async def complete(self, prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '[{"step": "Find the answer", "tools_hint": []}]'
            if "Step Result" in prompt or "step" in prompt.lower():
                return "Step Result: The answer is 42."
            return "The answer is 42."

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="What is 6*7?"), ctx)
    assert "42" in answer.text
    assert answer.strategy_name == "plan_execute"
    assert "plan" in ctx.trace


def test_multi_hop_plan_fallback():
    """When LLM returns unparseable plan, should fall back to single-step."""

    class FakeLLM:
        async def complete(self, prompt, **kw):
            if "planning" in prompt.lower() or "plan" in prompt.lower():
                return "I cannot generate a plan."
            return "Step Result: Fallback answer."

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="complex query"), ctx)
    assert len(answer.text) > 0
    assert len(ctx.trace["plan"]) == 1


# ---------------------------------------------------------------------------
# G-PER (Graph-Grounded Plan-Execute-Reflect) tests
# ---------------------------------------------------------------------------
def test_gper_plan_validates_against_schema():
    """Plan steps whose relation is not in the KG schema are degraded, not dropped."""
    from chimera_rag.core.types import Triple
    from chimera_rag.stores.graph import NetworkXGraphStore

    gs = NetworkXGraphStore()
    gs.add_triple(Triple(subject="Tokyo", predicate="population", object="14M"))

    class FakeLLM:
        async def complete(self, prompt, **kw):
            # Plan references a relation "population" (in schema) and a bogus
            # relation "color" (not in schema) plus an unknown entity.
            return (
                '[{"op": "attribute", "entity": "Tokyo", "relation": "population", '
                '"target": null, "step": "pop of Tokyo", "tools_hint": [], "depends_on": []}, '
                '{"op": "traverse", "entity": "Atlantis", "relation": "color", '
                '"target": null, "step": "color of Atlantis", "tools_hint": [], "depends_on": []}]'
            )

    agent = MultiHopAgent(
        agent_type="multi_hop", tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1, "graph_store": gs},
    )
    plan, stats, source = agent._plan(FakeLLM(), "x", "", 5, gs)
    assert source == "llm"
    # First step valid (relation in schema, entity exists).
    # Second step degraded (relation not in schema AND entity unknown).
    assert stats["valid"] == 1
    assert stats["degraded"] == 1
    assert len(plan) == 2
    assert "_degraded_from" in plan[1]


def test_gper_plan_degrades_without_graph_store():
    """With no graph store, plan validation is skipped (graph-poor fallback)."""
    class FakeLLM:
        async def complete(self, prompt, **kw):
            return '[{"op": "attribute", "entity": "X", "relation": "r", "target": null, "step": "s", "tools_hint": [], "depends_on": []}]'

    agent = MultiHopAgent(
        agent_type="multi_hop", tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1},
    )
    _plan, stats, source = agent._plan(FakeLLM(), "q", "", 5, None)
    assert stats["valid"] == 1
    assert stats["degraded"] == 0
    assert source == "llm"


def test_structural_completeness_detects_gaps():
    from chimera_rag.core.types import Triple
    from chimera_rag.plugins.colla_rag.gper import StructuralCompletenessChecker

    checker = StructuralCompletenessChecker(graph_store=None)
    # Comparison query: only one entity has an attribute -> gap for the other.
    gaps = checker.check(
        "Compare Tokyo and New York",
        ["Tokyo", "New York"],
        [Triple(subject="Tokyo", predicate="population", object="14M")],
    )
    assert len(gaps) == 1
    assert gaps[0].kind == "missing_attribute"
    assert gaps[0].repair_entities == ["New York"]

    # Unknown query (no entities) -> no structural gaps.
    assert checker.check("hello", [], []) == []


def test_gper_adversarial_parse_failure_is_safe():
    """A non-JSON adversarial response must not crash; defaults to 'confirmed'."""
    class FakeLLM:
        async def complete(self, prompt, **kw):
            return "I cannot assess this."

    from chimera_rag.core.types import Triple
    agent = MultiHopAgent(
        agent_type="multi_hop", tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1},
    )
    verdict = agent._adversarial_check(
        FakeLLM(), "q", "draft", [Triple(subject="a", predicate="r", object="b")],
    )
    assert verdict["verdict"] == "confirmed"


def test_gper_trace_carries_graph_grounded_signals():
    """End-to-end run records plan_stats, evidence_triples, adversarial_verdict."""
    class FakeLLM:
        def __init__(self):
            self.n = 0
        async def complete(self, prompt, **kw):
            self.n += 1
            if self.n == 1:
                return '[{"op": "attribute", "entity": "Tokyo", "relation": "population", "target": null, "step": "pop", "tools_hint": [], "depends_on": []}]'
            return "Step Result: 14 million people."

    from chimera_rag.core.types import Query
    from chimera_rag.interfaces.base_agent import AgentContext
    agent = MultiHopAgent(
        agent_type="multi_hop", tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="What is the population of Tokyo?"), ctx)
    assert "14" in answer.text
    assert answer.strategy_name == "plan_execute"
    assert "plan_stats" in ctx.trace
    assert "evidence_triples" in ctx.trace
    assert "adversarial_verdict" in ctx.trace


# ---------------------------------------------------------------------------
# C3: Graph-Grounded Memory — plan template recall + gap lesson pre-emption
# ---------------------------------------------------------------------------
def test_c3_plan_template_seeds_plan_and_skips_llm():
    """When a matching plan template exists, _plan reuses it (source=template)."""
    import os
    import tempfile

    from chimera_rag.plugins.colla_rag.memory import PlanTemplateStore

    with tempfile.TemporaryDirectory() as d:
        store = PlanTemplateStore(path=os.path.join(d, "pt.json"))
        store.store(
            compositionality="comparison",
            entities=["Tokyo", "New York"],
            plan=[{"step": "template step", "tools_hint": []}],
            quality=0.9,
        )

        class FakeLLM:
            calls = 0
            async def complete(self, prompt, **kw):
                FakeLLM.calls += 1
                return "should not be called"

        agent = MultiHopAgent(
            agent_type="multi_hop", tools=[],
            config={"llm": FakeLLM(), "max_step_iterations": 1},
        )
        plan, _stats, source = agent._plan(
            FakeLLM(), "Compare Tokyo and New York", "", 5, None,
            plan_templates=store, compositionality="comparison",
            entities=["Tokyo", "New York"],
        )
        assert source == "template"
        assert plan[0]["step"] == "template step"
        assert FakeLLM.calls == 0  # LLM plan call skipped


def test_c3_gap_lessons_pre_emptive_retrieval_runs():
    """Recalled gap lessons trigger pre-emptive repair retrieval before Reflect."""
    import os
    import tempfile

    from chimera_rag.plugins.colla_rag.memory import GapLessonStore

    with tempfile.TemporaryDirectory() as d:
        store = GapLessonStore(path=os.path.join(d, "gl.json"))
        store.store(
            entities=["Tokyo", "Yen"],
            gaps=[{"kind": "missing_path", "detail": "no path Tokyo->Yen",
                   "repair_query": "How is Tokyo related to Yen?",
                   "repair_entities": ["Tokyo", "Yen"]}],
        )

        agent = MultiHopAgent(
            agent_type="multi_hop", tools=[],
            config={"llm": _StubLLM(), "max_step_iterations": 1,
                    "quality_threshold": 0.6, "gap_lessons": store},
        )
        # _recall_gap_lessons should surface the stored gap for overlapping entities.
        recalled = agent._recall_gap_lessons(store, ["Tokyo", "Yen"])
        assert len(recalled) == 1
        assert recalled[0].kind == "missing_path"
        assert "Yen" in recalled[0].repair_entities


class _StubLLM:
    """LLM stub that returns parseable plan + step results."""
    def __init__(self):
        self.n = 0
    async def complete(self, prompt, **kw):
        self.n += 1
        if "planner" in prompt.lower() or "graph-grounded retrieval planner" in prompt:
            return '[{"op":"attribute","entity":"Tokyo","relation":null,"target":null,"step":"find","tools_hint":[],"depends_on":[]}]'
        return "Step Result: 14 million."


def test_c3_trace_carries_compositionality_and_gaps_for_persistence():
    """End-to-end MultiHop run records compositionality + structural_gaps in trace."""
    from chimera_rag.core.types import Query
    from chimera_rag.interfaces.base_agent import AgentContext
    agent = MultiHopAgent(
        agent_type="multi_hop", tools=[],
        config={"llm": _StubLLM(), "max_step_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)
    agent.execute(Query(text="Compare Tokyo and New York"), ctx)
    assert ctx.trace.get("compositionality") == "comparison"
    assert "structural_gaps" in ctx.trace
    assert ctx.trace.get("plan_source") in ("llm", "template")


def test_multi_hop_plan_consumes_recalled_lessons():
    """Recalled failure lessons are injected into the Plan prompt and traced."""
    seen_prompts: list[str] = []

    class FakeLLM:
        async def complete(self, prompt, **kw):
            seen_prompts.append(prompt)
            return '[{"step": "Find the answer", "tools_hint": []}]'

    class FakeAgentMemory:
        def recall_lessons(self, agent_type, intent, top_n=3):
            return [{
                "outcome": "lesson", "tool_sequence": ["vector_search"],
                "final_quality": 0.2,
                "lesson": "vector_search alone insufficient for multi-hop",
            }]

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None, agent_memory=FakeAgentMemory())
    agent.execute(Query(text="Compare A and B"), ctx)

    # The plan prompt (first LLM call) carries the avoidance guidance.
    assert "avoid:" in seen_prompts[0]
    assert "vector_search alone insufficient" in seen_prompts[0]
    # And the count is recorded in the trace.
    assert ctx.trace.get("avoided_lessons") == 1


def test_multi_hop_plan_unseeded_without_memory():
    """With no agent_memory, the planner runs unseeded (graceful degradation)."""
    seen_prompts: list[str] = []

    class FakeLLM:
        async def complete(self, prompt, **kw):
            seen_prompts.append(prompt)
            return '[{"step": "Find the answer", "tools_hint": []}]'

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)  # agent_memory defaults to None
    agent.execute(Query(text="What is X?"), ctx)

    assert "avoid:" not in seen_prompts[0]
    assert ctx.trace.get("avoided_lessons") == 0

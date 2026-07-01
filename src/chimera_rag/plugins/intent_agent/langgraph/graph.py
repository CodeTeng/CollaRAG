"""LangGraph StateGraph orchestrating innovation 2b's agentic retrieval.

Maps Chimera-RAG's Memory -> Reasoning -> Decision -> Execution stages onto
a LangGraph ``StateGraph`` while replacing the *Decision* stage's static
strategy lookup with an LLM-driven tool-selection loop, plus an explicit
**reflection** loop that the original IntentAgent only stubbed out.

Topology::

    START -> memory -> reasoning -> decision
                                      |  ^
                                      v  | (re-plan if low quality)
                                    tools |
                                      |   |
                                      v   |
                                    reflect --(quality>=tau or iters>=max)--> END

* ``memory_node``    reads session history into the conversation.
* ``reasoning_node`` runs the existing intent classifier as a *hint*.
* ``decision_node``  asks the ReAct ChatModel which tool to call next
  (or to give a final answer).
* ``tool_node``      executes the chosen tool (LangGraph ``ToolNode``).
* ``reflect_node``   scores accumulated evidence; the conditional edge
  decides whether to stop or re-plan, bounded by ``max_iterations``.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from chimera_rag.core.types import Intent
from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier
from chimera_rag.interfaces.memory import BaseMemory
from chimera_rag.interfaces.reflector import BaseReflector
from chimera_rag.plugins.intent_agent.langgraph.llm_adapter import ChimeraReActChatModel
from chimera_rag.plugins.intent_agent.langgraph.tools import ToolContext


class AgentState(TypedDict, total=False):
    """Shared state threaded through the graph (LangGraph reducer-managed).

    ``reflection_text`` carries the verbal self-critique from the reflect
    node back into the decision node, enabling Reflexion-style
    self-correction: the LLM reads its own reflection and adjusts tool
    selection on the next iteration.
    """

    query: str
    session_id: str | None
    intent: str
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
    quality: float
    tool_call_log: list[str]
    reflection_text: str | None


def build_agent_graph(
    *,
    chat_model: ChimeraReActChatModel,
    tools: list,
    context: ToolContext,
    classifier: BaseIntentClassifier,
    reflector: BaseReflector,
    memory: BaseMemory | None = None,
    quality_threshold: float = 0.6,
    max_iterations: int = 2,
):
    """Compile and return the agentic retrieval graph.

    The returned object is a compiled LangGraph runnable; call
    ``graph.invoke({...initial AgentState...})``.
    """
    from chimera_rag.core.types import Query
    from chimera_rag.defaults.llm_extractor import _run_async

    model_with_tools = chat_model.bind_tools(tools)
    tool_node = ToolNode(tools)

    # -- memory_node ------------------------------------------------------
    def memory_node(state: AgentState) -> dict[str, Any]:
        messages: list[BaseMessage] = []
        history: list[str] = []
        if memory is not None and state.get("session_id"):
            history = memory.get_history(state["session_id"])
        if history:
            messages.append(
                SystemMessage(content="Prior conversation turns: " + " | ".join(history))
            )
        messages.append(HumanMessage(content=state["query"]))
        return {"messages": messages, "iteration_count": 0, "tool_call_log": []}

    # -- reasoning_node ---------------------------------------------------
    def reasoning_node(state: AgentState) -> dict[str, Any]:
        intent: Intent = classifier.classify(Query(text=state["query"]))
        hint = SystemMessage(
            content=(
                f"Detected intent: {intent.label} (confidence {intent.confidence:.2f}). "
                "Use it only as a hint when choosing tools."
            )
        )
        return {"intent": intent.label, "messages": [hint]}

    # -- decision_node ----------------------------------------------------
    def decision_node(state: AgentState) -> dict[str, Any]:
        response: AIMessage = model_with_tools.invoke(state["messages"])
        log = list(state.get("tool_call_log", []))
        for call in response.tool_calls or []:
            log.append(call["name"])
        return {"messages": [response], "tool_call_log": log}

    # -- reflect_node -----------------------------------------------------
    def reflect_node(state: AgentState) -> dict[str, Any]:
        from chimera_rag.core.types import Answer

        merged = context.merged_result()
        draft = Answer(
            text=" ".join(c.text for c in merged.chunks),
            evidence_chunk_ids=[c.chunk_id for c in merged.chunks],
        )
        feedback = _run_async(reflector.evaluate(Query(text=state["query"]), draft))

        # Innovation #2d — Self-Reflection: if the reflector produced a
        # verbal critique (reflection_text), inject it as a SystemMessage
        # so the next decision_node round can read it and adjust tool
        # selection. This is the Reflexion-style self-correction mechanism.
        extra_messages: list[BaseMessage] = []
        reflection_text = feedback.reflection_text
        if reflection_text and feedback.quality < quality_threshold:
            extra_messages.append(
                SystemMessage(
                    content=(
                        f"Self-Reflection (quality={feedback.quality:.2f}): {reflection_text}"
                    )
                )
            )

        return {
            "quality": float(feedback.quality),
            "iteration_count": state.get("iteration_count", 0) + 1,
            "reflection_text": reflection_text,
            "messages": extra_messages,
        }

    # -- conditional edges ------------------------------------------------
    def route_after_decision(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "reflect"

    def route_after_reflect(state: AgentState) -> str:
        if state.get("quality", 0.0) >= quality_threshold:
            return END
        if state.get("iteration_count", 0) >= max_iterations:
            return END
        return "decision"

    # -- assemble ---------------------------------------------------------
    builder = StateGraph(AgentState)
    builder.add_node("memory", memory_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("decision", decision_node)
    builder.add_node("tools", tool_node)
    builder.add_node("reflect", reflect_node)

    builder.add_edge(START, "memory")
    builder.add_edge("memory", "reasoning")
    builder.add_edge("reasoning", "decision")
    builder.add_conditional_edges(
        "decision", route_after_decision, {"tools": "tools", "reflect": "reflect"}
    )
    builder.add_edge("tools", "reflect")
    builder.add_conditional_edges(
        "reflect", route_after_reflect, {"decision": "decision", END: END}
    )

    return builder.compile()


__all__ = ["AgentState", "build_agent_graph"]

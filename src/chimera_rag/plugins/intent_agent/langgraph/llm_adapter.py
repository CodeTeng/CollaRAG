"""Bridge Chimera-RAG's text-only ``BaseLLMProvider`` to a LangChain ChatModel.

Chimera-RAG already ships LLM providers (openai / deepseek / ollama / mock)
that all implement a single async method ``complete(prompt) -> str``. They do
**not** expose native function/tool calling. To drive a LangGraph agent we
therefore adapt them with a **ReAct text protocol**:

1. Render the bound tools + the running conversation into a single text
   prompt that instructs the model to reply with either::

       Action: <tool_name>
       Action Input: {"json": "args"}

   or, when it has enough evidence::

       Final Answer: <the answer>

2. Call the underlying ``provider.complete(prompt)``.
3. Parse the text back into a LangChain :class:`AIMessage` — populating
   ``tool_calls`` when an Action is requested, or returning plain content
   for a Final Answer.

This keeps the adapter the *only* place that knows about tool-calling, lets
the existing MockProvider drive deterministic tests (it just returns canned
ReAct text), and avoids pulling in a second LLM client such as
``langchain-openai``.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from chimera_rag.defaults.llm_extractor import _run_async
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

_ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z_][\w]*)", re.I)
_ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(\{.*?\}|\[.*?\]|.+)", re.I | re.DOTALL)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.I | re.DOTALL)


def _render_tools(tools: list[BaseTool]) -> str:
    lines = []
    for tool in tools:
        schema_hint = ""
        if tool.args_schema is not None:
            try:
                fields = tool.args_schema.model_json_schema().get("properties", {})
                schema_hint = ", ".join(fields.keys())
            except Exception:
                schema_hint = ""
        lines.append(f"- {tool.name}({schema_hint}): {tool.description}")
    return "\n".join(lines)


def _render_messages(messages: list[BaseMessage]) -> str:
    rendered = []
    for msg in messages:
        role = getattr(msg, "type", msg.__class__.__name__)
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        rendered.append(f"[{role}] {content}")
    return "\n".join(rendered)


_REACT_SYSTEM = """You are a retrieval-planning agent for a knowledge-graph RAG system.
Given a user question, decide which retrieval tool to call next, or give a final answer.

Available tools:
{tools}

Respond in EXACTLY one of these two formats and nothing else:

Action: <one of the tool names above>
Action Input: <a single-line JSON object of arguments>

OR, once the gathered evidence is sufficient:

Final Answer: <a short answer grounded in the retrieved evidence>

Conversation so far:
{conversation}
"""


def _parse_action_input(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    # Greedy outermost-brace slice for robustness against trailing text.
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Fallback: treat the whole thing as a 'query' argument.
    first_line = raw.splitlines()[0].strip() if raw else ""
    return {"query": first_line} if first_line else {}


class ChimeraReActChatModel(BaseChatModel):
    """LangChain ChatModel backed by a Chimera-RAG ``BaseLLMProvider``.

    Uses a ReAct text protocol so providers without native tool-calling
    (including the deterministic MockProvider) can still drive the agent.
    """

    provider: BaseLLMProvider
    bound_tools: list[BaseTool] = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "chimera_react"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> ChimeraReActChatModel:
        """Return a copy of this model that knows about ``tools``.

        LangGraph calls this to give the model its tool catalogue. We coerce
        any callables/StructuredTools into :class:`BaseTool` instances.
        """
        from langchain_core.tools import StructuredTool

        coerced: list[BaseTool] = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                coerced.append(tool)
            elif callable(tool):
                coerced.append(StructuredTool.from_function(tool))
        return self.__class__(provider=self.provider, bound_tools=coerced)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _REACT_SYSTEM.format(
            tools=_render_tools(self.bound_tools) or "(no tools)",
            conversation=_render_messages(messages),
        )
        raw = _run_async(self.provider.complete(prompt))
        ai_message = self._parse_response(raw)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ------------------------------------------------------------------
    def _parse_response(self, raw: str) -> AIMessage:
        """Map a ReAct text reply into an AIMessage (tool_calls or content)."""
        final = _FINAL_RE.search(raw)
        action = _ACTION_RE.search(raw)

        # A Final Answer takes precedence only if no Action precedes it.
        if final and (action is None or final.start() < action.start()):
            # If an Action block follows the Final Answer, slice the answer
            # text up to that block so trailing tool syntax does not leak in.
            answer_start = final.end(0) - len(final.group(1))
            if action is not None and action.start() > final.start():
                answer = raw[answer_start : action.start()]
            else:
                answer = final.group(1)
            return AIMessage(content=answer.strip())

        if action:
            tool_name = action.group(1).strip()
            args_match = _ACTION_INPUT_RE.search(raw, action.end())
            args = _parse_action_input(args_match.group(1)) if args_match else {}
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    }
                ],
            )

        # Neither a clean Action nor Final Answer — treat the raw text as the
        # answer so the graph can still terminate gracefully.
        return AIMessage(content=raw.strip())


__all__ = ["ChimeraReActChatModel"]

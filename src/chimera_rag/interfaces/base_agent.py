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

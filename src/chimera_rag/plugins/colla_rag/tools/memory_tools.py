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
            description="Read from shared memory (QA cache, entity aliases, etc).",
            args_schema=_ReadSharedArgs),
        StructuredTool.from_function(_write_shared_memory, name="write_shared_memory",
            description="Write to shared memory.", args_schema=_WriteSharedArgs),
        StructuredTool.from_function(_read_agent_memory, name="read_agent_memory",
            description="Read agent-private experience memory.", args_schema=_ReadAgentArgs),
        StructuredTool.from_function(_write_agent_memory, name="write_agent_memory",
            description="Write to agent-private experience memory.", args_schema=_WriteAgentArgs),
    ]


__all__ = ["build_memory_tools"]

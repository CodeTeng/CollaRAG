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

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
            return [
                {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")}
                for r in results
            ]
        except ImportError:
            logger.warning("duckduckgo-search not installed; returning empty results")
            return []
        except Exception as e:
            logger.warning("web_search failed: %s", e)
            return []

    def _web_fetch(url: str, max_chars: int = 3000) -> str:
        try:
            import re
            import urllib.request

            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
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

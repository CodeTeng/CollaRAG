"""Entity & relation extraction tools for the multi-agent architecture.

These tools bridge the gap between natural-language queries and the
knowledge graph: they extract entities and relations from the query
text so that graph retrieval tools (graph_neighbors, graph_path_search)
can operate on concrete entity names instead of relying on the LLM to
guess them.
"""
from __future__ import annotations

import json
import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chimera_rag.defaults.extractor import _run_async
from chimera_rag.plugins.colla_rag.prompts import (
    ENTITY_EXTRACT_PROMPT,
    ENTITY_LINK_PROMPT,
    RELATION_EXTRACT_PROMPT,
)

logger = logging.getLogger(__name__)


class _EntityExtractArgs(BaseModel):
    query: str = Field(description="Natural-language query to extract entities from.")


class _RelationExtractArgs(BaseModel):
    query: str = Field(description="Query to extract relations from.")


class _EntityLinkArgs(BaseModel):
    entities: list[str] = Field(description="Entities to link to the knowledge graph.")


def _parse_json_list(raw: str) -> list[str]:
    """Robustly parse a JSON object with a single key containing a list."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Greedy: find first { and last }
        lbr = raw.find("{")
        rbr = raw.rfind("}")
        if lbr == -1 or rbr == -1:
            return []
        try:
            data = json.loads(raw[lbr : rbr + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                return [str(v) for v in val]
    if isinstance(data, list):
        return [str(v) for v in data]
    return []


def build_entity_tools(*, llm, graph_store) -> list[StructuredTool]:
    """Build entity extraction, relation extraction, and entity linking tools.

    Args:
        llm: LLM provider for extraction prompts.
        graph_store: Knowledge graph store for entity linking (sampling known nodes).
    """

    def _entity_extract(query: str) -> dict:
        """Extract named entities and key concepts from a query using LLM."""
        prompt = ENTITY_EXTRACT_PROMPT.format(query=query)
        try:
            raw = _run_async(llm.complete(prompt, max_tokens=200))
        except Exception as e:
            logger.warning("entity_extract LLM call failed: %s", e)
            return {"tool": "entity_extract", "entities": [], "count": 0}

        entities = _parse_json_list(raw)
        # Deduplicate preserving order
        seen = set()
        unique = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique.append(e)
        return {"tool": "entity_extract", "entities": unique, "count": len(unique)}

    def _relation_extract(query: str) -> dict:
        """Extract subject-predicate-object relations from a query using LLM."""
        prompt = RELATION_EXTRACT_PROMPT.format(query=query)
        try:
            raw = _run_async(llm.complete(prompt, max_tokens=300))
        except Exception as e:
            logger.warning("relation_extract LLM call failed: %s", e)
            return {"tool": "relation_extract", "relations": []}

        relations = _parse_json_list(raw)
        # _parse_json_list returns list[str] for flat lists, but here we expect
        # list[dict]. Parse again from the raw JSON.
        try:
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                lines = raw_clean.split("\n")
                raw_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            data = json.loads(raw_clean)
        except json.JSONDecodeError:
            lbr = raw.find("{")
            rbr = raw.rfind("}")
            if lbr != -1 and rbr != -1:
                try:
                    data = json.loads(raw[lbr : rbr + 1])
                except json.JSONDecodeError:
                    return {"tool": "relation_extract", "relations": []}
            else:
                return {"tool": "relation_extract", "relations": []}

        relations_list = []
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    relations_list = val
                    break
        elif isinstance(data, list):
            relations_list = data

        result = []
        for item in relations_list:
            if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                result.append({
                    "subject": str(item["subject"]),
                    "predicate": str(item["predicate"]),
                    "object": str(item["object"]),
                })
        return {"tool": "relation_extract", "relations": result, "count": len(result)}

    def _entity_link(entities: list[str]) -> dict:
        """Link extracted entity names to canonical names in the knowledge graph.

        Samples graph nodes (up to 500) and uses LLM to match extracted entities
        to canonical graph entity names. Falls back to substring matching on LLM failure.
        """
        if not entities:
            return {"tool": "entity_link", "mapping": {}, "matched": 0, "total": 0}

        # Sample known graph entities
        all_triples = graph_store.all_triples()
        graph_nodes: set[str] = set()
        for t in all_triples:
            graph_nodes.add(t.subject)
            graph_nodes.add(t.object)
            if len(graph_nodes) >= 500:
                break

        graph_entities_sample = sorted(graph_nodes)[:500]

        if not graph_entities_sample:
            # Empty graph: map everything to itself
            return {
                "tool": "entity_link",
                "mapping": {e: e for e in entities},
                "matched": 0,
                "total": len(entities),
            }

        prompt = ENTITY_LINK_PROMPT.format(
            entities=json.dumps(entities, ensure_ascii=False),
            graph_entities=json.dumps(graph_entities_sample, ensure_ascii=False),
        )
        try:
            raw = _run_async(llm.complete(prompt, max_tokens=400))
        except Exception as e:
            logger.warning("entity_link LLM call failed: %s", e)
            raw = ""

        mapping: dict[str, str | None] = {}
        try:
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                lines = raw_clean.split("\n")
                raw_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            data = json.loads(raw_clean)
            if isinstance(data, dict):
                mapping = {k: (v if v is not None else None) for k, v in data.items()}
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: substring match for entities not linked by LLM
        for entity in entities:
            if entity not in mapping:
                # Try exact match (case-insensitive)
                matches = [n for n in graph_entities_sample if n.lower() == entity.lower()]
                if matches:
                    mapping[entity] = matches[0]
                else:
                    # Try substring match
                    matches = [n for n in graph_entities_sample if entity.lower() in n.lower()]
                    if len(matches) == 1:
                        mapping[entity] = matches[0]
                    elif len(matches) > 1:
                        # Pick the shortest matching name (most likely canonical)
                        mapping[entity] = min(matches, key=len)
                    else:
                        mapping[entity] = None

        matched = sum(1 for v in mapping.values() if v is not None)
        return {
            "tool": "entity_link",
            "mapping": mapping,
            "matched": matched,
            "total": len(entities),
        }

    return [
        StructuredTool.from_function(
            _entity_extract, name="entity_extract",
            description="Extract named entities and key concepts from a query. Use this FIRST before graph_neighbors or graph_path_search.",
            args_schema=_EntityExtractArgs,
        ),
        StructuredTool.from_function(
            _relation_extract, name="relation_extract",
            description="Extract explicit or implied relations (subject-predicate-object) from a query. Useful for understanding what the query is asking about.",
            args_schema=_RelationExtractArgs,
        ),
        StructuredTool.from_function(
            _entity_link, name="entity_link",
            description="Link extracted entity names to canonical names in the knowledge graph. Use after entity_extract to get the correct graph entity names.",
            args_schema=_EntityLinkArgs,
        ),
    ]


__all__ = ["build_entity_tools"]

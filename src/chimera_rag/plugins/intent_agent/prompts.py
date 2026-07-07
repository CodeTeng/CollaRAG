"""Centralized prompt templates for all agents and tools.

All prompts live here so they can be versioned, tuned, and translated
independently from the agent/tool implementation code.
"""

# ---------------------------------------------------------------------------
# Tree-Intent Classifier
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = """Classify the following query into exactly one category.
Categories:
- single_hop: simple factual question answerable from one passage
- multi_hop: requires reasoning across multiple facts, comparison, or exploration
- summarization: asks for a summary, overview, or synthesis of information
- other: does not fit any above category (e.g. weather, jokes, math)

Query: {query}

Respond with ONLY the category name, nothing else."""

# ---------------------------------------------------------------------------
# Preprocessor Agent
# ---------------------------------------------------------------------------
QUERY_REWRITE_PROMPT = """Rewrite the following query to be clearer, more specific, and self-contained.
Context: {context}
Original query: {query}
Rewritten query:"""

COREFERENCE_RESOLVE_PROMPT = """Resolve all pronouns and references in the query using the conversation history.
History: {history}
Query: {query}
Resolved query (replace pronouns with explicit entities):"""

FOLLOWUP_MERGE_PROMPT = """Rewrite the following follow-up question into a self-contained query.
Previous question: {prev_query}
Previous answer: {prev_answer}
Follow-up question: {query}
Self-contained query:"""

# ---------------------------------------------------------------------------
# Native RAG Agent
# ---------------------------------------------------------------------------
NATIVE_RAG_GENERATE_PROMPT = """Answer the question based on the following evidence.
Evidence:
{evidence}

Question: {query}
Answer:"""

# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------
REACT_SYSTEM_PROMPT = """You are a retrieval agent. You have access to these tools:
{tool_descriptions}

Use the following format:
Thought: reason about what to do next
Action: tool_name
Action Input: {{"param": "value"}}
... (repeat Thought/Action/Action Input as needed)
Final Answer: your final answer based on gathered evidence

Begin!

Question: {query}"""

# ---------------------------------------------------------------------------
# Map-Reduce Agent
# ---------------------------------------------------------------------------
MAP_PROMPT = """Extract key information relevant to the query from this passage.
Query: {query}
Passage: {text}
Key information:"""

REDUCE_PROMPT = """Based on the following extracted information, provide a comprehensive summary.
Query: {query}
Extracted information:
{summaries}
Summary:"""

# ---------------------------------------------------------------------------
# WebSearch + LLM Agent
# ---------------------------------------------------------------------------
WEBSEARCH_GENERATE_PROMPT = """Answer the question based on the following web search results and local knowledge.
Web results:
{web_results}

Local knowledge:
{local_knowledge}

Question: {query}
Answer:"""

# ---------------------------------------------------------------------------
# Reasoning Tools
# ---------------------------------------------------------------------------
SUB_QUERY_DECOMPOSE_PROMPT = """Decompose the following complex query into simpler sub-queries.
Return a JSON list of strings, e.g. ["sub-query 1", "sub-query 2"].
Query: {query}
Sub-queries:"""

# ---------------------------------------------------------------------------
# Entity & Relation Extraction Tools
# ---------------------------------------------------------------------------
ENTITY_EXTRACT_PROMPT = """Extract all named entities and key concepts from the query.
Return a JSON object with a single key "entities" containing a list of strings.

Query: {query}
JSON:"""

RELATION_EXTRACT_PROMPT = """Extract explicit or implied relationships between entities mentioned in the query.
Return a JSON object with a single key "relations" containing a list of objects,
each with keys "subject", "predicate", "object".

Query: {query}
JSON:"""

ENTITY_LINK_PROMPT = """Given a list of extracted entities and the knowledge graph's known entities,
match each extracted entity to its canonical form in the graph.
Return a JSON object mapping original entity names to canonical graph entity names.
If an entity is not found in the graph, map it to null.

Extracted entities: {entities}
Known graph entities (sample): {graph_entities}
JSON:"""

__all__ = [
    "CLASSIFY_PROMPT",
    "QUERY_REWRITE_PROMPT",
    "COREFERENCE_RESOLVE_PROMPT",
    "FOLLOWUP_MERGE_PROMPT",
    "NATIVE_RAG_GENERATE_PROMPT",
    "REACT_SYSTEM_PROMPT",
    "MAP_PROMPT",
    "REDUCE_PROMPT",
    "WEBSEARCH_GENERATE_PROMPT",
    "SUB_QUERY_DECOMPOSE_PROMPT",
    "ENTITY_EXTRACT_PROMPT",
    "RELATION_EXTRACT_PROMPT",
    "ENTITY_LINK_PROMPT",
]
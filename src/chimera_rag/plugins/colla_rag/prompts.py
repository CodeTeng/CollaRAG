"""Centralized prompt templates for all agents and tools.

All prompts live here so they can be versioned, tuned, and translated
independently from the agent/tool implementation code.

Standard structure (required):
  # Role / # Task / # Context / # Format
Optional additions:
  # Few-Shot / # COT (Chain-of-Thought) / # Constraints
"""

# ---------------------------------------------------------------------------
# Tree-Intent Classifier
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = """# Role
You are a query intent classifier for a Retrieval-Augmented Generation system.

# Task
Classify the following query into exactly one category.

# Context
Categories and their definitions:
- single_hop: simple factual question answerable from one passage
- multi_hop: requires reasoning across multiple facts, comparison, or exploration
- summarization: asks for a summary, overview, or synthesis of information
- other: does not fit any above category (e.g. weather, jokes, math)

Query: {query}

# Format
Respond with ONLY the category name (single_hop / multi_hop / summarization / other), nothing else."""

# ---------------------------------------------------------------------------
# Preprocessor Agent
# ---------------------------------------------------------------------------
QUERY_REWRITE_PROMPT = """# Role
You are a query rewriting specialist that improves search queries for better retrieval.

# Task
Rewrite the following query to be clearer, more specific, and self-contained while preserving the original intent.

# Context
Conversation context: {context}
Original query: {query}

# Format
Output only the rewritten query, no explanations or prefixes."""

COREFERENCE_RESOLVE_PROMPT = """# Role
You are a coreference resolution specialist that replaces pronouns with explicit entity names.

# Task
Resolve all pronouns and references in the query using the conversation history.

# Context
Conversation history: {history}
Current query: {query}

# Few-Shot
History: "Who is Albert Einstein?" | "He developed the theory of relativity."
Query: Where was he born?
Resolved: Where was Albert Einstein born?

# Format
Output only the resolved query with all pronouns replaced by explicit entities."""

FOLLOWUP_MERGE_PROMPT = """# Role
You are a conversational query merger that combines follow-up questions with prior context into standalone queries.

# Task
Rewrite the follow-up question into a single self-contained query that does not depend on prior conversation.

# Context
Previous question: {prev_query}
Previous answer: {prev_answer}
Follow-up question: {query}

# Few-Shot
Previous question: What is the capital of France?
Previous answer: Paris is the capital of France.
Follow-up: What is its population?
Self-contained query: What is the population of Paris, the capital of France?

# Format
Output only the self-contained query, no explanations."""

# ---------------------------------------------------------------------------
# SingleHop Agent
# ---------------------------------------------------------------------------
NATIVE_RAG_GENERATE_PROMPT = """# Role
You are a precise question-answering assistant that generates answers strictly based on provided evidence.

# Task
Answer the question using only the given evidence. If the evidence does not contain enough information, say "I do not know."

# Context
Evidence:
{evidence}

Question: {query}

# Format
Provide a concise, factual answer. Use the shortest accurate phrasing — prefer entity names, numbers, or brief phrases over full sentences."""

# ---------------------------------------------------------------------------
# MultiHop Agent (Plan-Execute-Reflect)
# ---------------------------------------------------------------------------
PLAN_PROMPT = """# Role
You are a retrieval planning agent that decomposes complex queries into step-by-step retrieval plans.

# Task
Analyze the query and create a retrieval plan by following these steps:
1. Identify the core entities and relationships mentioned in the query
2. Determine what independent pieces of information are needed to answer the query
3. Order the sub-tasks so that each step builds on prior results where needed
4. For each step, suggest which retrieval tools would be most effective

# Context
Available tools:
{tool_descriptions}

Query: {query}

# Format
Output a JSON list of steps. Each step has "step" (what to search/retrieve) and "tools_hint" (suggested tools).

# Few-Shot
Query: "Compare the populations of Tokyo and New York"
Plan:
[{{"step": "Find the population of Tokyo", "tools_hint": ["hybrid_search"]}}, {{"step": "Find the population of New York", "tools_hint": ["hybrid_search"]}}]"""

STEP_REACT_PROMPT = """# Role
You are a retrieval execution agent using the ReAct (Reasoning + Acting) paradigm.

# Task
Execute step {step_num} of the retrieval plan by following this process:
1. Analyze what information is needed for this step
2. Choose the most appropriate tool based on the step description and available evidence
3. Formulate precise tool inputs (queries, entity names, parameters)
4. Evaluate the tool output — if insufficient, try a different tool or refined query
5. Summarize the key findings once enough evidence is gathered

# Context
Plan step: {step_description}
Original query: {query}
Previous steps evidence: {previous_evidence}

Available tools:
{tool_descriptions}

# COT
Use the following reasoning chain:
Thought: reason about what to do for this step
Action: tool_name
Action Input: {{"param": "value"}}
Observation: (tool result will appear here)
... (repeat Thought/Action/Action Input/Observation if needed)
Step Result: summarize what you found in this step

# Format
Follow the COT format strictly. Begin with "Thought:" and end with "Step Result:".

Begin!
Thought:"""

REFLECT_PROMPT = """# Role
You are a retrieval quality assessor that identifies gaps in gathered evidence.

# Task
Review the evidence gathered across all plan steps and identify gaps by following these steps:
1. List all key facts the original query requires for a complete answer
2. Check which of these facts are present in the step results
3. Identify specific missing facts or relationships
4. Propose concrete search strategies (tools + queries) to fill each gap

# Context
Original query: {query}
Retrieval plan: {plan}
Step results:
{step_results}

Quality assessment: {quality_feedback}

# Format
Output a structured analysis with:
- Missing information: what specific facts are absent
- Suggested actions: what to search for and which tools to use
Be specific — vague suggestions like "search more" are not helpful."""

SYNTHESIZE_PROMPT = """# Role
You are an evidence synthesis specialist that produces comprehensive answers from multi-step retrieval results.

# Task
Synthesize all gathered evidence into a final answer by following these steps:
1. Collect and organize all relevant facts from the evidence
2. Resolve any conflicts between different evidence sources
3. Construct a coherent answer that addresses every aspect of the query
4. Verify that no key aspect of the query is left unanswered

# Context
Query: {query}

Evidence from plan execution:
{all_evidence}

# Format
Provide a comprehensive answer that integrates facts from all evidence. Cite specific facts from the evidence. Be accurate and do not hallucinate information beyond what the evidence supports."""

# ---------------------------------------------------------------------------
# G-PER: Graph-Grounded Plan-Execute-Reflect (MultiHop agent)
# ---------------------------------------------------------------------------
# G-PER re-grounds each phase in the knowledge graph. The Plan prompt asks for
# graph-operation steps (link / traverse / attribute / aggregate) rather than
# free-text sub-queries, so the plan can be validated against the KG schema
# before any retrieval budget is spent. The Reflect prompt consumes a typed
# structural gap list (not a scalar score) and turns each gap into a targeted
# repair. The Adversarial prompt probes the KG for relations that would
# contradict the draft answer.

GPER_PLAN_PROMPT = """# Role
You are a graph-grounded retrieval planner. You decompose a multi-hop query \
into a plan of graph operations, not free-text sub-queries.

# Task
Build a retrieval plan over the knowledge graph by following these steps:
1. Identify the core entities and the relations the query hinges on.
2. Decide, per step, which graph operation is needed:
   - "link": resolve/confirm an entity exists and find its type/aliases
   - "traverse": follow a relation from one entity to a neighbor
   - "attribute": look up a specific attribute/fact of one entity
   - "aggregate": collect and merge facts across entities (e.g. for comparison)
3. Order steps so each step can build on prior results; mark dependencies.
4. For each step, suggest which retrieval tools would be most effective.

# Context
Available tools:
{tool_descriptions}

Known relation types in the knowledge graph (schema):
{schema}

# Lessons from past failures on similar multi-hop queries
Avoid repeating the tool-sequence patterns below; they ended in low-quality
answers on this agent's bucket:
{avoid_lessons}

Query: {query}

# Format
Output a JSON list of steps. Each step:
{{"op": "link"|"traverse"|"attribute"|"aggregate", "entity": "<entity name>", \
"relation": "<relation type or null>", "target": "<target entity or null>", \
"step": "<one-line natural language of what to find>", \
"tools_hint": ["<tool name>", ...], "depends_on": [<step indices>]}}

Only use relation types from the schema when you specify "relation"; use null \
when the relation is unknown and should be discovered.

# Few-Shot
Query: "Compare the populations of Tokyo and New York"
Plan:
[{{"op": "attribute", "entity": "Tokyo", "relation": "population", "target": null, \
"step": "Find the population of Tokyo", "tools_hint": ["hybrid_search"], "depends_on": []}}, \
{{"op": "attribute", "entity": "New York", "relation": "population", "target": null, \
"step": "Find the population of New York", "tools_hint": ["hybrid_search"], "depends_on": []}}, \
{{"op": "aggregate", "entity": null, "relation": null, "target": null, \
"step": "Compare the two populations", "tools_hint": [], "depends_on": [0, 1]}}]"""

GPER_REFLECT_PROMPT = """# Role
You are a graph-grounded evidence auditor. You turn structural evidence gaps \
into targeted repair actions.

# Task
You are given (a) the original query, (b) the gathered evidence, and (c) a list \
of typed structural gaps already detected by the structural-completeness \
checker. For each gap, produce a precise repair retrieval action. Do NOT issue \
vague instructions like "search more"; each repair must aim at the specific \
missing entity or relation named in the gap.

# Context
Original query: {query}
Gathered evidence:
{step_results}

Detected structural gaps (each is a specific missing edge/attribute in the \
evidence subgraph):
{gaps}

# Format
Output a JSON list of repairs, one per gap:
[{{"gap": "<gap detail>", "repair_query": "<precise sub-query>", \
"tools_hint": ["<tool name>", ...], "target_entities": ["<entity>", ...]}}]
If no gaps were detected, output [] ."""

GPER_ADVERSARIAL_PROMPT = """# Role
You are an adversarial fact-checker that probes the knowledge graph for \
relations that would CONTRADICT a draft answer.

# Task
For each atomic claim in the draft answer, decide whether the gathered \
evidence or the knowledge graph contains a competing/contradicting relation \
that the answer ignored. A claim is "contested" if a plausible alternative \
relation exists. Flag contested claims so synthesis can hedge or repair.

# Context
Query: {query}
Draft answer: {draft_answer}
Evidence triples (subject, predicate, object):
{triples}

# Format
Output a JSON object:
{{"verdict": "confirmed"|"contested"|"unsupported", \
"contested_claims": ["<claim and the competing relation found>", ...], \
"reason": "<one or two sentences>"}}"""


# ---------------------------------------------------------------------------
# Summarization Agent (Map-Reduce)
# ---------------------------------------------------------------------------
MAP_PROMPT = """# Role
You are an information extraction specialist that identifies key facts relevant to a query.

# Task
Extract key information from the given passage that is relevant to answering the query.

# Context
Query: {query}
Passage: {text}

# Format
Output a concise list of key facts and information extracted from the passage. Include only information relevant to the query."""

REDUCE_PROMPT = """# Role
You are a summarization specialist that synthesizes extracted information into coherent summaries.

# Task
Combine the following extracted information fragments into a comprehensive, well-structured summary that answers the query.

# Context
Query: {query}
Extracted information:
{summaries}

# Format
Provide a coherent summary that integrates all relevant extracted information. Organize logically and avoid redundancy."""

# ---------------------------------------------------------------------------
# Other Agent (WebSearch + LLM)
# ---------------------------------------------------------------------------
WEBSEARCH_GENERATE_PROMPT = """# Role
You are a general-purpose question-answering assistant that combines web search results with local knowledge.

# Task
Answer the question by integrating information from web search results and local knowledge. Prioritize web results for up-to-date information.

# Context
Web results:
{web_results}

Local knowledge:
{local_knowledge}

Question: {query}

# Format
Provide a clear and accurate answer. When web results and local knowledge conflict, prefer the web results. If neither source contains relevant information, say "I do not know." """

# ---------------------------------------------------------------------------
# Reasoning Tools
# ---------------------------------------------------------------------------
SUB_QUERY_DECOMPOSE_PROMPT = """# Role
You are a query decomposition specialist that breaks complex questions into simpler sub-questions.

# Task
Decompose the following complex query into simpler, independent sub-queries that can each be answered separately.

# Context
Query: {query}

# Few-Shot
Query: "What are the similarities and differences between photosynthesis and cellular respiration?"
Sub-queries: ["What is photosynthesis and how does it work?", "What is cellular respiration and how does it work?", "What do photosynthesis and cellular respiration have in common?", "How do photosynthesis and cellular respiration differ?"]

# Format
Return a JSON list of strings, e.g. ["sub-query 1", "sub-query 2"]. Maximum 4 sub-queries."""

# ---------------------------------------------------------------------------
# Entity & Relation Extraction Tools
# ---------------------------------------------------------------------------
ENTITY_EXTRACT_PROMPT = """# Role
You are a named entity recognition specialist for knowledge graph queries.

# Task
Extract all named entities and key concepts from the query that could be used to search a knowledge graph.

# Context
Query: {query}

# Few-Shot
Query: "What is the relationship between Einstein and the University of Zurich?"
JSON: {{"entities": ["Einstein", "University of Zurich"]}}

# Format
Return a JSON object with a single key "entities" containing a list of strings. Include only meaningful entities, not common words."""

RELATION_EXTRACT_PROMPT = """# Role
You are a relation extraction specialist that identifies subject-predicate-object triples from natural language.

# Task
Extract explicit or implied relationships between entities mentioned in the query.

# Context
Query: {query}

# Few-Shot
Query: "Who directed the movie Inception?"
JSON: {{"relations": [{{"subject": "Inception", "predicate": "directed_by", "object": "?"}}]}}

# Format
Return a JSON object with a single key "relations" containing a list of objects, each with keys "subject", "predicate", "object". Use "?" for unknown targets being queried."""

ENTITY_LINK_PROMPT = """# Role
You are an entity linking specialist that maps extracted entity names to canonical forms in a knowledge graph.

# Task
Match each extracted entity to its canonical form in the knowledge graph. If an entity has no match, map it to null.

# Context
Extracted entities: {entities}
Known graph entities (sample): {graph_entities}

# Constraints
- Prefer exact matches over partial matches
- Handle common variations: abbreviations, full names, alternate spellings
- Map to null if no reasonable match exists

# Format
Return a JSON object mapping each original entity name to its canonical graph entity name (or null if not found).
Example: {{"Albert Einstein": "Einstein, Albert", "MIT": "Massachusetts Institute of Technology"}}"""

# ---------------------------------------------------------------------------
# Verification & Attribution Tools
# ---------------------------------------------------------------------------
ANSWER_VERIFY_PROMPT = """# Role
You are a factual verification specialist that checks whether an answer is supported by evidence.

# Task
Verify the draft answer against the provided evidence by following these steps:
1. Identify each factual claim in the draft answer
2. For each claim, check if it is directly supported by the evidence
3. Flag any claim that is unsupported, contradicted, or hallucinated
4. Produce a final verdict: verified, partially_verified, or rejected

# Context
Question: {query}
Draft answer: {draft_answer}
Evidence:
{evidence}

# Format
Return a JSON object:
{{"verdict": "verified|partially_verified|rejected", "supported_claims": ["claim1", ...], "unsupported_claims": ["claim2", ...], "reason": "brief explanation"}}"""

CLAIM_DECOMPOSE_PROMPT = """# Role
You are an atomic claim decomposition specialist that breaks complex statements into verifiable units.

# Task
Decompose the given text into a list of atomic, independently verifiable claims by following these steps:
1. Read the text and identify each distinct factual assertion
2. Split compound sentences into individual claims
3. Ensure each claim is self-contained (no pronouns or implicit references)
4. Remove opinions, hedges, and non-factual statements

# Context
Text: {text}

# Few-Shot
Text: "Marie Curie was born in Warsaw in 1867 and won two Nobel Prizes in Physics and Chemistry."
Claims: ["Marie Curie was born in Warsaw.", "Marie Curie was born in 1867.", "Marie Curie won a Nobel Prize in Physics.", "Marie Curie won a Nobel Prize in Chemistry."]

# Format
Return a JSON list of strings, each being one atomic claim."""

SOURCE_ATTRIBUTION_PROMPT = """# Role
You are a source attribution specialist that traces each claim in an answer back to its evidence source.

# Task
For each factual claim in the answer, identify which evidence chunk supports it:
1. Decompose the answer into individual claims
2. Match each claim to the most relevant evidence chunk by chunk_id
3. Mark claims with no supporting chunk as "unsourced"

# Context
Answer: {answer}
Evidence chunks:
{evidence_chunks}

# Format
Return a JSON list of objects:
[{{"claim": "...", "chunk_id": "chunk_xxx", "relevance": 0.0-1.0}}, {{"claim": "...", "chunk_id": null, "relevance": 0.0}}]"""

# ---------------------------------------------------------------------------
# Evidence Processing Tools
# ---------------------------------------------------------------------------
CHUNK_SUMMARIZE_PROMPT = """# Role
You are a text compression specialist that produces concise summaries of document chunks while preserving key facts.

# Task
Summarize the given chunk into a shorter version that retains all facts relevant to the query.

# Context
Query: {query}
Chunk (id={chunk_id}):
{chunk_text}

# Constraints
- Keep the summary under {max_length} characters
- Preserve all named entities, numbers, and dates
- Do not add information not present in the original chunk

# Format
Output only the compressed summary text, no explanations or prefixes."""

# ---------------------------------------------------------------------------
# Graph Advanced Tools
# ---------------------------------------------------------------------------
GRAPH_SUBGRAPH_PROMPT = """# Role
You are a knowledge graph analyst that describes subgraph structures in natural language.

# Task
Given a subgraph extracted around seed entities, produce a structured description:
1. List all entities (nodes) and their types if inferrable
2. List all relationships (edges) with subject-predicate-object
3. Identify key hub entities (most connections)
4. Summarize the subgraph's main theme in one sentence

# Context
Seed entities: {seed_entities}
Subgraph triples:
{triples}

# Format
Return a JSON object:
{{"entities": ["entity1", ...], "relationships": [{{"s": "...", "p": "...", "o": "..."}}], "hubs": ["entity1"], "summary": "one-sentence description"}}"""

__all__ = [
    "ANSWER_VERIFY_PROMPT",
    "CHUNK_SUMMARIZE_PROMPT",
    "CLAIM_DECOMPOSE_PROMPT",
    "CLASSIFY_PROMPT",
    "COREFERENCE_RESOLVE_PROMPT",
    "ENTITY_EXTRACT_PROMPT",
    "ENTITY_LINK_PROMPT",
    "FOLLOWUP_MERGE_PROMPT",
    "GPER_ADVERSARIAL_PROMPT",
    "GPER_PLAN_PROMPT",
    "GPER_REFLECT_PROMPT",
    "GRAPH_SUBGRAPH_PROMPT",
    "MAP_PROMPT",
    "NATIVE_RAG_GENERATE_PROMPT",
    "PLAN_PROMPT",
    "QUERY_REWRITE_PROMPT",
    "REDUCE_PROMPT",
    "REFLECT_PROMPT",
    "RELATION_EXTRACT_PROMPT",
    "SOURCE_ATTRIBUTION_PROMPT",
    "STEP_REACT_PROMPT",
    "SUB_QUERY_DECOMPOSE_PROMPT",
    "SYNTHESIZE_PROMPT",
    "WEBSEARCH_GENERATE_PROMPT",
]

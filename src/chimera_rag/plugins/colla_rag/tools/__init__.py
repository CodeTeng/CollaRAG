"""Shared tool layer for multi-agent architecture.

Tools are organized by category:
- retrieval_tools: vector, BM25, hybrid, graph neighbors/path/community
- entity_tools: entity extraction, relation extraction, entity linking
- reasoning_tools: sub-query decomposition, evidence assessment, reranking
- preprocess_tools: query rewrite, coreference resolution, follow-up merge
- memory_tools: read/write session, shared, and agent-private memory
- web_tools: web search, web fetch
- io_tools: save/load to/from file
- verification_tools: answer verify, claim decompose, source attribution
- graph_advanced_tools: subgraph extraction, graph statistics
- evidence_tools: temporal filter, evidence dedup, chunk summarize, confidence calibrate
- quality_evaluator: heuristic quality scoring (no LLM)
"""

"""GET /api/agents/* — CollaRAG multi-agent architecture introspection.

Exposes the 5 agent types, 31 tools, and the tool-permission matrix so the
frontend can render an interactive architecture overview (Innovation 2).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from chimera_rag import ChimeraRAG
from chimera_rag.web.deps import get_chimera
from chimera_rag.web.schemas import (
    AgentInfo,
    AgentsOverviewResponse,
    ToolCategoryInfo,
    ToolMatrixResponse,
)

router = APIRouter(tags=["agents"])

# --- Static metadata -----------------------------------------------------------

_AGENT_META: dict[str, dict] = {
    "preprocessor": {
        "cn_name": "预处理智能体",
        "description": "查询改写、共指消解、追问合并",
        "icon": "filter",
        "color": "accent",
    },
    "single_hop": {
        "cn_name": "单跳智能体",
        "description": "单轮检索 + 直接生成，适用于事实型 / 分析型问题",
        "icon": "zap",
        "color": "intent",
    },
    "multi_hop": {
        "cn_name": "多跳智能体",
        "description": "Plan-Execute-Reflect 架构，多步推理 + 图谱路径搜索",
        "icon": "git-branch",
        "color": "intent",
    },
    "summarization": {
        "cn_name": "摘要智能体",
        "description": "Map-Reduce 摘要，大范围证据压缩与综合",
        "icon": "file-text",
        "color": "intent",
    },
    "other": {
        "cn_name": "兜底智能体",
        "description": "Web 搜索 + 本地知识混合，处理图谱外查询",
        "icon": "globe",
        "color": "accent",
    },
}

_TOOL_CATEGORIES: list[dict] = [
    {
        "category": "retrieval",
        "cn_name": "检索工具",
        "tools": [
            {"name": "vector_search", "cn_name": "向量检索", "uses_llm": False},
            {"name": "bm25_search", "cn_name": "BM25 检索", "uses_llm": False},
            {"name": "hybrid_search", "cn_name": "混合检索", "uses_llm": False},
            {"name": "graph_neighbors", "cn_name": "图谱邻居", "uses_llm": False},
            {"name": "graph_path_search", "cn_name": "路径搜索", "uses_llm": False},
            {"name": "graph_community_search", "cn_name": "社区搜索", "uses_llm": False},
        ],
    },
    {
        "category": "entity",
        "cn_name": "实体工具",
        "tools": [
            {"name": "entity_extract", "cn_name": "实体抽取", "uses_llm": True},
            {"name": "relation_extract", "cn_name": "关系抽取", "uses_llm": True},
            {"name": "entity_link", "cn_name": "实体链接", "uses_llm": True},
        ],
    },
    {
        "category": "preprocess",
        "cn_name": "预处理工具",
        "tools": [
            {"name": "query_rewrite", "cn_name": "查询改写", "uses_llm": True},
            {"name": "coreference_resolve", "cn_name": "共指消解", "uses_llm": True},
            {"name": "followup_merge", "cn_name": "追问合并", "uses_llm": True},
        ],
    },
    {
        "category": "reasoning",
        "cn_name": "推理工具",
        "tools": [
            {"name": "sub_query_decompose", "cn_name": "子问题分解", "uses_llm": True},
            {"name": "assess_evidence", "cn_name": "证据评估", "uses_llm": False},
            {"name": "rerank", "cn_name": "重排序", "uses_llm": False},
        ],
    },
    {
        "category": "memory",
        "cn_name": "记忆工具",
        "tools": [
            {"name": "read_session_context", "cn_name": "读取会话上下文", "uses_llm": False},
            {"name": "read_shared_memory", "cn_name": "读取共享记忆", "uses_llm": False},
            {"name": "read_agent_memory", "cn_name": "读取智能体记忆", "uses_llm": False},
        ],
    },
    {
        "category": "io",
        "cn_name": "IO 工具",
        "tools": [
            {"name": "save_to_file", "cn_name": "保存文件", "uses_llm": False},
            {"name": "load_from_file", "cn_name": "加载文件", "uses_llm": False},
        ],
    },
    {
        "category": "web",
        "cn_name": "网络工具",
        "tools": [
            {"name": "web_search", "cn_name": "网络搜索", "uses_llm": False},
            {"name": "web_fetch", "cn_name": "网页抓取", "uses_llm": False},
        ],
    },
    {
        "category": "verification",
        "cn_name": "验证工具",
        "tools": [
            {"name": "answer_verify", "cn_name": "答案验证", "uses_llm": True},
            {"name": "claim_decompose", "cn_name": "原子声明分解", "uses_llm": True},
            {"name": "source_attribution", "cn_name": "来源归因", "uses_llm": True},
        ],
    },
    {
        "category": "graph_analysis",
        "cn_name": "图分析工具",
        "tools": [
            {"name": "graph_subgraph_extract", "cn_name": "子图抽取", "uses_llm": True},
            {"name": "graph_statistics", "cn_name": "图统计", "uses_llm": False},
        ],
    },
    {
        "category": "evidence",
        "cn_name": "证据处理工具",
        "tools": [
            {"name": "temporal_filter", "cn_name": "时间过滤", "uses_llm": False},
            {"name": "evidence_dedup", "cn_name": "证据去重", "uses_llm": False},
            {"name": "chunk_summarize", "cn_name": "片段摘要", "uses_llm": True},
            {"name": "confidence_calibrate", "cn_name": "置信度校准", "uses_llm": False},
        ],
    },
]


@router.get("/agents/overview", response_model=AgentsOverviewResponse)
def agents_overview(chimera: ChimeraRAG = Depends(get_chimera)) -> AgentsOverviewResponse:
    """Return the full multi-agent architecture overview."""
    from chimera_rag.plugins.colla_rag.agent_factory import TOOL_MATRIX

    colla_enabled = chimera.config.plugins.colla_rag.enabled

    agents = []
    for agent_type, meta in _AGENT_META.items():
        tool_names = TOOL_MATRIX.get(agent_type, [])
        agents.append(AgentInfo(
            agent_type=agent_type,
            cn_name=meta["cn_name"],
            description=meta["description"],
            icon=meta["icon"],
            color=meta["color"],
            tool_count=len(tool_names),
            tool_names=tool_names,
        ))

    categories = []
    for cat in _TOOL_CATEGORIES:
        categories.append(ToolCategoryInfo(
            category=cat["category"],
            cn_name=cat["cn_name"],
            tools=cat["tools"],
        ))

    all_tool_names = set()
    for cat in _TOOL_CATEGORIES:
        for t in cat["tools"]:
            all_tool_names.add(t["name"])
    llm_tools = sum(
        1 for cat in _TOOL_CATEGORIES for t in cat["tools"] if t["uses_llm"]
    )

    return AgentsOverviewResponse(
        enabled=colla_enabled,
        agent_count=len(agents),
        tool_count=len(all_tool_names),
        llm_tool_count=llm_tools,
        deterministic_tool_count=len(all_tool_names) - llm_tools,
        agents=agents,
        tool_categories=categories,
    )


@router.get("/agents/matrix", response_model=ToolMatrixResponse)
def tool_matrix() -> ToolMatrixResponse:
    """Return the tool-permission matrix (agent → tool names)."""
    from chimera_rag.plugins.colla_rag.agent_factory import TOOL_MATRIX

    return ToolMatrixResponse(
        matrix=TOOL_MATRIX,
        agent_types=list(TOOL_MATRIX.keys()),
        tool_totals={k: len(v) for k, v in TOOL_MATRIX.items()},
    )


__all__ = ["router"]

"""PrismRAG —— 图谱投影驱动的多粒度缺陷修复式分块（创新点一）。

命名对称：``CollaRAG`` = **Colla**borative RAG（查询侧 · 多智能体协作）；
``PrismRAG`` = **Prism** RAG（构建侧 · 一篇文档经棱镜分出多粒度层，
每层绑定一个图谱投影，正如白光分出光谱）。

🚧 **当前实装状态**：仅 Phase 2 的单粒度过渡版（``prism.chunker``）。
完整设计（双粒度 + 类型化缺陷 D1–D6 + 审计修复闭环）见
``docs/innovation-1/design.md``。
"""

from chimera_rag.plugins.prism import chunker  # noqa: F401

"""PrismRAG —— 图谱投影驱动的多粒度缺陷修复式分块（创新点一）。

命名对称：``CollaRAG`` = **Colla**borative RAG（查询侧 · 多智能体协作）；
``PrismRAG`` = **Prism** RAG（构建侧 · 一篇文档经棱镜分出多粒度层，
每层绑定一个图谱投影，正如白光分出光谱）。

模块划分：

* :mod:`~chimera_rag.plugins.prism.features` —— 零依赖结构信号（复杂度 / 实体密度 /
  句子切分）。内部特征，不是对外宣称的贡献。
* :mod:`~chimera_rag.plugins.prism.chunker` —— Phase 1–2 双粒度构建，并实现
  ``BaseChunkAuditor`` 能力以接入 Phase 3–4。
* :mod:`~chimera_rag.plugins.prism.defects` —— Phase 3 的 D1–D6 类型化缺陷检测器。
* :mod:`~chimera_rag.plugins.prism.audit` —— 把检测器组装成审计器，并为 D1 构造
  「本轮三元组 + 已提交图谱」的审计视图。
* :mod:`~chimera_rag.plugins.prism.repair` —— Phase 4 的边界修复动作。

设计与消融方案见 ``docs/innovation-1/design.md``。
"""

from chimera_rag.plugins.prism import chunker  # noqa: F401

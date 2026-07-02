"""
Figure 4: 检索质量分析 — 水平分组条形图
证据覆盖率 (左) 和 平均相关性 (右)
输出: ../figures/figure4.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = [
    "Vanilla RAG", "RAPTOR", "TOG", "G-Retriever",
    "LightRAG", "GraphRAG", "HippoRAG", "HippoRAG2",
    "IntentAgent\n(Static)", "IntentAgent\n(Dynamic)", "IntentAgent\n(Memory)",
]
coverage = [0.414, 0.510, 0.304, 0.475, 0.445, 0.507, 0.635, 0.671, 0.554, 0.700, 0.738]
relevance = [2.90,  3.89,  2.75, 3.54,  3.25,  3.61,  3.42,  3.73,  3.78,  3.97,  4.21]

# color scheme: text=steelblue, graph=seagreen, hybrid=darkorange, ours=crimson
colors = [
    "#4682B4", "#4682B4",                           # text
    "#2E8B57", "#2E8B57",                           # graph
    "#E87722", "#E87722", "#E87722", "#E87722",     # hybrid
    "#B22222", "#B22222", "#B22222",                # ours
]

y = np.arange(len(methods))
fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
fig.subplots_adjust(wspace=0.05)

bar_h = 0.65

# ── Left: Evidence Coverage ──────────────────────────────────────────────────
ax = axes[0]
bars = ax.barh(y, coverage, height=bar_h, color=colors, edgecolor="white", linewidth=0.4)
ax.set_xlim(0, 0.85)
ax.set_xlabel("Evidence Coverage", fontsize=10)
ax.set_yticks(y)
ax.set_yticklabels(methods, fontsize=8.5)
ax.invert_yaxis()
ax.axvline(x=coverage[-1], color="#B22222", linewidth=0.8, linestyle="--", alpha=0.5)
for bar, val in zip(bars, coverage):
    ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

# ── Right: Average Relevance ──────────────────────────────────────────────────
ax2 = axes[1]
bars2 = ax2.barh(y, relevance, height=bar_h, color=colors, edgecolor="white", linewidth=0.4)
ax2.set_xlim(2.0, 4.7)
ax2.set_xlabel("Avg. Relevance (1–5)", fontsize=10)
ax2.set_yticks(y)
ax2.set_yticklabels([], fontsize=8.5)
ax2.invert_yaxis()
ax2.axvline(x=relevance[-1], color="#B22222", linewidth=0.8, linestyle="--", alpha=0.5)
for bar, val in zip(bars2, relevance):
    ax2.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va="center", ha="left", fontsize=7.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

# ── Legend ────────────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4682B4", label="Text-based"),
    Patch(facecolor="#2E8B57", label="Graph-based"),
    Patch(facecolor="#E87722", label="Hybrid"),
    Patch(facecolor="#B22222", label="IntentAgent"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.savefig("../figures/figure4.pdf", bbox_inches="tight", dpi=300)
print("Saved figures/figure4.pdf")

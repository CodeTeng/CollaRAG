"""
Figure 5: 规模敏感性分析 — 折线图
x=样本量, y=EM (HotpotQA, Qwen3-32B)
输出: ../figures/figure5.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sample_sizes = [100, 200, 300, 400, 500]

data = {
    "RAPTOR":       ([0.393, 0.433, 0.430, 0.423, 0.422], "o",  "#4682B4", "--"),
    "GraphRAG":     ([0.415, 0.387, 0.404, 0.405, 0.407], "s",  "#6495ED", ":"),
    "HippoRAG":     ([0.493, 0.458, 0.470, 0.459, 0.470], "^",  "#E87722", "-."),
    "HippoRAG2":    ([0.507, 0.475, 0.486, 0.480, 0.488], "D",  "#FF8C00", "-"),
    "IntentAgent":  ([0.523, 0.507, 0.528, 0.534, 0.538], "*",  "#B22222", "-"),
}

fig, ax = plt.subplots(figsize=(6, 4))

for name, (vals, marker, color, ls) in data.items():
    lw = 2.2 if name == "IntentAgent" else 1.4
    ms = 10 if marker == "*" else 6
    ax.plot(sample_sizes, vals, marker=marker, color=color, linestyle=ls,
            linewidth=lw, markersize=ms, label=name, zorder=3 if name == "IntentAgent" else 2)

ax.set_xlabel("Number of Samples (N)", fontsize=11)
ax.set_ylabel("Exact Match (EM)", fontsize=11)
ax.set_xticks(sample_sizes)
ax.set_xlim(80, 520)
ax.set_ylim(0.34, 0.56)
ax.legend(fontsize=7.5, framealpha=0.85, loc="upper right", handlelength=1.5, handletextpad=0.6, borderpad=0.4, labelspacing=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("../figures/figure5.pdf", bbox_inches="tight", dpi=300)
print("Saved figures/figure5.pdf")

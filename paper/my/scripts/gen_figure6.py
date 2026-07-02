"""
Figure 6: 经验记忆增长曲线 — 双线折线图
x=查询窗口中点, y=EM 和 F1 (HotpotQA, Qwen3-32B)
输出: ../figures/figure6.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 用各窗口中点值作为 x 刻度
x_labels = ["1–50\n(cold)", "51–100", "101–200", "201–300", "301–400", "401–500"]
x = np.arange(len(x_labels))

em = [0.424, 0.455, 0.463, 0.474, 0.492, 0.495]
f1 = [0.571, 0.602, 0.619, 0.638, 0.645, 0.647]

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.plot(x, em, marker="o", color="#B22222", linewidth=2.2, markersize=7, label="EM")
ax.plot(x, f1, marker="s", color="#2E8B57", linewidth=2.2, markersize=7, label="F1",
        linestyle="--")

# Annotate first and last points
for vals, color in [(em, "#B22222"), (f1, "#2E8B57")]:
    ax.annotate(f"{vals[0]:.3f}", xy=(0, vals[0]),
                xytext=(-12, 6), textcoords="offset points",
                fontsize=8, color=color, ha="center")
    ax.annotate(f"{vals[-1]:.3f}", xy=(len(x)-1, vals[-1]),
                xytext=(12, 6), textcoords="offset points",
                fontsize=8, color=color, ha="center")

# Cold-start shading
ax.axvspan(-0.4, 0.4, color="gray", alpha=0.08, label="Cold start")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_xlabel("Query Window", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylim(0.38, 0.70)
ax.legend(fontsize=10, framealpha=0.9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("../figures/figure6.pdf", bbox_inches="tight", dpi=300)
print("Saved figures/figure6.pdf")

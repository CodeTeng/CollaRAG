"""Generate Figure 5: Grouped bar chart for ablation study (Qwen3-32B)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

datasets = ["2Wiki", "HotpotQA", "MuSiQue-A", "MuSiQue-F"]
variants = [
    "IntentAgent\n(Full)",
    "w/o\nIntent",
    "w/o\nDynamic",
    "w/o\nReflect",
    "w/o\nMemory",
    "w/o\nDual-Path",
]

em_data = {
    "2Wiki":      [0.589, 0.513, 0.528, 0.562, 0.568, 0.582],
    "HotpotQA":   [0.483, 0.434, 0.455, 0.466, 0.463, 0.463],
    "MuSiQue-A":  [0.271, 0.226, 0.255, 0.254, 0.258, 0.265],
    "MuSiQue-F":  [0.182, 0.145, 0.172, 0.165, 0.179, 0.179],
}

colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
hatches = ["", "//", "\\\\", "xx", "..", "oo"]

n_variants = len(variants)
n_datasets = len(datasets)
x = np.arange(n_datasets)
width = 0.12
offsets = np.arange(n_variants) - (n_variants - 1) / 2

fig, ax = plt.subplots(figsize=(9, 4.5))

for i, variant in enumerate(variants):
    vals = [em_data[ds][i] for ds in datasets]
    bars = ax.bar(x + offsets[i] * width, vals, width * 0.92,
                  label=variant.replace("\n", " "),
                  color=colors[i], hatch=hatches[i],
                  edgecolor="white", linewidth=0.5)
    if i == 0:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylabel("Exact Match (EM)", fontsize=11)
ax.set_ylim(0.08, 0.68)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True, fancybox=True,
          borderpad=0.4, columnspacing=0.8)

plt.tight_layout()
out = Path(__file__).parent / "fig5_ablation.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")

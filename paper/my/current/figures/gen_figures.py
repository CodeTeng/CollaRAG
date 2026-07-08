#!/usr/bin/env python3
"""Generate publication-quality PDF figures for the CollaRAG paper.

Outputs (in this directory):
  fig_radar.pdf         - performance radar (Qwen3-8B, 4 multi-hop datasets)
  fig_ablation.pdf      - ablation grouped bar (Qwen3-32B, 5 datasets)
  fig_hopdepth.pdf      - EM vs hop depth (Qwen3-8B, MuSiQue)
  fig_memory_curve.pdf  - memory accumulation curve (Qwen3-32B, HotpotQA)
  fig_hyperparam.pdf    - hyperparameter sensitivity small-multiples (Qwen3-32B, HotpotQA)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- global style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # embed TrueType, editable
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

# ---------- palette (colorblind-aware, print-safe) ----------
C = {
    "ours":  "#C0392B",   # crimson  - CollaRAG
    "h2":    "#2C3E50",   # slate    - HippoRAG2
    "h1":    "#2980B9",   # blue     - HippoRAG
    "grag":  "#16A085",   # teal     - GraphRAG
    "rap":   "#8E44AD",   # purple   - RAPTOR
    "vrag":  "#95A5A6",   # gray     - Vanilla RAG
    "llm":   "#BDC3C7",   # lgray    - LLM-only
    "tog":   "#D35400",   # pumpkin  - TOG
    "gretr": "#7D3C98",   # violet   - G-Retriever
}
M = {"ours": "o", "h2": "s", "h1": "^", "grag": "D", "rap": "v", "vrag": "P", "tog": "X", "gretr": "*"}

OUT = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# 1. Radar  (Qwen3-8B, EM, 4 multi-hop datasets)
# =====================================================================
def radar():
    labels = ["2Wiki", "HotpotQA", "MuSiQue-A", "MuSiQue-F"]
    series = [
        ("CollaRAG (ours)",  [0.602, 0.510, 0.277, 0.189], C["ours"]),
        ("HippoRAG2",        [0.561, 0.466, 0.254, 0.178], C["h2"]),
        ("GraphRAG",         [0.314, 0.371, 0.169, 0.137], C["grag"]),
        ("RAPTOR",           [0.312, 0.352, 0.132, 0.085], C["rap"]),
    ]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.set_ylim(0.0, 0.65)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"], fontsize=7.5, color="#555")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", pad=8)
    ax.grid(color="#CCC", linewidth=0.7)
    ax.spines["polar"].set_color("#BBB")

    for name, vals, color in series:
        v = vals + vals[:1]
        lw = 2.2 if "ours" in name.lower() else 1.4
        ax.plot(angles, v, color=color, linewidth=lw, label=name,
                marker=M.get(_key(name), "o") if False else None)
        ax.fill(angles, v, color=color, alpha=0.10 if "ours" in name.lower() else 0.05)

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig_radar.pdf"))
    plt.close(fig)


def _key(name):
    n = name.lower()
    if "ours" in n or "collarag" in n: return "ours"
    if "hipporag2" in n: return "h2"
    if "hipporag" in n: return "h1"
    if "graphrag" in n: return "grag"
    if "raptor" in n: return "rap"
    if "vanilla" in n: return "vrag"
    if "tog" in n: return "tog"
    if "retriever" in n: return "gretr"
    return "ours"


# =====================================================================
# 2. Ablation grouped bar (Qwen3-32B, EM, 5 datasets)
# =====================================================================
def ablation():
    datasets = ["2Wiki", "HotpotQA", "MuSi-A", "MuSi-F", "ASQA"]
    variants = [
        ("Full",            [0.602, 0.510, 0.277, 0.189, 0.338], C["ours"]),
        ("w/o Query Triage",[0.485, 0.409, 0.205, 0.130, 0.216], C["h2"]),
        ("w/o PER",         [0.559, 0.467, 0.247, 0.170, 0.301], C["grag"]),
        ("w/o Memory",      [0.572, 0.479, 0.259, 0.179, 0.316], C["rap"]),
    ]
    x = np.arange(len(datasets))
    w = 0.2
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for i, (name, vals, color) in enumerate(variants):
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=name, color=color,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.006, f"{v:.3f}"[1:],
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("EM")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              columnspacing=1.2, handletextpad=0.4)
    ax.grid(axis="y", color="#DDD", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, "fig_ablation.pdf"))
    plt.close(fig)


# =====================================================================
# 3. Hop-depth line (Qwen3-8B, MuSiQue, EM)
# =====================================================================
def hopdepth():
    hops = [2, 3, 4]
    series = [
        ("CollaRAG (ours)", [0.267, 0.207, 0.153], C["ours"], 2.4, "o"),
        ("HippoRAG2",       [0.237, 0.192, 0.138], C["h2"],   1.5, "s"),
        ("HippoRAG",        [0.218, 0.175, 0.125], C["h1"],   1.5, "^"),
        ("GraphRAG",        [0.185, 0.148, 0.091], C["grag"], 1.5, "D"),
        ("RAPTOR",          [0.142, 0.095, 0.061], C["rap"],  1.5, "v"),
        ("Vanilla RAG",     [0.098, 0.052, 0.035], C["vrag"], 1.5, "P"),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for name, vals, color, lw, mk in series:
        ax.plot(hops, vals, color=color, linewidth=lw, marker=mk, markersize=5,
                label=name, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xticks(hops)
    ax.set_xticklabels(["2-hop\n(n=187)", "3-hop\n(n=198)", "4-hop\n(n=115)"])
    ax.set_xlabel("Reasoning depth (MuSiQue)")
    ax.set_ylabel("EM")
    ax.set_ylim(0, 0.30)
    ax.grid(axis="y", color="#DDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", columnspacing=1.0,
              handletextpad=0.4, labelspacing=0.35)
    fig.savefig(os.path.join(OUT, "fig_hopdepth.pdf"))
    plt.close(fig)


# =====================================================================
# 4. Memory accumulation curve (Qwen3-32B, HotpotQA)
# =====================================================================
def memory_curve():
    mid = [25, 75, 150, 250, 350, 450]            # query-window midpoints
    em  = [0.422, 0.454, 0.469, 0.484, 0.499, 0.507]
    f1  = [0.535, 0.569, 0.585, 0.604, 0.616, 0.625]
    hit = [0.0, 2.0, 4.5, 6.8, 8.0, 8.5]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    l1, = ax.plot(mid, em, color=C["ours"],  linewidth=2.2, marker="o", markersize=5,
                  markeredgecolor="white", markeredgewidth=0.6, label="EM")
    l2, = ax.plot(mid, f1, color=C["h2"],    linewidth=2.0, marker="s", markersize=5,
                  markeredgecolor="white", markeredgewidth=0.6, label="F1")
    ax.set_xlabel("Query index (window midpoint)")
    ax.set_ylabel("Answer quality (EM / F1)")
    ax.set_ylim(0.38, 0.65)
    ax.set_xticks(mid)
    ax.set_xticklabels(["1-50", "51-100", "101-200", "201-300", "301-400", "401-500"],
                       rotation=0, fontsize=8)
    ax.grid(axis="y", color="#EEE", linewidth=0.6)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    l3, = ax2.plot(mid, hit, color=C["grag"], linewidth=2.0, marker="^", markersize=5,
                   markeredgecolor="white", markeredgewidth=0.6, linestyle="--",
                   label="QA-cache hit (%)")
    ax2.set_ylabel("QA-cache hit rate (%)", color=C["grag"])
    ax2.tick_params(axis="y", labelcolor=C["grag"])
    ax2.set_ylim(0, 12)

    ax.legend(handles=[l1, l2, l3], frameon=False, loc="lower right",
              fontsize=8.5, columnspacing=1.0, handletextpad=0.5)
    fig.savefig(os.path.join(OUT, "fig_memory_curve.pdf"))
    plt.close(fig)


# =====================================================================
# 5. Hyperparameter sensitivity (Qwen3-32B, HotpotQA, EM)
# =====================================================================
def hyperparam():
    panels = [
        ("max_plan_steps",        [3, 5, 7, 10],            [0.490, 0.510, 0.507, 0.500], 5),
        ("quality threshold τ",   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                  [0.496, 0.502, 0.506, 0.510, 0.504, 0.492], 0.6),
        ("reflect_retries",       [0, 1, 2, 3],             [0.479, 0.505, 0.510, 0.511], 2),
        ("max_step_iterations",   [1, 2, 3, 4],             [0.480, 0.502, 0.510, 0.511], 3),
        ("SingleHop top-k",       [5, 10, 15, 20],          [0.492, 0.501, 0.502, 0.501], 10),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.0))
    axes = axes.ravel()
    for ax, (name, xs, ys, default) in zip(axes, panels):
        ax.plot(xs, ys, color=C["h2"], linewidth=1.6, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.5)
        # highlight default
        di = xs.index(default)
        ax.scatter([xs[di]], [ys[di]], s=70, color=C["ours"], zorder=5,
                   marker="*", edgecolor="white", linewidth=0.6, label="default")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.004, f"{y:.3f}"[1:], ha="center", va="bottom", fontsize=6.5)
        ax.set_title(name, fontsize=9.5)
        ax.set_ylim(min(ys) - 0.012, max(ys) + 0.012)
        ax.grid(axis="y", color="#EEE", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7.5)
    # 6th panel: legend only
    axes[5].axis("off")
    handles = [
        plt.Line2D([0], [0], color=C["h2"], linewidth=1.6, marker="o", markersize=5,
                   markeredgecolor="white", markeredgewidth=0.5, label="EM"),
        plt.Line2D([0], [0], color=C["ours"], linewidth=0, marker="*", markersize=12,
                   markeredgecolor="white", markeredgewidth=0.6, label="default config"),
    ]
    axes[5].legend(handles=handles, loc="center", frameon=False, fontsize=9.5)
    fig.suptitle("Hyperparameter sensitivity (Qwen3-8B, HotpotQA)", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "fig_hyperparam.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    radar()
    ablation()
    hopdepth()
    memory_curve()
    hyperparam()
    print("done")

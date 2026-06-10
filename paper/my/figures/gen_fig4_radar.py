"""Generate Figure 4: Radar chart comparing IntentAgent vs Top-3 baselines on EM across 4 datasets."""
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

data_8b = {
    "RAPTOR":       [0.428, 0.382, 0.205, 0.171],
    "GraphRAG":     [0.314, 0.371, 0.169, 0.137],
    "HippoRAG":    [0.535, 0.451, 0.241, 0.162],
    "IntentAgent":  [0.551, 0.462, 0.249, 0.155],
}

data_32b = {
    "RAPTOR":       [0.471, 0.422, 0.227, 0.186],
    "GraphRAG":     [0.368, 0.407, 0.197, 0.145],
    "HippoRAG":    [0.576, 0.470, 0.262, 0.190],
    "IntentAgent":  [0.589, 0.483, 0.271, 0.182],
}

colors = {
    "RAPTOR":      "#7eb0d5",
    "GraphRAG":    "#b2e061",
    "HippoRAG":   "#fd7f6f",
    "IntentAgent": "#bd7ebe",
}
line_styles = {
    "RAPTOR":      "--",
    "GraphRAG":    "-.",
    "HippoRAG":   ":",
    "IntentAgent": "-",
}
markers = {
    "RAPTOR":      "s",
    "GraphRAG":    "D",
    "HippoRAG":   "^",
    "IntentAgent": "o",
}

def draw_radar(ax, data, title):
    N = len(datasets)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=9)

    ax.set_ylim(0.1, 0.65)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(["0.2", "0.3", "0.4", "0.5", "0.6"], fontsize=7, color="grey")

    for method, vals in data.items():
        values = vals + vals[:1]
        lw = 2.2 if method == "IntentAgent" else 1.4
        ax.plot(angles, values, linestyle=line_styles[method], linewidth=lw,
                color=colors[method], label=method, marker=markers[method], markersize=5)
        if method == "IntentAgent":
            ax.fill(angles, values, alpha=0.08, color=colors[method])

    ax.set_title(title, fontsize=11, fontweight="bold", pad=18)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2), subplot_kw=dict(polar=True))
draw_radar(ax1, data_8b, "Qwen3-8B")
draw_radar(ax2, data_32b, "Qwen3-32B")

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
           frameon=True, fancybox=True, shadow=False, borderpad=0.4)

plt.tight_layout(rect=[0, 0.08, 1, 1])
out = Path(__file__).parent / "fig4_radar.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")

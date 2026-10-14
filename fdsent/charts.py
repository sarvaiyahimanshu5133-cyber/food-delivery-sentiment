"""Save the charts as PNG files in output/ (handy for your report or slides)."""
import matplotlib

matplotlib.use("Agg")                 # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from config import LABELS, OUT

COLORS = {"negative": "#c0392b", "neutral": "#aab2bd", "positive": "#2e8b57"}


def save_all(r):
    OUT.mkdir(exist_ok=True)
    brand_bars(r["summary"])
    topics(r["topics"])
    model_comparison(r["results"])
    confusion(r["matrices"])


def brand_bars(summary):
    s = summary.set_index("brand")
    fig, ax = plt.subplots(figsize=(7, 1 + 0.9 * len(s)))
    left = np.zeros(len(s))
    for label in [l for l in LABELS if s[f"{l}_%"].sum() > 0]:      # skip a class nobody was assigned to
        v = s[f"{label}_%"].values
        ax.barh(s.index, v, left=left, color=COLORS[label], label=label)
        for x, w, y in zip(left, v, range(len(s))):
            if w > 6:
                ax.text(x + w / 2, y, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=10)
        left += v
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of reviews")
    ax.set_title("Sentiment of reviews, by brand")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "sentiment_by_brand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def topics(t):
    p = t.pivot(index="topic", columns="brand", values="negative_reviews_%")
    p = p.loc[p.mean(axis=1).sort_values().index]
    ax = p.plot.barh(figsize=(8, 4.5), color=["#f28e2b", "#4e79a7", "#59a14f", "#b07aa1"][:p.shape[1]])
    ax.set_xlabel("% of NEGATIVE reviews that mention the topic")
    ax.set_ylabel("")
    ax.set_title("Why people complain")
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "complaint_topics.png", dpi=150)
    plt.close(ax.figure)


def model_comparison(res):
    m = res.set_index("model")[["accuracy", "accuracy_pos_vs_neg", "macro_f1"]]
    m.columns = ["Accuracy (3 classes)", "Accuracy (positive vs negative)", "Macro F1"]
    ax = m.T.plot.bar(figsize=(9, 4.2), rot=0, color=["#94a3b8", "#c084fc", "#fbbf24", "#38bdf8"][:len(m)])
    ax.set_ylim(0, 1)
    ax.set_title("VADER (ready-made) vs Logistic Regression (trained by us)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=9)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close(ax.figure)


def confusion(matrices):
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4))
    for ax, (name, cm) in zip(np.atleast_1d(axes), matrices.items()):
        cm = np.array(cm, dtype=float)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual (from stars)")
        ax.set_title(name)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] / max(1, cm[i].sum()) > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=150)
    plt.close(fig)

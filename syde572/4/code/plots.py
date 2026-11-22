import json
import numpy as np
import matplotlib.pyplot as plt
from nb import wilson

BLUE, ORANGE, AQUA, VIOLET, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#0b0b0b", "#8a8984"
plt.rcParams.update({
    "figure.dpi": 160, "savefig.bbox": "tight", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8, "legend.frameon": False, "figure.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})
OUT = "../media/"
R = json.load(open("results.json"))


def save(fig, name):
    fig.savefig(OUT + name)
    plt.close(fig)


def smoothing():
    a, acc = R["cv"]["alphas"], np.array(R["cv"]["acc"]) * 100
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(a, acc, "-o", color=BLUE, lw=2, ms=7, mec="white")
    best = int(np.argmax(acc))
    ax.plot(a[best], acc[best], "*", color=VIOLET, ms=16, mec="white", zorder=5, label=f"best α = {a[best]:g} ({acc[best]:.2f}%)")
    ax.set_xscale("log")
    ax.set(xlabel="smoothing α (log scale)", ylabel="5-fold CV accuracy (%)", title="Choosing the smoothing parameter")
    ax.legend(loc="lower right")
    save(fig, "smoothing.png")


def curve():
    n, acc = np.array(R["curve"]).T
    lo, hi = np.array([wilson(x, R["main"]["n"]) for x in acc]).T
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.fill_between(n, lo * 100, hi * 100, color=BLUE, alpha=0.15, lw=0, label="95% Wilson interval")
    ax.plot(n, acc * 100, "-o", color=BLUE, lw=2, ms=7, mec="white", label="evaluation accuracy")
    ax.set(xlabel="training documents", ylabel="accuracy (%)", title="Learning curve (α = 5, fixed 2,383-document evaluation set)")
    ax.legend(loc="lower right")
    save(fig, "curve.png")


def features():
    pos, neg = R["features"]["pos"][:15], R["features"]["neg"][:15]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    words = [w for w, _ in neg][::-1] + [w for w, _ in pos]
    vals = [v for _, v in neg][::-1] + [v for _, v in pos]
    y = np.arange(len(words))
    ax.barh(y, vals, color=[BLUE if v > 0 else ORANGE for v in vals], height=0.7)
    ax.set_yticks(y, words)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="log P(w | pos) − log P(w | neg)", title=f"Most informative words (α = {R['features']['alpha']})")
    ax.text(0.98, 0.04, "positive →", transform=ax.transAxes, ha="right", color=INK)
    ax.text(0.02, 0.96, "← negative", transform=ax.transAxes, ha="left", va="top", color=INK)
    save(fig, "features.png")


def heatmap(ax, M, rows, cols, fmt, cmap="Blues"):
    im = ax.imshow(M, cmap=cmap)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9,
                    color="white" if (v - M.min()) / (M.max() - M.min()) > 0.6 else INK)
    return im


def topic_cm():
    t = R["topic"]
    M = np.array(t["confusion"])
    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap(ax, M, t["classes"], t["classes"], lambda v: f"{v}")
    ax.set(xlabel="predicted topic", ylabel="true topic", title=f"Topic classifier: {t['acc'] * 100:.1f}% accuracy")
    save(fig, "topic.png")


def domain():
    topics = sorted({k.split(">")[0] for k in R["domain_full"]})
    M = np.array([[R["domain_full"][f"{s}>{d}"] * 100 for d in topics] for s in topics])
    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap(ax, M, topics, topics, lambda v: f"{v:.0f}")
    ax.set(xlabel="tested on", ylabel="trained on", title="Sentiment accuracy (%) across domains")
    save(fig, "domain.png")


if __name__ == "__main__":
    for f in (smoothing, curve, features, topic_cm, domain):
        f()

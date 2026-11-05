import itertools
import numpy as np
import matplotlib.pyplot as plt
from pca import FEATURES, pca_by_hand, reconstruct, iris, pca, explained, isomap, isomap_smacof

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
X, y, NAMES = iris()
COLORS, MARKERS = [BLUE, ORANGE, AQUA], ["o", "s", "^"]
UNITS = [f"{f} (cm)" for f in FEATURES]


def save(fig, name):
    fig.savefig(OUT + name)
    plt.close(fig)


def scatter(ax, A, B, C=None, s=22):
    for c in range(3):
        m = y == c
        args = (A[m], B[m]) if C is None else (A[m], B[m], C[m])
        ax.scatter(*args, s=s, color=COLORS[c], marker=MARKERS[c], edgecolor="white", linewidth=0.5,
                   alpha=0.9, label=NAMES[c])


def legend(fig, ax):
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))


def strips(ax, z):
    bins = np.linspace(z.min(), z.max(), 30)
    for c in range(3):
        ax.hist(z[y == c], bins=bins, color=COLORS[c], alpha=0.35)
        ax.hist(z[y == c], bins=bins, color=COLORS[c], histtype="step", lw=1.6, label=NAMES[c])
    ax.set_ylabel("count")


def hand():
    P = np.array([[-2, -1], [-1, 0], [0, 1], [1, 2], [2, 3]], float)
    mu, _, W = pca_by_hand(P)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    t = np.linspace(-4.5, 4.5, 2)
    ax.plot(mu[0] + t * W[0], mu[1] + t * W[1], color=VIOLET, lw=2, label="PC1 line through mean")
    ax.annotate("", mu + 1.6 * W.ravel(), mu, arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=2))
    ax.plot(*P.T, "o", color=BLUE, ms=8, mec="white", label="data")
    ax.plot(*mu, "P", color=INK, ms=10, mec="white", label="mean (0, 1)")
    for x, c in [([3, 3], ORANGE), ([3, 4], AQUA)]:
        x = np.array(x, float)
        _, xh, e = reconstruct(x, mu, W)
        ax.plot([x[0], xh[0]], [x[1], xh[1]], color=c, lw=1.6, ls="--")
        ax.plot(*x, "D", color=c, ms=8, mec="white", label=f"({x[0]:g}, {x[1]:g}): error {e:.4f}")
        if e > 1e-9:
            ax.plot(*xh, "x", color=c, ms=9, mew=2)
            ax.annotate(f"x̂ = ({xh[0]:g}, {xh[1]:g})", xh, xytext=(-80, 6), textcoords="offset points", fontsize=9)
    ax.set_aspect("equal")
    ax.set(xlim=(-3, 4.5), ylim=(-2, 5), xlabel="$x_1$", ylabel="$x_2$", title="PC1 = (1, 1)/√2 and reconstructions")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "p1_hand.png")


def marginals_1d():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for i, ax in enumerate(axes):
        strips(ax, X[:, i])
        ax.set(xlabel=UNITS[i], title=FEATURES[i].capitalize())
        if i:
            ax.set_ylabel("")
    legend(fig, axes[0])
    save(fig, "p2_1d.png")


def marginals_2d():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for ax, (i, j) in zip(axes.flat, itertools.combinations(range(4), 2)):
        scatter(ax, X[:, i], X[:, j])
        ax.set(xlabel=UNITS[i], ylabel=UNITS[j])
    legend(fig, axes[0, 0])
    fig.tight_layout()
    save(fig, "p2_2d.png")


def marginals_3d():
    fig = plt.figure(figsize=(12, 5))
    for k, idx in enumerate([(0, 1, 2), (0, 2, 3)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        scatter(ax, *[X[:, i] for i in idx])
        ax.set(xlabel=UNITS[idx[0]], ylabel=UNITS[idx[1]], zlabel=UNITS[idx[2]])
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect(None, zoom=0.85)
    legend(fig, ax)
    save(fig, "p2_3d.png")


def pca_views():
    _, Z = pca(X)
    fig = plt.figure(figsize=(14, 4.4))
    a1 = fig.add_subplot(1, 3, 1)
    strips(a1, Z[:, 0])
    a1.set(xlabel="PC1", title="Top 1 PC")
    a2 = fig.add_subplot(1, 3, 2)
    scatter(a2, Z[:, 0], Z[:, 1])
    a2.set(xlabel="PC1", ylabel="PC2", title="Top 2 PCs")
    a3 = fig.add_subplot(1, 3, 3, projection="3d")
    scatter(a3, Z[:, 0], Z[:, 1], Z[:, 2])
    a3.set(xlabel="PC1", ylabel="PC2", zlabel="PC3")
    a3.set_title("Top 3 PCs", loc="left")
    a3.set_box_aspect(None, zoom=0.85)
    legend(fig, a2)
    save(fig, "p3_pca.png")


def variance():
    r = explained(X) * 100
    k = np.arange(1, 5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(k, r, color=BLUE, width=0.55, label="per PC")
    ax.plot(k, np.cumsum(r), "-o", color=ORANGE, lw=2, ms=7, mec="white", label="cumulative")
    ax.axhline(50, color=MUTED, lw=1, ls="--")
    for ki, ri, ci in zip(k, r, np.cumsum(r)):
        if ki > 1:
            ax.annotate(f"{ri:.2f}%", (ki, ri), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
        ax.annotate(f"{ci:.2f}%", (ki, ci), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set(xticks=k, xticklabels=[f"PC{i}" for i in k], ylim=(0, 110), ylabel="variance explained (%)",
           title="Explained variance, σᵢ² / ‖X‖²_F")
    ax.legend(loc="center right")
    save(fig, "p4_variance.png")


def isomap_views():
    iso, Y = isomap(X)
    Ys = isomap_smacof(iso)
    _, Z = pca(X)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, E, title in [(axes[0], Y, "Isomap, k = 3"), (axes[1], Ys, "Isomap distances + SMACOF MDS"),
                         (axes[2], Z, "PCA, top 2")]:
        scatter(ax, E[:, 0], E[:, 1])
        ax.set(title=title, xlabel="component 1", ylabel="component 2")
    legend(fig, axes[0])
    save(fig, "p5_isomap.png")


if __name__ == "__main__":
    for fn in [hand, marginals_1d, marginals_2d, marginals_3d, pca_views, variance, isomap_views]:
        fn()

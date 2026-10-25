import numpy as np
import matplotlib.pyplot as plt
from regression import (LAMBDAS, NAMES, FEATURE, polynomial_regression, polynomial_regression_1d_vis,
                        polynomial_regression_reg, gaussian_regression, gaussian_regression_reg, poly_design)

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


def save(fig, name):
    fig.savefig(OUT + name)
    plt.close(fig)


def poly_errors():
    degrees, train, test, _ = polynomial_regression()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(degrees, train, "-o", color=BLUE, lw=2, ms=7, mec="white", label="training RMS")
    ax.plot(degrees, test, "-s", color=ORANGE, lw=2, ms=7, mec="white", label="test RMS")
    for d, a, b in zip(degrees, train, test):
        ax.annotate(f"{b:.3g}", (d, b), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.annotate(f"{a:.2f}", (d, a), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 1e17)
    ax.set(xlabel="polynomial degree", ylabel="RMS error ($1000s, log scale)",
           title="Polynomial regression, all 13 features, no regularization")
    ax.legend(loc="upper left")
    save(fig, "p1_poly_errors.png")


def weights():
    _, _, _, ws = polynomial_regression(degrees=[1])
    w = ws[0][1:]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(NAMES))[::-1]
    colors = [VIOLET if n in ("RM", "DIS") else BLUE for n in NAMES]
    ax.barh(y, w, color=colors, height=0.6)
    for yi, wi in zip(y, w):
        ax.annotate(f"{wi:+.3f}", (wi, yi), xytext=(5 if wi >= 0 else -5, 0), textcoords="offset points",
                    ha="left" if wi >= 0 else "right", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [f"w{i + 1}  {n}" for i, n in enumerate(NAMES)])
    ax.set_xlim(-3, 5.2)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="weight ($1000s per training standard deviation)",
           title=f"Degree-1 weights on normalized features (bias w0 = {ws[0][0]:.2f})")
    save(fig, "p1_weights.png")


def vis():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, d in zip(axes, (1, 3, 10)):
        r = polynomial_regression_1d_vis(d)
        t = np.linspace(-2.2, 7, 600)
        ax.plot(r["xte"], r["tte"], "o", color=ORANGE, ms=4.5, alpha=0.55, mec="none", label="test")
        ax.plot(r["xtr"], r["ttr"], "o", color=BLUE, ms=6, mec="white", mew=0.8, label="training")
        ax.plot(t, poly_design(t, d) @ r["w"], color=VIOLET, lw=2, label="fit")
        ax.axvspan(r["xtr"].min(), r["xtr"].max(), color="#f1f0ec", zorder=0)
        ax.set(xlim=(-2.2, 7), ylim=(-5, 60), xlabel=f"INDUS, normalized  (x = (INDUS − {r['mu']:.2f}) / {r['sd']:.2f})")
        ax.set_title(f"Degree {d}\ntrain RMS {r['train']:.2f},  test RMS {r['test']:.3g}")
    axes[0].set_ylabel("MEDV ($1000s)")
    h, l = axes[0].get_legend_handles_labels()
    h[2], l[2] = axes[0].lines[2], "learned polynomial"
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    save(fig, "p1_vis.png")


def cv_plot(mean, errs, name, title):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    lam = np.array(LAMBDAS[1:])
    for j in range(errs.shape[1]):
        ax.plot(lam, errs[1:, j], color=MUTED, lw=0.8, alpha=0.5, label="individual folds" if j == 0 else None)
    ax.plot(lam, mean[1:], "-o", color=BLUE, lw=2, ms=7, mec="white", label="mean over 10 folds")
    ax.axhline(mean[0], color=ORANGE, lw=1.6, ls="--", label=f"λ = 0 baseline ({mean[0]:.1f})")
    best = int(np.argmin(mean))
    ax.plot(LAMBDAS[best], mean[best], "*", color=VIOLET, ms=16, mec="white", zorder=5,
            label=f"best λ = {LAMBDAS[best]:g} ({mean[best]:.2f})")
    for l, m in zip(lam, mean[1:]):
        ax.annotate(f"{m:.2f}", (l, m), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="regularization λ (log scale)", ylabel="validation RMS ($1000s, log scale)", title=title)
    ax.legend(loc="best", fontsize=8.5)
    save(fig, name)


def gauss_errors():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for seed in range(1, 10):
        Ks, train, test = gaussian_regression(seed=seed)
        ax.plot(Ks, train, color=BLUE, lw=0.8, alpha=0.2)
        ax.plot(Ks, test, color=ORANGE, lw=0.8, alpha=0.2)
    Ks, train, test = gaussian_regression()
    ax.plot(Ks, train, "-o", color=BLUE, lw=2, ms=7, mec="white", label="training RMS")
    ax.plot(Ks, test, "-s", color=ORANGE, lw=2, ms=7, mec="white", label="test RMS")
    ax.plot([], [], color=MUTED, lw=0.8, label="9 other random center draws")
    ax.set_yscale("log")
    ax.set(xticks=Ks, xlabel="number of Gaussian basis functions K", ylabel="RMS error ($1000s, log scale)",
           title="Gaussian basis regression, all 13 features, s = 2, no regularization")
    ax.legend(loc="lower left")
    save(fig, "p2_gauss_errors.png")


if __name__ == "__main__":
    poly_errors()
    weights()
    vis()
    cv_plot(*polynomial_regression_reg(), "p1_reg_cv.png",
            "10-fold CV, degree-8 polynomial on INDUS")
    gauss_errors()
    cv_plot(*gaussian_regression_reg(), "p2_reg_cv.png",
            "10-fold CV, 90 Gaussian basis functions, all features")

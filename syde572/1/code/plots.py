import math
import numpy as np
import matplotlib.pyplot as plt
from part1 import newton, golden, analytic, parabola, derivs
from part2 import x as px, y as py, design, analytic as fit, newton as fit_newton, mse

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
POINTS = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
f, df, ddf = parabola(1, 0, 5)


def save(fig, name):
    fig.savefig(OUT + name)
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    xs = np.linspace(-3.2, 3.2, 300)
    ax.plot(xs, f(xs), color=BLUE, lw=2, label=r"$y = x^2 + 5$")
    for x0, y0 in POINTS:
        d, x = analytic(x0, y0, 1, 0, 5)
        ax.plot([x0, x], [y0, f(x)], color=ORANGE, lw=1.4, ls="--")
        ax.plot(x, f(x), "o", color=BLUE, ms=7, mec="white", mew=1.5)
        ax.plot(x0, y0, "o", color=ORANGE, ms=8, mec="white", mew=1.5)
        dy = -52 if x0 == 2 else -30
        ax.annotate(f"({x0}, {y0})\nd = {d:.3f}", (x0, y0), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_aspect("equal")
    ax.set(xlim=(-9.5, 7.5), ylim=(-4, 11), xlabel="x", ylabel="y",
           title="Closest point on the parabola for each query point")
    ax.legend(loc="upper left")
    save(fig, "p1_overview.png")


def newton_steps(x0=-8, y0=0):
    _, xs_, path = newton(x0, y0, f, df, ddf, x=x0)
    g = lambda x: 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
    h = lambda x: 2 + 2 * df(x) ** 2 + 2 * (f(x) - y0) * ddf(x)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    t = np.linspace(-8.5, 1, 400)
    a1.plot(t, g(t), color=BLUE, lw=2, label=r"$D'(x) = 2(x-x_0) + 2(f(x)-y_0)f'(x)$")
    a1.axhline(0, color=MUTED, lw=1)
    for k, xk in enumerate(path[:5]):
        xn = xk - g(xk) / h(xk)
        a1.plot([xk, xn], [g(xk), 0], color=ORANGE, lw=1.3)
        a1.plot([xk, xk], [0, g(xk)], color=MUTED, lw=0.8, ls=":")
        a1.plot(xk, g(xk), "o", color=ORANGE, ms=6, mec="white")
        a1.annotate(f"$x_{k}$", (xk, 0), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    a1.set(xlabel="x", ylabel="D'(x)", title=f"Newton–Raphson on D'(x) = 0, point ({x0}, {y0})")
    a1.legend(loc="lower right", fontsize=8.5)
    a2.plot(t, f(t), color=BLUE, lw=2, label=r"$y = x^2 + 5$")
    a2.plot(x0, y0, "o", color=ORANGE, ms=8, mec="white")
    for k, xk in enumerate(path):
        a2.plot([x0, xk], [y0, f(xk)], color=ORANGE, lw=1, alpha=0.25 + 0.75 * k / len(path))
        a2.plot(xk, f(xk), "o", color=VIOLET, ms=5, mec="white")
    a2.plot([x0, xs_], [y0, f(xs_)], color=ORANGE, lw=2, label=f"d* = {math.hypot(xs_ - x0, f(xs_)):.4f}")
    for k in range(3):
        a2.annotate(f"$x_{k}$", (path[k], f(path[k])), xytext=(8, -4), textcoords="offset points", fontsize=9)
    a2.set(xlim=(-9, 2), ylim=(-2, 30), xlabel="x", ylabel="y", title="Iterates on the curve")
    a2.legend(loc="upper right")
    save(fig, "p1_newton.png")


def golden_steps(x0=-8, y0=0, a=-10, b=10):
    _, xs_, path = golden(x0, y0, f, a, b)
    D = lambda x: (x - x0) ** 2 + (f(x) - y0) ** 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    t = np.linspace(a, b, 400)
    a1.plot(t, D(t), color=BLUE, lw=2, label=r"$D(x) = (x-x_0)^2 + (f(x)-y_0)^2$")
    a1.set_yscale("log")
    a1.set(xlabel="x", ylabel="D(x)  (log scale)", title=f"Golden-section search, point ({x0}, {y0})")
    for k, (lo, hi, x1, x2) in enumerate(path[:8]):
        a1.plot([x1, x2], [D(x1), D(x2)], "o", color=plt.cm.Oranges(0.3 + 0.09 * k), ms=6, mec="white",
                label="probe points x₁, x₂ (darker = later)" if k == 7 else None)
    a1.legend(loc="upper center", fontsize=8.5)
    for k, (lo, hi, x1, x2) in enumerate(path[:12]):
        a2.plot([lo, hi], [k, k], color=BLUE, lw=5, solid_capstyle="round", alpha=0.85)
        a2.plot([x1, x2], [k, k], "|", color=ORANGE, ms=10, mew=2)
    a2.axvline(xs_, color=VIOLET, lw=1, ls="--", label=f"x* = {xs_:.4f}")
    a2.invert_yaxis()
    a2.set(xlabel="x", ylabel="iteration", title="Bracket [a, b] shrinks by 0.618 each step")
    a2.legend(loc="lower left")
    save(fig, "p1_golden.png")


def convergence():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (x0, y0), c in zip(POINTS[1:], [BLUE, ORANGE, AQUA, VIOLET]):
        _, xs_, pn = newton(x0, y0, f, df, ddf, x=x0)
        ax.plot(range(len(pn)), [abs(p - xs_) + 1e-16 for p in pn], "-o", color=c, ms=4, lw=2, label=f"Newton ({x0}, {y0})")
    _, _, pg = golden(0, 0, f, -10, 10)
    ax.plot(range(len(pg)), [hi - lo for lo, hi, _, _ in pg], "--", color=MUTED, lw=1.6, label="Golden section (any point)")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e2)
    ax.set(xlabel="iteration", ylabel="error in x", title="Convergence: Newton error vs golden-section bracket width")
    ax.legend(ncol=2, fontsize=8.5)
    save(fig, "p1_convergence.png")


def poly_name(a, b, c):
    terms = [(a, "x²"), (b, "x"), (c, "")]
    out = ""
    for v, t in terms:
        if v == 0:
            continue
        sign = "−" if v < 0 else "+"
        mag = "" if abs(v) == 1 and t else f"{abs(v):g}"
        out += f"{sign if out or v < 0 else ''}{' ' if out else ''}{mag}{t} "
    return "y = " + out.strip()


def other_parabolas():
    cases = [((0.5, -2, 1), (3, 4)), ((0.5, -2, 1), (0, -2)), ((-1, 0, 4), (0, 0)), ((2, 3, -1), (-3, 5))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, ((a, b, c), (x0, y0)) in zip(axes.flat, cases):
        fn, dfn, ddfn = parabola(a, b, c)
        dn, xn, pn = newton(x0, y0, fn, dfn, ddfn, x=x0 + 1)
        dg, xg, _ = golden(x0, y0, fn, x0 - 5, x0 + 5)
        t = np.linspace(x0 - 4, x0 + 4, 300)
        ax.plot(t, fn(t), color=BLUE, lw=2)
        ax.plot([x0, xn], [y0, fn(xn)], color=ORANGE, lw=1.6, ls="--")
        ax.plot(x0, y0, "o", color=ORANGE, ms=8, mec="white")
        ax.plot(xn, fn(xn), "o", color=VIOLET, ms=7, mec="white", label=f"Newton  d = {dn:.4f}")
        ax.plot([x0, xg], [y0, fn(xg)], color=AQUA, lw=1.6, ls=":")
        ax.plot(pn[:-1], [fn(p) for p in pn[:-1]], "o", color=VIOLET, ms=4, alpha=0.4)
        ax.plot(xg, fn(xg), "x", color=AQUA, ms=9, mew=2, label=f"Golden  d = {dg:.4f}")
        ax.set(xlim=(x0 - 4, x0 + 4), ylim=(y0 - 3.5, y0 + 3.5))
        ax.set_aspect("equal")
        ax.set_title(f"{poly_name(a, b, c)},  point ({x0}, {y0})")
        ax.legend(loc="best", fontsize=8.5)
    save(fig, "p1_parabolas.png")


def nonpoly():
    cases = [("y = eˣ", np.exp, (0, 0), 0.0, (-3, 3), (-2.5, 1.5)),
             ("y = ln x", np.log, (0, 0), 1.0, (0.01, 3), (0.05, 3)),
             ("y = 1/x", lambda x: 1 / x, (0, 0), 2.0, (0.1, 5), (0.25, 4)),
             ("y = √x", np.sqrt, (4, 0), 4.0, (0, 8), (0, 7)),
             ("y = 1/(1+x²)", lambda x: 1 / (1 + x * x), (2, 2), 2.0, (-3, 5), (-3, 4)),
             ("y = sin x", np.sin, (1, 2), 1.0, (-1, 3), (-2, 4))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    for ax, (name, fn, (x0, y0), guess, (a, b), (lo, hi)) in zip(axes.flat, cases):
        dfn, ddfn = derivs(fn)
        dn, xn, pn = newton(x0, y0, fn, dfn, ddfn, x=guess)
        dg, xg, _ = golden(x0, y0, fn, a, b)
        t = np.linspace(lo, hi, 300)
        ax.plot(t, fn(t), color=BLUE, lw=2)
        ax.plot(pn[:-1], [fn(p) for p in pn[:-1]], "o", color=VIOLET, ms=4, alpha=0.4)
        ax.plot([x0, xn], [y0, fn(xn)], color=ORANGE, lw=1.6, ls="--")
        ax.plot(x0, y0, "o", color=ORANGE, ms=8, mec="white")
        ax.plot(xn, fn(xn), "o", color=VIOLET, ms=7, mec="white", label=f"Newton d = {dn:.4f}")
        ax.plot(xg, fn(xg), "x", color=AQUA, ms=9, mew=2, label=f"Golden d = {dg:.4f}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{name},  point ({x0}, {y0})")
        ax.legend(loc="best", fontsize=8.5)
    save(fig, "p1_nonpoly.png")


def fits():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    t = np.linspace(-0.3, 3.3, 200)
    for deg, c, name in [(1, BLUE, "line"), (2, AQUA, "parabola")]:
        p, e = fit(deg)
        ax.plot(t, np.polyval(p, t), color=c, lw=2, label=f"{name}  MSE = {e:.4f}")
        for xi, yi in zip(px, py):
            ax.plot([xi, xi], [yi, np.polyval(p, xi)], color=c, lw=1, ls=":")
    ax.plot(px, py, "o", color=ORANGE, ms=9, mec="white", mew=1.5, label="data", zorder=5)
    ax.set(xlabel="x", ylabel="y", title="Least-squares fits (dotted = residuals)")
    ax.legend(loc="upper left")
    save(fig, "p2_fits.png")


def line_path():
    _, _, path = fit_newton(1)
    X = design(px, 1)
    M, B = np.meshgrid(np.linspace(-0.5, 3.2, 200), np.linspace(-2, 3, 200))
    E = np.mean((M[..., None] * px + B[..., None] - py) ** 2, axis=-1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    cs = a1.contour(M, B, E, levels=np.geomspace(0.6, 40, 14), cmap="Blues_r", linewidths=1)
    a1.clabel(cs, fontsize=7, fmt="%.1f")
    a1.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.5, lw=1.4, label="coordinate Newton path")
    a1.plot(2.3, -0.2, "*", color=VIOLET, ms=14, mec="white", label="optimum (2.3, −0.2)")
    a1.set(xlabel="slope m", ylabel="intercept b", title="MSE(m, b) contours and coordinate steps")
    a1.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    M, B = np.meshgrid(np.linspace(2.18, 2.34, 200), np.linspace(-0.26, 0.03, 200))
    E = np.mean((M[..., None] * px + B[..., None] - py) ** 2, axis=-1)
    a2.contour(M, B, E, levels=np.geomspace(0.5752, 0.6, 12), cmap="Blues_r", linewidths=1)
    a2.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=4, lw=1.4)
    for k in range(1, 6):
        a2.annotate(f"{k}", path[k], xytext=(6, 4 if k % 2 else -12), textcoords="offset points", fontsize=8, color=INK)
    a2.plot(2.3, -0.2, "*", color=VIOLET, ms=14, mec="white")
    a2.set(xlabel="slope m", ylabel="intercept b", xlim=(2.19, 2.33), ylim=(-0.23, 0.02), title="Zoom: alternating m-step then b-step")
    save(fig, "p2_line.png")


def parabola_path():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    _, _, path = fit_newton(2)
    X = design(px, 2)
    t = np.linspace(-0.3, 3.3, 100)
    for i, k in enumerate([1, 2, 5, 20]):
        a1.plot(t, np.polyval(path[3 * k], t), color=plt.cm.Greens(0.3 + 0.15 * i), lw=1.4, label=f"sweep {k}")
    a1.plot(t, np.polyval(path[-1], t), color=VIOLET, lw=2, label=f"final  MSE = {mse(X, path[-1]):.4f}")
    a1.plot(px, py, "o", color=ORANGE, ms=9, mec="white", zorder=5)
    a1.set(xlabel="x", ylabel="y", xlim=(-0.3, 3.9), title="Intermediate parabolas")
    a1.legend(loc="upper left")
    for deg, c, name in [(1, BLUE, "line"), (2, AQUA, "parabola")]:
        _, e_star, path = fit_newton(deg)
        Xd = design(px, deg)
        sweeps = path[:: deg + 1]
        a2.plot(range(len(sweeps)), [mse(Xd, p) - e_star + 1e-16 for p in sweeps], color=c, lw=2, label=name)
    a2.set_yscale("log")
    a2.set_ylim(1e-14, 1e2)
    a2.set(xlabel="sweep (all parameters updated once)", ylabel="MSE − MSE*", title="Coordinate Newton convergence")
    a2.legend()
    save(fig, "p2_parabola.png")


if __name__ == "__main__":
    for fn in [overview, newton_steps, golden_steps, convergence, other_parabolas, nonpoly, fits, line_path, parabola_path]:
        fn()

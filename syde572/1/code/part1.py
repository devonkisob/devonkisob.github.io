import math
import numpy as np


def newton(x0, y0, f, df, ddf, x=0.0, tol=1e-7, max_iter=100):
    path = [x]
    for _ in range(max_iter):
        g = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        h = 2 + 2 * df(x) ** 2 + 2 * (f(x) - y0) * ddf(x)
        x_new = x - g / h
        path.append(x_new)
        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return math.hypot(x - x0, f(x) - y0), x, path


def golden(x0, y0, f, a, b, tol=1e-7):
    r = (3 - math.sqrt(5)) / 2
    D = lambda x: (x - x0) ** 2 + (f(x) - y0) ** 2
    x1, x2 = a + r * (b - a), b - r * (b - a)
    D1, D2 = D(x1), D(x2)
    path = [(a, b, x1, x2)]
    while b - a > tol:
        if D1 < D2:
            b, x2, D2 = x2, x1, D1
            x1 = a + r * (b - a)
            D1 = D(x1)
        else:
            a, x1, D1 = x1, x2, D2
            x2 = b - r * (b - a)
            D2 = D(x2)
        path.append((a, b, x1, x2))
    x = (a + b) / 2
    return math.sqrt(D(x)), x, path


def analytic(x0, y0, a, b, c):
    k = c - y0
    roots = np.roots([2 * a * a, 3 * a * b, b * b + 2 * a * k + 1, b * k - x0])
    xs = roots[abs(roots.imag) < 1e-9].real
    d = np.hypot(xs - x0, a * xs ** 2 + b * xs + c - y0)
    return d.min(), xs[d.argmin()]


def parabola(a, b, c):
    f = lambda x: a * x * x + b * x + c
    df = lambda x: 2 * a * x + b
    ddf = lambda x: 2 * a
    return f, df, ddf


def derivs(f, h=1e-4):
    return (lambda x: (f(x + h) - f(x - h)) / (2 * h),
            lambda x: (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2)


if __name__ == "__main__":
    f, df, ddf = parabola(1, 0, 5)
    for x0, y0 in [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]:
        print((x0, y0), analytic(x0, y0, 1, 0, 5)[0],
              newton(x0, y0, f, df, ddf, x=x0)[0],
              golden(x0, y0, f, -10, 10)[0])

    for (a, b, c), (x0, y0) in [((0.5, -2, 1), (3, 4)), ((0.5, -2, 1), (0, -2)),
                                ((-1, 0, 4), (0, 0)), ((2, 3, -1), (-3, 5))]:
        f, df, ddf = parabola(a, b, c)
        print((a, b, c), (x0, y0), analytic(x0, y0, a, b, c)[0],
              newton(x0, y0, f, df, ddf, x=x0 + 1)[0],
              golden(x0, y0, f, x0 - 5, x0 + 5)[0])

    for f, (x0, y0), guess, (a, b) in [(math.exp, (0, 0), 0.0, (-3, 3)),
                                       (math.log, (0, 0), 1.0, (0.01, 3)),
                                       (lambda x: 1 / x, (0, 0), 2.0, (0.1, 5)),
                                       (math.sqrt, (4, 0), 4.0, (0, 8)),
                                       (lambda x: 1 / (1 + x * x), (2, 2), 2.0, (-3, 5)),
                                       (math.sin, (1, 2), 1.0, (-1, 3))]:
        df, ddf = derivs(f)
        print((x0, y0), newton(x0, y0, f, df, ddf, x=guess)[0], golden(x0, y0, f, a, b)[0])

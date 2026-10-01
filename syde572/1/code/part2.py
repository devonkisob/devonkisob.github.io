import numpy as np

x = np.array([0, 2, 1, 3.0])
y = np.array([0.5, 3.5, 1.5, 7.5])


def design(x, degree):
    return np.vstack([x ** k for k in range(degree, -1, -1)]).T


def mse(X, p):
    return np.mean((X @ p - y) ** 2)


def analytic(degree):
    X = design(x, degree)
    p = np.linalg.solve(X.T @ X, X.T @ y)
    return p, mse(X, p)


def newton(degree, tol=1e-10, max_sweeps=500):
    X = design(x, degree)
    n = len(y)
    p = np.zeros(degree + 1)
    path = [p.copy()]
    for _ in range(max_sweeps):
        old = p.copy()
        for j in range(len(p)):
            g = 2 / n * X[:, j] @ (X @ p - y)
            h = 2 / n * X[:, j] @ X[:, j]
            p[j] -= g / h
            path.append(p.copy())
        if np.abs(p - old).max() < tol:
            break
    return p, mse(X, p), np.array(path)


def full_newton(degree):
    X = design(x, degree)
    p = np.zeros(degree + 1)
    g = 2 / len(y) * X.T @ (X @ p - y)
    H = 2 / len(y) * X.T @ X
    return p - np.linalg.solve(H, g)


if __name__ == "__main__":
    for degree in (1, 2):
        p, e = analytic(degree)
        pn, en, path = newton(degree)
        print("analytic", p.round(6), e)
        print("newton  ", pn.round(6), en, len(path) - 1, "steps")
        print("full    ", full_newton(degree).round(6))

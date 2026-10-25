import os
import numpy as np

NAMES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]
N_TRAIN = 100
FEATURE = 2   # X[:, 2] = INDUS (0-indexed, matching w5 = 6th feature)
S = 2.0       # Gaussian basis width, in normalized units
SEED = 0


def load(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "housing.data")):
    data = np.loadtxt(path)
    return data[:, :13], data[:, 13]


def normalize(X_train, X_test):
    mu, sd = X_train.mean(0), X_train.std(0)
    sd[sd == 0] = 1   # CHAS is 0 for all of the first 100 towns
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def split(X, t, n=N_TRAIN):
    Xtr, Xte, mu, sd = normalize(X[:n], X[n:])
    return Xtr, t[:n], Xte, t[n:], mu, sd


def poly_design(X, degree):
    X = X.reshape(len(X), -1)
    return np.hstack([np.ones((len(X), 1))] + [X ** k for k in range(1, degree + 1)])


def gauss_design(X, centers, s=S):
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return np.hstack([np.ones((len(X), 1)), np.exp(-d2 / (2 * s * s))])


def fit(Phi, t, lam=0.0):
    if lam == 0:
        return np.linalg.lstsq(Phi, t, rcond=None)[0]
    I = np.eye(Phi.shape[1])
    I[0, 0] = 0   # do not penalize the bias
    return np.linalg.solve(lam * I + Phi.T @ Phi, Phi.T @ t)


def rms(Phi, w, t):
    return np.sqrt(np.mean((Phi @ w - t) ** 2))


def cross_validate(design, X, t, lambdas=LAMBDAS, k=10):
    folds = np.array_split(np.arange(len(X)), k)
    errs = np.zeros((len(lambdas), k))
    for j, val in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(X)), val)
        Phi_tr, Phi_val = design(X[tr]), design(X[val])
        for i, lam in enumerate(lambdas):
            errs[i, j] = rms(Phi_val, fit(Phi_tr, t[tr], lam), t[val])
    return errs.mean(1), errs


def polynomial_regression(degrees=range(1, 8)):
    Xtr, ttr, Xte, tte, _, _ = split(*load())
    train, test, weights = [], [], []
    for d in degrees:
        w = fit(poly_design(Xtr, d), ttr)
        train.append(rms(poly_design(Xtr, d), w, ttr))
        test.append(rms(poly_design(Xte, d), w, tte))
        weights.append(w)
    return list(degrees), np.array(train), np.array(test), weights


def polynomial_regression_1d_vis(degree, feature=FEATURE):
    Xtr, ttr, Xte, tte, mu, sd = split(*load())
    xtr, xte = Xtr[:, feature], Xte[:, feature]
    w = fit(poly_design(xtr, degree), ttr)
    return dict(w=w, xtr=xtr, ttr=ttr, xte=xte, tte=tte, mu=mu[feature], sd=sd[feature],
                train=rms(poly_design(xtr, degree), w, ttr), test=rms(poly_design(xte, degree), w, tte))


def polynomial_regression_reg(degree=8, feature=FEATURE, lambdas=LAMBDAS):
    Xtr, ttr, _, _, _, _ = split(*load())
    return cross_validate(lambda X: poly_design(X[:, feature], degree), Xtr, ttr, lambdas)


def centers(Xtr, K, seed=SEED):
    return Xtr[np.random.default_rng(seed).permutation(len(Xtr))[:K]]


def gaussian_regression(Ks=range(5, 100, 10), seed=SEED):
    Xtr, ttr, Xte, tte, _, _ = split(*load())
    train, test = [], []
    for K in Ks:
        mu = centers(Xtr, K, seed)
        w = fit(gauss_design(Xtr, mu), ttr)
        train.append(rms(gauss_design(Xtr, mu), w, ttr))
        test.append(rms(gauss_design(Xte, mu), w, tte))
    return list(Ks), np.array(train), np.array(test)


def gaussian_regression_reg(K=90, lambdas=LAMBDAS, seed=SEED):
    Xtr, ttr, _, _, _, _ = split(*load())
    mu = centers(Xtr, K, seed)
    return cross_validate(lambda X: gauss_design(X, mu), Xtr, ttr, lambdas)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    degrees, train, test, weights = polynomial_regression()
    print("(a) degree  train RMS  test RMS")
    for d, a, b in zip(degrees, train, test):
        print(f"    {d}       {a:8.4f}  {b:10.4f}")

    _, _, _, _, mu, sd = split(*load())
    w = weights[0]
    print("(b) w0 (bias) =", round(w[0], 4))
    for i, name in enumerate(NAMES):
        print(f"    w{i + 1:<2} {name:8s} normalized {w[i + 1]:9.4f}   per unit {w[i + 1] / sd[i]:10.5f}")

    for d in (1, 3, 5, 7, 10):
        r = polynomial_regression_1d_vis(d)
        print(f"(c) degree {d}: train {r['train']:.4f}  test {r['test']:.4f}")

    for name, f in [("(d)", polynomial_regression_reg), ("(f)", gaussian_regression_reg)]:
        mean, _ = f()
        print(name, dict(zip(LAMBDAS, mean.round(4))), "best lambda =", LAMBDAS[mean.argmin()])

    Ks, train, test = gaussian_regression()
    print("(e) K  train RMS  test RMS")
    for K, a, b in zip(Ks, train, test):
        print(f"    {K:<3} {a:8.4f}  {b:9.4f}")

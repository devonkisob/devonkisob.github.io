import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")   # k = 3 graph is disconnected; sklearn joins the pieces
FEATURES = ["sepal length", "sepal width", "petal length", "petal width"]


def pca_by_hand(X, k=1):
    mu = X.mean(0)
    S = (X - mu).T @ (X - mu) / len(X)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    return mu, vals[order], vecs[:, order][:, :k]


def reconstruct(x, mu, W):
    z = W.T @ (x - mu)
    x_hat = mu + W @ z
    return z, x_hat, np.linalg.norm(x - x_hat)


def iris():
    data = load_iris()
    return data.data, data.target, list(data.target_names)


def pca(X, k=4):
    model = PCA(n_components=k).fit(X)
    return model, model.transform(X)


def explained(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s ** 2 / np.linalg.norm(Xc, "fro") ** 2


def isomap(X, k=3, d=2):
    model = Isomap(n_neighbors=k, n_components=d).fit(X)
    return model, model.embedding_


def isomap_smacof(iso, d=2):
    # metric MDS by iterative stress minimization on the same geodesic distances
    mds = MDS(n_components=d, dissimilarity="precomputed", n_init=4, random_state=0, normalized_stress="auto")
    return mds.fit_transform(iso.dist_matrix_)


def residual_variance(D_high, Y):
    D_low = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
    iu = np.triu_indices(len(Y), 1)
    return 1 - np.corrcoef(D_high[iu], D_low[iu])[0, 1] ** 2


def knn_accuracy(Z, y, k=5):
    return cross_val_score(KNeighborsClassifier(k), Z.reshape(len(Z), -1), y, cv=10).mean()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    P = np.array([[-2, -1], [-1, 0], [0, 1], [1, 2], [2, 3]], float)
    mu, vals, W = pca_by_hand(P)
    print("Part 1: mean", mu, "eigenvalues", vals, "PC1", W.ravel())
    for x in ([3, 3], [3, 4]):
        z, x_hat, err = reconstruct(np.array(x, float), mu, W)
        print(f"  {x}: z = {z[0]:.4f}, x_hat = {x_hat}, error = {err:.4f}")

    X, y, names = iris()
    model, Z = pca(X)
    print("Part 3: mean", model.mean_)
    for i, w in enumerate(model.components_):
        print(f"  PC{i + 1}", w, "variance", round(model.explained_variance_[i], 4))

    r = explained(X)
    print("Part 4: explained", r, "cumulative", np.cumsum(r))
    print("  PCs for 50%:", int(np.searchsorted(np.cumsum(r), 0.5) + 1))

    iso, Y = isomap(X)
    D_geo = iso.dist_matrix_
    D_euc = np.linalg.norm(X[:, None] - X[None], axis=-1)
    for d in (1, 2):
        Yd = isomap(X, d=d)[1]
        print(f"Part 5: {d}-D  isomap residual var {residual_variance(D_geo, Yd):.4f}  "
              f"pca residual var {residual_variance(D_euc, Z[:, :d]):.4f}  "
              f"5-NN acc isomap {knn_accuracy(Yd, y):.4f}  pca {knn_accuracy(Z[:, :d], y):.4f}")
    Ys = isomap_smacof(iso)
    print(f"  2-D  isomap+SMACOF residual var {residual_variance(D_geo, Ys):.4f}  5-NN acc {knn_accuracy(Ys, y):.4f}")
    print("  raw 4-D 5-NN acc", round(knn_accuracy(X, y), 4))

import os
from collections import Counter
import numpy as np
from scipy.stats import binomtest

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "all_sentiment_shuffled.txt")
ALPHAS = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
NEGATIONS = {"not", "no", "never", "n't", "cannot", "nothing", "nobody", "none", "nor", "without"}
PUNCT = {".", ",", "!", "?", ";", ":"}


# ---------- data ----------

def read_documents(doc_file=DATA, label_col=1):
    docs, labels = [], []
    with open(doc_file, encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[label_col])
    return docs, labels


def split(docs, labels, frac=0.8):
    k = int(frac * len(docs))
    return docs[:k], labels[:k], docs[k:], labels[k:]


# ---------- Naive Bayes (multinomial, additive smoothing) ----------

def train_nb(documents, labels, alpha=1.0):
    classes = sorted(set(labels))
    counts = {c: Counter() for c in classes}
    for doc, c in zip(documents, labels):
        counts[c].update(doc)
    vocab = set().union(*counts.values())
    n = Counter(labels)
    return dict(classes=classes, counts=counts, totals={c: sum(counts[c].values()) for c in classes},
                V=len(vocab), alpha=alpha, log_prior={c: np.log(n[c] / len(labels)) for c in classes})


def with_alpha(model, alpha):
    return {**model, "alpha": alpha}


def log_p_word(w, c, model):
    a = model["alpha"]
    return np.log((model["counts"][c][w] + a) / (model["totals"][c] + a * model["V"]))


def score_doc_label(document, label, model):
    return model["log_prior"][label] + sum(log_p_word(w, label, model) for w in document)


def classify_nb(document, model):
    return max(model["classes"], key=lambda c: score_doc_label(document, c, model))


def classify_documents(docs, model):
    return [classify_nb(d, model) for d in docs]


def accuracy(true_labels, guessed_labels):
    return np.mean([t == g for t, g in zip(true_labels, guessed_labels)])


def posterior(document, model):
    s = np.array([score_doc_label(document, c, model) for c in model["classes"]])
    return dict(zip(model["classes"], np.exp(s - np.logaddexp.reduce(s))))


# ---------- statistics ----------

def wilson(acc, n, z=1.96):
    centre = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar(true, a, b):
    a_only = sum(x == t != y for t, x, y in zip(true, a, b))
    b_only = sum(y == t != x for t, x, y in zip(true, a, b))
    return a_only, b_only, binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0


def cv_alpha(docs, labels, alphas=ALPHAS, k=5):
    folds = np.array_split(np.arange(len(docs)), k)
    acc = np.zeros((len(alphas), k))
    for j, val in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(docs)), val)
        model = train_nb([docs[i] for i in tr], [labels[i] for i in tr])
        for i, a in enumerate(alphas):
            acc[i, j] = accuracy([labels[v] for v in val], classify_documents([docs[v] for v in val], with_alpha(model, a)))
    return acc.mean(1)


# ---------- optional tasks ----------

def print_top_nb(k, model, min_count=0):
    pos, neg = model["classes"][1], model["classes"][0]
    words = [w for w in set(model["counts"][pos]) | set(model["counts"][neg])
             if model["counts"][pos][w] + model["counts"][neg][w] >= min_count]
    lr = {w: log_p_word(w, pos, model) - log_p_word(w, neg, model) for w in words}
    ranked = sorted(lr, key=lr.get)
    top_pos, top_neg = ranked[::-1][:k], ranked[:k]
    print(f"{pos}:", ", ".join(top_pos))
    print(f"{neg}:", ", ".join(top_neg))
    return [(w, lr[w]) for w in top_pos], [(w, lr[w]) for w in top_neg]


def learning_curve(train_docs, train_labels, eval_docs, eval_labels, alpha, fracs=np.arange(0.1, 1.01, 0.1)):
    out = []
    for f in fracs:
        n = int(f * len(train_docs))
        model = train_nb(train_docs[:n], train_labels[:n], alpha)
        out.append((n, accuracy(eval_labels, classify_documents(eval_docs, model))))
    return out


def mark_negation(doc):
    out, neg = [], False
    for w in doc:
        if w in PUNCT:
            neg = False
            out.append(w)
        else:
            out.append("NOT_" + w if neg else w)
            if w in NEGATIONS or w.endswith("n't"):
                neg = True
    return out


def add_bigrams(doc):
    return doc + [a + "_" + b for a, b in zip(doc, doc[1:])]


def domain_matrix(docs, labels, topics, domains, alpha):
    data = {}
    for d in domains:
        idx = [i for i, t in enumerate(topics) if t == d]
        data[d] = split([docs[i] for i in idx], [labels[i] for i in idx])
    acc = {}
    for src in domains:
        model = train_nb(data[src][0], data[src][1], alpha)
        for dst in domains:
            acc[src, dst] = accuracy(data[dst][3], classify_documents(data[dst][2], model))
    return acc


def train_sklearn(docs, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None)
    return make_pipeline(vec, LinearSVC()).fit(docs, labels)


# ---------- Gaussian Naive Bayes for numerical data ----------

def train_gnb(X, y):
    classes = sorted(set(y))
    y = np.asarray(y)
    return dict(classes=classes,
                log_prior={c: np.log(np.mean(y == c)) for c in classes},
                mean={c: X[y == c].mean(0) for c in classes},
                var={c: X[y == c].var(0) for c in classes})   # maximum likelihood (divide by n)


def score_gnb(x, c, model):
    m, v = model["mean"][c], model["var"][c]
    return model["log_prior"][c] + np.sum(-0.5 * np.log(2 * np.pi * v) - (x - m) ** 2 / (2 * v))


def classify_gnb(X, model):
    return [max(model["classes"], key=lambda c: score_gnb(x, c, model)) for x in X]


if __name__ == "__main__":
    all_docs, all_labels = read_documents()
    train_docs, train_labels, eval_docs, eval_labels = split(all_docs, all_labels)
    print(f"{len(train_docs)} training, {len(eval_docs)} evaluation documents")

    cv = cv_alpha(train_docs, train_labels)
    alpha = ALPHAS[int(np.argmax(cv))]
    print("5-fold CV accuracy by alpha:", dict(zip(ALPHAS, cv.round(4))), "-> alpha =", alpha)
    model = train_nb(train_docs, train_labels, alpha)

    for w in ["great", "bad"]:
        print(f"sanity 1 [{w}]:", {c: f"{np.exp(score_doc_label([w], c, model)):.5f}" for c in model["classes"]})
    print("sanity 2:", {c: round(score_doc_label(["a", "top-quality", "performance"], c, model), 3) for c in model["classes"]})
    for d in (["great"], ["bad"], ["not", "good"], ["waste", "of", "money"], ["highly", "recommend"]):
        print("  classify", d, "->", classify_nb(d, model))

    guesses = classify_documents(eval_docs, model)
    acc = accuracy(eval_labels, guesses)
    print(f"accuracy {acc:.4f}, 95% Wilson CI {wilson(acc, len(eval_docs))}")

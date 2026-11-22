import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from nb import *

R = {}
docs, labels = read_documents()
topics = read_documents(label_col=0)[1]
tr, tl, ev, el = split(docs, labels)

# smoothing by 5-fold CV on the training part only
cv = cv_alpha(tr, tl)
alpha = ALPHAS[int(np.argmax(cv))]
R["cv"] = dict(alphas=ALPHAS, acc=cv.tolist(), alpha=alpha)
model = train_nb(tr, tl, alpha)

R["sanity"] = {w: {c: float(np.exp(score_doc_label([w], c, model))) for c in model["classes"]} for w in ("great", "bad")}
R["sanity2"] = {c: float(score_doc_label(["a", "top-quality", "performance"], c, model)) for c in model["classes"]}
small = [["great"], ["bad"], ["not", "good"], ["waste", "of", "money"], ["highly", "recommend"], ["not", "bad"]]
R["small"] = [(" ".join(d), classify_nb(d, model), posterior(d, model)["pos"]) for d in small]

guess = classify_documents(ev, model)
acc = accuracy(el, guess)
R["main"] = dict(acc=acc, ci=wilson(acc, len(ev)), n=len(ev),
                 confusion={f"{t}>{g}": sum(a == t and b == g for a, b in zip(el, guess)) for t in ("pos", "neg") for g in ("pos", "neg")})

# error analysis: short misclassified documents with the most confident wrong posterior
wrong = []
for d, t, g in zip(ev, el, guess):
    if t != g and len(d) <= 60:
        wrong.append((posterior(d, model)[g], " ".join(d), t, g, len(d)))
R["errors"] = sorted(wrong, reverse=True)[:8]

# per-topic accuracy
et = topics[len(tr):]
R["by_topic"] = {t: accuracy([l for l, x in zip(el, et) if x == t], [g for g, x in zip(guess, et) if x == t])
                 for t in sorted(set(topics))}

# six-category topic classifier
ttr, ttl, tev, tel = split(docs, topics)
tmodel = train_nb(ttr, ttl, 1.0)
tguess = classify_documents(tev, tmodel)
cls = tmodel["classes"]
R["topic"] = dict(acc=accuracy(tel, tguess), ci=wilson(accuracy(tel, tguess), len(tel)), classes=cls,
                  confusion=[[sum(a == t and b == g for a, b in zip(tel, tguess)) for g in cls] for t in cls])

# learning curve
R["curve"] = learning_curve(tr, tl, ev, el, alpha)

# informative features (high smoothing, as the hint suggests)
pos, neg = print_top_nb(20, with_alpha(model, 200))
R["features"] = dict(pos=pos, neg=neg, alpha=200)

# feature variants
variants = {"unigrams": lambda d: d, "negation marking": mark_negation, "unigrams + bigrams": add_bigrams,
            "negation + bigrams": lambda d: add_bigrams(mark_negation(d))}
R["variants"] = {}
for name, f in variants.items():
    g = classify_documents([f(d) for d in ev], train_nb([f(d) for d in tr], tl, alpha))
    a = accuracy(el, g)
    R["variants"][name] = dict(acc=a, ci=wilson(a, len(ev)), mcnemar=mcnemar(el, g, guess))
    if name == "negation + bigrams":
        best_guess = g

# scikit-learn LinearSVC, compared with McNemar's exact test
svm = list(train_sklearn(tr, tl).predict(ev))
a = accuracy(el, svm)
R["svm"] = dict(acc=a, ci=wilson(a, len(ev)), mcnemar=mcnemar(el, svm, guess), vs_best=mcnemar(el, svm, best_guess))

# domain sensitivity
dm = domain_matrix(docs, labels, topics, ["camera", "books"], alpha)
R["domain"] = {f"{s}>{d}": v for (s, d), v in dm.items()}
full = domain_matrix(docs, labels, topics, sorted(set(topics)), alpha)
R["domain_full"] = {f"{s}>{d}": v for (s, d), v in full.items()}

# Gaussian Naive Bayes on iris
iris = load_iris()
X, y = iris.data, iris.target_names[iris.target]
rng = np.random.default_rng(0)
gacc, sacc = [], []
for _ in range(100):
    e = np.zeros(len(y), bool)
    e[rng.choice(len(y), 40, replace=False)] = True
    gm = train_gnb(X[~e], y[~e])
    gacc.append(accuracy(y[e], classify_gnb(X[e], gm)))
    sacc.append(accuracy(y[e], GaussianNB().fit(X[~e], y[~e]).predict(X[e])))
R["gnb"] = dict(mean=float(np.mean(gacc)), std=float(np.std(gacc)), sk_mean=float(np.mean(sacc)),
                agree=float(np.mean(np.array(gacc) == np.array(sacc))),
                params={c: dict(mean=gm["mean"][c].tolist(), std=np.sqrt(gm["var"][c]).tolist()) for c in gm["classes"]})

json.dump(R, open("results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ("errors", "domain_full")}, indent=1, default=float))
for p, text, t, g, n in R["errors"]:
    print(f"\n[{t} -> {g}, P={p:.4f}, {n} words] {text}")
print(R["domain_full"])

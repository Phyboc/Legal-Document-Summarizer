"""Train a sentence-level section classifier on MARRO Indian rhetorical-role data."""
import glob, csv, json, sys, joblib, numpy as np, scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MAP = {"FAC": "facts", "RLC": "facts", "ARG": "arguments", "PRE": "reasoning",
       "STA": "reasoning", "Ratio": "reasoning", "RPC": "judgement"}
ROOT = "artifacts/marro"

def load(split):
    X, P, Y, G = [], [], [], []
    for f in sorted(glob.glob(f"{ROOT}/IN-{split}-set/**/*.txt", recursive=True)):
        rows = []
        for line in open(f, encoding="utf-8", errors="ignore"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[-1].strip() in MAP:
                rows.append((p[0].strip(), MAP[p[-1].strip()]))
        n = max(len(rows) - 1, 1)
        for i, (t, l) in enumerate(rows):
            X.append(t); P.append(i / n); Y.append(l); G.append(f)
    return X, np.array(P), np.array(Y), G

def posfeat(P):
    return np.c_[P, P**2, P > 0.85, P < 0.2]

def build(Xtr, Ptr):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    M = vec.fit_transform(Xtr)
    return vec, sp.hstack([M, sp.csr_matrix(posfeat(Ptr))]).tocsr()

def feats(vec, X, P):
    return sp.hstack([vec.transform(X), sp.csr_matrix(posfeat(P))]).tocsr()

def gold_rows():
    A = list(csv.DictReader(open("review/section_labels.a.csv", encoding="utf-8")))
    C = list(csv.DictReader(open("review/section_labels.claude.csv", encoding="utf-8")))
    m = lambda l: "arguments" if l.startswith("arguments") else l
    return [(a["text"], float(a["position"]), m(a["gold_label"])) for a, c in zip(A, C)
            if c["gold_label"] and a["gold_label"] == c["gold_label"] and a["gold_label"] != "issues"]

if __name__ == "__main__":
    Xtr, Ptr, Ytr, _ = load("train"); Xte, Pte, Yte, _ = load("test")
    print("train", len(Xtr), Counter(Ytr)); print("test", len(Xte))
    vec, Mtr = build(Xtr, Ptr)
    clf = LogisticRegression(C=5, max_iter=3000, class_weight="balanced").fit(Mtr, Ytr)
    from sklearn.metrics import classification_report
    pr = clf.predict(feats(vec, Xte, Pte))
    print("MARRO test accuracy", round(float(np.mean(pr == Yte)), 3))
    print(classification_report(Yte, pr, digits=3))
    g = gold_rows(); gy = np.array([r[2] for r in g])
    gp = clf.predict(feats(vec, [r[0] for r in g], np.array([r[1] for r in g])))
    print("YOUR gold (Claude+A agree, n=%d) accuracy" % len(g), round(float(np.mean(gp == gy)), 3))
    print(classification_report(gy, gp, digits=3, zero_division=0))
    joblib.dump((vec, clf), "models/section_classifier.joblib")

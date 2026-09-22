"""Train a sentence-level section classifier on MARRO + OpenNyAI InRhetoricalRoles data.

MARRO gives volume for facts/reasoning/judgement but collapses appellant vs
respondent arguments into one "ARG" tag. OpenNyAI InRhetoricalRoles (gated,
https://huggingface.co/datasets/opennyaiorg/InRhetoricalRoles) is the only
source with a genuine ARG_PETITIONER vs ARG_RESPONDENT split, so it supplies
the real training signal for that distinction. Both are combined here.
"""
import glob, csv, json, sys, joblib, numpy as np, scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# MARRO's "ARG" is intentionally excluded: it doesn't distinguish appellant vs
# respondent, and mixing an ambiguous label into those two classes would just
# blur the boundary OpenNyAI's ARG_PETITIONER/ARG_RESPONDENT actually supplies.
MAP = {"FAC": "facts", "RLC": "facts", "PRE": "reasoning",
       "STA": "reasoning", "Ratio": "reasoning", "RPC": "judgement"}
ROOT = "artifacts/marro"

# OpenNyAI uses the finer 13-role taxonomy; this preserves the ARG split MARRO discards.
OPENNYAI_MAP = {
    "FAC": "facts", "RLC": "facts",
    "ARG_PETITIONER": "arguments_appellant", "ARG_RESPONDENT": "arguments_respondent",
    "ANALYSIS": "reasoning", "STA": "reasoning", "PRE_RELIED": "reasoning",
    "PRE_NOT_RELIED": "reasoning", "RATIO": "reasoning",
    "RPC": "judgement",
    # PREAMBLE, ISSUE, NONE are dropped: no equivalent bucket in our 5-section taxonomy.
}

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


def load_opennyai(split):
    """Load OpenNyAI InRhetoricalRoles, mapped to our 5-section taxonomy with the
    real ARG_PETITIONER/ARG_RESPONDENT split preserved."""
    from datasets import load_dataset
    hf_split = {"train": "train", "test": "test"}.get(split)
    if hf_split is None:
        return [], np.array([]), np.array([]), []

    ds = load_dataset("opennyaiorg/InRhetoricalRoles")[hf_split]
    X, P, Y, G = [], [], [], []
    for doc_idx, example in enumerate(ds):
        rows = []
        for ann in example["annotations"]:
            for r in ann["result"]:
                raw_label = r["value"]["labels"][0] if r["value"]["labels"] else None
                mapped = OPENNYAI_MAP.get(raw_label)
                if mapped:
                    rows.append((r["value"]["text"].strip(), mapped))
        n = max(len(rows) - 1, 1)
        for i, (t, l) in enumerate(rows):
            if not t:
                continue
            X.append(t); P.append(i / n); Y.append(l); G.append(f"opennyai_{doc_idx}")
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
    # Keep the real appellant/respondent split (previously collapsed to "arguments",
    # which hid whether the model actually gets the split right on real judgments).
    return [(a["text"], float(a["position"]), a["gold_label"]) for a, c in zip(A, C)
            if c["gold_label"] and a["gold_label"] == c["gold_label"] and a["gold_label"] != "issues"]

if __name__ == "__main__":
    Xtr_m, Ptr_m, Ytr_m, _ = load("train"); Xte_m, Pte_m, Yte_m, _ = load("test")
    Xtr_o, Ptr_o, Ytr_o, _ = load_opennyai("train"); Xte_o, Pte_o, Yte_o, _ = load_opennyai("test")

    Xtr = Xtr_m + list(Xtr_o); Ptr = np.concatenate([Ptr_m, Ptr_o]); Ytr = np.concatenate([Ytr_m, Ytr_o])
    Xte = Xte_m + list(Xte_o); Pte = np.concatenate([Pte_m, Pte_o]); Yte = np.concatenate([Yte_m, Yte_o])

    print("MARRO train", len(Xtr_m), Counter(Ytr_m))
    print("OpenNyAI train", len(Xtr_o), Counter(Ytr_o))
    print("combined train", len(Xtr), Counter(Ytr))
    print("combined test", len(Xte), Counter(Yte))

    vec, Mtr = build(Xtr, Ptr)
    clf = LogisticRegression(C=5, max_iter=3000, class_weight="balanced").fit(Mtr, Ytr)
    from sklearn.metrics import classification_report
    pr = clf.predict(feats(vec, Xte, Pte))
    print("combined test accuracy", round(float(np.mean(pr == Yte)), 3))
    print(classification_report(Yte, pr, digits=3))
    g = gold_rows(); gy = np.array([r[2] for r in g])
    gp = clf.predict(feats(vec, [r[0] for r in g], np.array([r[1] for r in g])))
    print("YOUR gold (Claude+A agree, n=%d) accuracy" % len(g), round(float(np.mean(gp == gy)), 3))
    print(classification_report(gy, gp, digits=3, zero_division=0))
    joblib.dump((vec, clf), "models/section_classifier.joblib")

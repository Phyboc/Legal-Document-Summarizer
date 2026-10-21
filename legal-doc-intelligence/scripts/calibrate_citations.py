#!/usr/bin/env python3
"""Calibrate Cite & Verify on labelled supported / unsupported sentences.

There is no human-labelled hallucination set for IN-ABS, so one is constructed
from the section-aligned build (``artifacts/training_data/full/test.jsonl``):

* **supported** - a reference-summary sentence checked against its *own*
  judgement's chunks. These are what the headnote writer derived from the
  judgement, so the judgement supports them (alignment noise aside).
* **unsupported** - the hallucinations a summarizer actually makes, checked
  against the same judgement:
    - ``cross_doc``   a sentence from a *different* judgement
    - ``outcome_flip`` the disposition reversed (allowed <-> dismissed, ...)
    - ``number_swap`` a section / year / amount changed
    - ``negation``    a claim negated ("was held" -> "was not held")

Every sentence goes through ``cite_verify.verify_sentences`` - the exact code
the pipeline runs - and both the NLI entailment score and the old MS-MARCO
relevance score are evaluated, so the report shows how much the change buys.
The chosen threshold maximises F1 for detecting *supported* sentences.

Usage:
    python scripts/calibrate_citations.py --docs 80 --output artifacts/citation_calibration.json
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.postprocess.cite_verify import verify_sentences  # noqa: E402

OUTCOME_FLIPS = [
    ("allowed", "dismissed"), ("dismissed", "allowed"),
    ("upheld", "set aside"), ("set aside", "upheld"),
    ("convicted", "acquitted"), ("acquitted", "convicted"),
    ("granted", "refused"), ("refused", "granted"),
    ("rejected", "accepted"), ("accepted", "rejected"),
    ("valid", "invalid"), ("invalid", "valid"),
    ("constitutional", "unconstitutional"), ("unconstitutional", "constitutional"),
    ("liable", "not liable"), ("guilty", "not guilty"),
]
NEGATABLE = re.compile(r"\b(is|was|were|are|had|has|have|could|would|should|did)\b(?! not)")


def outcome_flip(sentence: str):
    for old, new in OUTCOME_FLIPS:
        pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
        if pattern.search(sentence):
            return pattern.sub(new, sentence, count=1)
    return None


def number_swap(sentence: str, rng: random.Random):
    numbers = [m for m in re.finditer(r"\b\d{1,4}\b", sentence)]
    if not numbers:
        return None
    m = rng.choice(numbers)
    value = int(m.group())
    new = value + rng.choice([-7, -3, 2, 5, 11, 40]) if value > 50 else value + rng.choice([1, 2, 3, 7])
    if new <= 0 or new == value:
        new = value + 13
    return sentence[:m.start()] + str(new) + sentence[m.end():]


def negation(sentence: str):
    m = NEGATABLE.search(sentence)
    if not m:
        return None
    return sentence[:m.end()] + " not" + sentence[m.end():]


def build_examples(records_path: Path, n_docs: int, seed: int):
    rng = random.Random(seed)
    chunks_by_doc, sentences_by_doc = defaultdict(list), defaultdict(list)
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            chunks_by_doc[r["doc_id"]].append((r["chunk_index"], r["chunk_text"]))
            sentences_by_doc[r["doc_id"]].extend(s for s in r["matched_sentences"] if len(s.split()) >= 5)

    doc_ids = [d for d in sorted(chunks_by_doc) if len(chunks_by_doc[d]) >= 2 and sentences_by_doc[d]]
    doc_ids = doc_ids[:n_docs]
    examples = []
    for doc in doc_ids:
        sources = [text for _, text in sorted(chunks_by_doc[doc])]
        own = rng.sample(sentences_by_doc[doc], min(4, len(sentences_by_doc[doc])))
        for s in own:
            examples.append({"doc_id": doc, "sources": sources, "sentence": s, "label": 1, "kind": "supported"})
            for kind, fn in (("outcome_flip", outcome_flip), ("number_swap", lambda x: number_swap(x, rng)), ("negation", negation)):
                altered = fn(s)
                if altered and altered != s:
                    examples.append({"doc_id": doc, "sources": sources, "sentence": altered, "label": 0, "kind": kind})
        other = rng.choice([d for d in doc_ids if d != doc])
        for s in rng.sample(sentences_by_doc[other], min(2, len(sentences_by_doc[other]))):
            examples.append({"doc_id": doc, "sources": sources, "sentence": s, "label": 0, "kind": "cross_doc"})
    return examples


def build_generated_examples(model: str, n_docs: int, seed: int, split: str = "test"):
    """Positives are the fine-tuned model's own summary sentences, checked against
    *all* chunks of the judgement they were generated from - exactly what Cite &
    Verify sees in the app. Negatives perturb those same sentences, or borrow a
    sentence generated for a different judgement."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_dataset import load_split, resolve_columns
    from src.data.chunker import chunk_document
    from src.data.cleaning import preprocess, split_paragraphs
    from src.data.pii_redaction import redact
    from src.data.section_detector import detect_sections, fallback_to_single_section, group_by_section
    from src.model.summarizer import get_summarizer
    from src.postprocess.cite_verify import split_sentences

    rng = random.Random(seed)
    ds = load_split(split, limit=n_docs)
    text_col, _, _ = resolve_columns(ds)
    summarizer = get_summarizer(model)
    docs = []
    for i, row in enumerate(ds):
        clean = preprocess(row[text_col] or "")
        redacted, _ = redact(clean)
        paras = detect_sections(split_paragraphs(redacted))
        groups = {"reasoning": paras} if fallback_to_single_section(paras) else group_by_section(paras)
        chunks = chunk_document(groups)
        if len(chunks) < 2:
            continue
        generated = summarizer.summarize_chunks(chunks)
        sentences = [s for g in generated for s in split_sentences(g["summary"]) if len(s.split()) >= 5]
        docs.append((f"{split}_{i:05d}", [c.text for c in chunks], sentences))
        print(f"  generated {docs[-1][0]}: {len(chunks)} chunks, {len(sentences)} sentences", flush=True)

    examples = []
    for n, (doc, sources, sentences) in enumerate(docs):
        for s in rng.sample(sentences, min(6, len(sentences))):
            examples.append({"doc_id": doc, "sources": sources, "sentence": s, "label": 1, "kind": "supported"})
            for kind, fn in (("outcome_flip", outcome_flip), ("number_swap", lambda x: number_swap(x, rng)), ("negation", negation)):
                altered = fn(s)
                if altered and altered != s:
                    examples.append({"doc_id": doc, "sources": sources, "sentence": altered, "label": 0,
                                     "kind": kind, "original": s})
        others = [d for d in docs if d[0] != doc]
        for s in rng.sample(rng.choice(others)[2], 2):
            examples.append({"doc_id": doc, "sources": sources, "sentence": s, "label": 0, "kind": "cross_doc"})
    return examples


def paired_catch_rates(examples, scores, threshold):
    """Of perturbations whose *original* sentence was verified, the share now rejected."""
    verified_original = {(e["doc_id"], e["sentence"]) for e, s in zip(examples, scores)
                         if e["label"] == 1 and s >= threshold}
    caught = defaultdict(lambda: [0, 0])
    for e, s in zip(examples, scores):
        if e.get("original") and (e["doc_id"], e["original"]) in verified_original:
            caught[e["kind"]][0] += s < threshold
            caught[e["kind"]][1] += 1
    return {k: {"caught": round(c / n, 3), "n": n} for k, (c, n) in caught.items()}


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def sweep(scores, labels, examples):
    rows = []
    for t in [round(x * 0.05, 2) for x in range(1, 20)]:
        pred = [s >= t for s in scores]
        tp = sum(p and l for p, l in zip(pred, labels))
        fp = sum(p and not l for p, l in zip(pred, labels))
        fn = sum((not p) and l for p, l in zip(pred, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        caught = defaultdict(lambda: [0, 0])
        for p, e in zip(pred, examples):
            if e["label"] == 0:
                caught[e["kind"]][0] += (not p)
                caught[e["kind"]][1] += 1
        rows.append({
            "threshold": t, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "verified_rate_supported": round(recall, 3),
            "verified_rate_unsupported": round(fp / max(1, sum(1 for l in labels if l == 0)), 3),
            "caught_by_kind": {k: round(c / n, 3) for k, (c, n) in caught.items()},
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", type=Path, default=Path("artifacts/training_data/full/test.jsonl"))
    parser.add_argument("--docs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=Path("artifacts/citation_calibration.json"))
    parser.add_argument("--positives", choices=("generated", "reference"), default="generated",
                        help="generated: the model's own summary sentences (what the app verifies); "
                             "reference: IN-ABS headnote sentences (harder, noisier labels)")
    parser.add_argument("--model", default="models/legallens-bart-base-e3",
                        help="Summarizer used to generate positives with --positives generated")
    args = parser.parse_args()

    if args.positives == "generated":
        examples = build_generated_examples(args.model, args.docs, args.seed)
    else:
        examples = build_examples(args.records, args.docs, args.seed)
    by_doc = defaultdict(list)
    for i, e in enumerate(examples):
        by_doc[e["doc_id"]].append(i)

    for idx in by_doc.values():
        sources = examples[idx[0]]["sources"]
        checks = verify_sentences([examples[i]["sentence"] for i in idx], sources)
        for i, c in zip(idx, checks):
            examples[i].update(entailment=c["entailment"], contradiction=c["contradiction"],
                               relevance=c["retrieval_score"], method=c["method"])

    labels = [e["label"] for e in examples]
    report = {
        "positives": args.positives,
        "model": args.model if args.positives == "generated" else None,
        "records": str(args.records),
        "documents": len(by_doc),
        "examples": len(examples),
        "by_kind": {k: sum(e["kind"] == k for e in examples) for k in sorted({e["kind"] for e in examples})},
        "method": examples[0]["method"],
        "auc": {
            "nli_entailment": auc([e["entailment"] for e in examples], labels),
            "msmarco_relevance_old": auc([e["relevance"] for e in examples], labels),
        },
        "sweep_nli": sweep([e["entailment"] for e in examples], labels, examples),
        "sweep_msmarco_old": sweep([e["relevance"] for e in examples], labels, examples),
    }
    best = max(report["sweep_nli"], key=lambda r: r["f1"])
    report["recommended_threshold"] = best["threshold"]
    report["at_recommended"] = best
    nli_scores = [e["entailment"] for e in examples]
    report["paired_catch_rates_at_recommended"] = paired_catch_rates(examples, nli_scores, best["threshold"])
    contra = [e["contradiction"] for e in examples if e["kind"] == "outcome_flip"]
    report["outcome_flip_contradiction_median"] = round(sorted(contra)[len(contra) // 2], 3) if contra else None
    report["examples_sample"] = [
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k != "sources"}
        for e in examples[:40]
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("positives", "documents", "examples", "by_kind", "method", "auc",
                                             "recommended_threshold", "at_recommended",
                                             "paired_catch_rates_at_recommended")}, indent=2))
    old_best = max(report["sweep_msmarco_old"], key=lambda r: r["f1"])
    print("old MS-MARCO best:", {k: old_best[k] for k in ("threshold", "precision", "recall", "f1", "caught_by_kind")})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

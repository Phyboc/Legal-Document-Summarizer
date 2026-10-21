#!/usr/bin/env python3
"""Build (and score) a human-checked evidence evaluation set.

Why this exists
---------------
The alignment decides every training target, and its thresholds (0.40 / 0.90) were
chosen by a sweep, not by anyone reading the pairs. A sweep optimises agreement with
itself. This script turns the question into something measurable:

1. ``--sample`` draws a stratified set of (summary sentence, candidate chunk/span)
   pairs from an existing build. Stratification is deliberate: the interesting cases
   are not the average ones. It over-samples low-CE-score pairs, the ``judgement``
   section, and sentences carrying dates, amounts, statutory references and case
   citations, because those are where a wrong match is provable rather than arguable.

   Each row is written unlabelled, with ``label`` and ``notes`` empty, as JSONL plus
   a CSV for spreadsheet editing:

       {"id": "...", "doc_id": "...", "section": "judgement",
        "sentence": "...", "span": "...", "chunk_excerpt": "...", "ce_score": 0.71,
        "features": {"has_year": true, "has_amount": false, ...}, "reason": "low_score",
        "label": "", "notes": ""}

2. A human fills in ``label`` with one of ``supported`` / ``ambiguous`` /
   ``unsupported``. That is the ground truth the thresholds should be calibrated
   against, and it is the only artefact here that cannot be generated.

3. ``--score`` reads the labelled file back and reports, per CE score band and per
   threshold: precision/recall/F1 of ``ce_score >= threshold`` against the
   ``supported`` label, plus the false-support rate. It also names the threshold with
   the best F1 and the rate of "hard-evidence mismatch" pairs the current settings
   would still attach.

Nothing here loads a model or the network.

Usage
-----
    python scripts/sample_evidence_eval.py --dir artifacts/training_data/full \\
        --sample 150 --out artifacts/eval/evidence_eval

    # after labelling:
    python scripts/sample_evidence_eval.py --score artifacts/eval/evidence_eval.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "validation", "test")

# Legal-evidence cues. A pair that contradicts one of these is wrong in a way that
# can be checked by reading, so those pairs are what the sample over-represents.
_CUES = {
    "has_year": re.compile(r"\b(?:1[6-9]|20)\d{2}\b"),
    "has_amount": re.compile(r"(?:Rs\.?|rupees|inr)\s*[\d,.]+|\b\d[\d,]*\s*(?:lakh|crore|lakhs|crores)\b", re.I),
    "has_statute": re.compile(r"\b(?:section|sec\.|s\.|article|art\.|order|rule|clause|cl\.)\s*\d+", re.I),
    "has_case_citation": re.compile(r"\b[A-Z][A-Za-z.&'\- ]{2,40}\s+(?:vs?\.?|versus)\s+[A-Z]"),
    "has_report_citation": re.compile(r"\b(?:A\.?I\.?R\.?|I\.?L\.?R\.?|S\.?C\.?C\.?|Cr\.?L\.?J\.?)\b"),
    "multi_event": re.compile(r"\b(?:and|;)\b.*\b(?:also|further|in addition|another)\b", re.I),
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _features(sentence: str) -> dict:
    return {name: bool(pattern.search(sentence)) for name, pattern in _CUES.items()}


def _bands(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.4:
        return "mid"
    return "low"


def candidates_from_build(build_dir: Path) -> list[dict]:
    """Every attached (sentence, evidence span) pair in a build, plus low-score cases.

    Attached pairs come from the records. Disagreement cases come from
    ``unassigned.jsonl``: sentences the aligner refused but which carry hard evidence
    cues, i.e. candidates for a missed match.
    """
    rows: list[dict] = []
    for split in SPLITS:
        for record in _read_jsonl(build_dir / f"{split}.jsonl"):
            spans = record.get("matched_evidence_spans") or []
            scores = record.get("alignment_scores") or []
            for index, sentence in enumerate(record.get("matched_sentences") or []):
                span_entry = spans[index] if index < len(spans) else {}
                rows.append({
                    "doc_id": record["doc_id"],
                    "split": split,
                    "section": record["section"],
                    "chunk_id": record["chunk_id"],
                    "sentence": sentence,
                    "span": span_entry.get("span"),
                    "span_location": span_entry.get("location"),
                    "chunk_excerpt": (record.get("chunk_text") or "")[:1200],
                    "ce_score": float(scores[index]) if index < len(scores) else None,
                    "reason": "attached",
                })
    for entry in _read_jsonl(build_dir / "unassigned.jsonl"):
        rows.append({
            "doc_id": entry.get("doc_id"),
            "split": None,
            "section": entry.get("best_label"),
            "chunk_id": None,
            "sentence": entry.get("sentence"),
            "span": None,
            "span_location": None,
            "chunk_excerpt": None,
            "ce_score": float(entry.get("best_score") or 0.0),
            "reason": "unassigned_" + str(entry.get("reason")),
        })
    return rows


def stratified_sample(rows: list[dict], size: int, seed: int = 17) -> list[dict]:
    """Deterministic stratified draw: section x score band x evidence cue."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("ce_score") is None:
            continue
        features = _features(row["sentence"] or "")
        # The cue bucket makes the sample explicitly cover the checkable cases.
        cue = next((name for name in ("has_amount", "has_year", "has_report_citation",
                                      "has_case_citation", "has_statute", "multi_event")
                    if features[name]), "none")
        strata[(row["section"], _bands(row["ce_score"]), cue)].append(row)

    for key, bucket in strata.items():
        bucket.sort(key=lambda r: (r["doc_id"] or "", r["ce_score"], r["sentence"] or ""))

    # Round-robin over strata so every cell is represented even at small sizes,
    # rarest/most-informative cells first.
    ordered_keys = sorted(strata, key=lambda key: (-len(strata[key]), key))
    picked: list[dict] = []
    while len(picked) < size:
        progressed = False
        for key in ordered_keys:
            bucket = strata[key]
            if not bucket:
                continue
            index = rng.randrange(len(bucket))
            picked.append(bucket.pop(index))
            progressed = True
            if len(picked) >= size:
                break
        if not progressed:
            break

    for position, row in enumerate(picked, 1):
        key = f"{row['doc_id']}|{row['chunk_id']}|{row['sentence']}"
        row["id"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        row["features"] = _features(row["sentence"] or "")
        row["label"] = ""
        row["notes"] = ""
        row["position"] = position
    return picked


def write_sample(rows: list[dict], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out_prefix.with_suffix(".jsonl")
    jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "id", "label", "notes", "section", "reason", "ce_score",
            "sentence", "span", "chunk_excerpt",
        ])
        for row in rows:
            writer.writerow([
                row["id"], "", "", row["section"], row["reason"], row["ce_score"],
                row["sentence"], row["span"], row["chunk_excerpt"],
            ])
    print(f"Wrote {len(rows)} pairs to {jsonl} and {csv_path}")
    print("Fill in `label` with supported / ambiguous / unsupported, then run --score.")


def score(labelled_path: Path, thresholds: list[float]) -> int:
    rows = _read_jsonl(labelled_path)
    labelled = [r for r in rows if (r.get("label") or "").strip()]
    if not labelled:
        raise SystemExit(f"No labelled rows in {labelled_path}; fill in the `label` column first.")

    counts = Counter(r["label"].strip().lower() for r in labelled)
    supported = sum(1 for r in labelled if r["label"].strip().lower() == "supported")
    print(f"Labelled pairs: {len(labelled)}  {dict(counts)}")
    print(f"Supported share (the base rate a no-op 'attach everything' rule would get): "
          f"{supported / len(labelled):.1%}\n")

    print("| Threshold | Precision | Recall | F1 | False supports | Matched |")
    print("| --- | --- | --- | --- | --- | --- |")
    best = (None, -1.0)
    for threshold in thresholds:
        matched = [r for r in labelled if (r.get("ce_score") or 0.0) >= threshold]
        true_positive = sum(1 for r in matched if r["label"].strip().lower() == "supported")
        precision = true_positive / len(matched) if matched else 0.0
        recall = true_positive / supported if supported else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        false_supports = sum(1 for r in matched if r["label"].strip().lower() == "unsupported")
        print(f"| {threshold:.2f} | {precision:.2%} | {recall:.2%} | {f1:.3f} | "
              f"{false_supports} | {len(matched)} |")
        if f1 > best[1]:
            best = (threshold, f1)

    print(f"\nBest F1 at threshold {best[0]:.2f} (F1 {best[1]:.3f}).")
    current = 0.40
    matched = [r for r in labelled if (r.get("ce_score") or 0.0) >= current]
    false_supports = [r for r in matched if r["label"].strip().lower() == "unsupported"]
    hard = [
        r for r in false_supports
        if r.get("features", {}).get("has_amount")
        or r.get("features", {}).get("has_year")
        or r.get("features", {}).get("has_report_citation")
    ]
    print(f"At the current threshold {current:.2f}: {len(false_supports)} false supports, "
          f"{len(hard)} of them carrying a checkable date/amount/citation - those are the "
          "pairs that poison a training target.")

    by_section: dict[str, Counter] = defaultdict(Counter)
    for row in labelled:
        by_section[row.get("section") or "unknown"][row["label"].strip().lower()] += 1
    print("\nPer section:")
    for section, counter in sorted(by_section.items()):
        print(f"  {section:24s} {dict(counter)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=None, help="Build directory to sample from.")
    parser.add_argument("--sample", type=int, default=150, help="Number of pairs to draw.")
    parser.add_argument("--out", default="artifacts/eval/evidence_eval",
                        help="Output prefix (.jsonl and .csv are appended).")
    parser.add_argument("--seed", type=int, default=17, help="Sampling seed (deterministic).")
    parser.add_argument("--score", default=None, help="Score a labelled JSONL instead of sampling.")
    args = parser.parse_args(argv)

    if args.score:
        return score(Path(args.score),
                     [0.30, 0.40, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95])
    if not args.dir:
        parser.error("--dir is required unless --score is given.")

    rows = candidates_from_build(Path(args.dir))
    print(f"{len(rows)} candidate pairs in {args.dir}")
    if not rows:
        raise SystemExit("No pairs found - is this a build directory with matched_sentences?")
    sample = stratified_sample(rows, args.sample, seed=args.seed)
    write_sample(sample, Path(args.out))
    strata = Counter((row["section"], _bands(row["ce_score"]), row["reason"]) for row in sample)
    print("\nStratification:")
    for key, count in sorted(strata.items(), key=lambda item: -item[1])[:20]:
        print(f"  {count:4d}  section={key[0]} band={key[1]} reason={key[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

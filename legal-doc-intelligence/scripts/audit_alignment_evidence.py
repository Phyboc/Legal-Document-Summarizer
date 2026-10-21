#!/usr/bin/env python3
"""Offline calibration audit for the evidence-aware alignment layer.

What this is for
----------------
``src/data/evidence_alignment.py`` has two knobs that must NOT be chosen by
intuition: the Tier-3 coverage bound ``C`` and the ambiguity margin ``M`` (plus
the ``high_confidence`` vs ``off`` contradiction policy). This script is the
procedure that chooses them, using data rather than guesswork:

1. **Weak-supervision corpus.** From an already-built dataset JSONL it labels
   (sentence, chunk) pairs with two high-precision, model-independent signals:
     * a pair whose Tier-1 evidence is *contradicted* (the sentence carries a
       date/year/amount and the chunk positively talks about a different one) is
       a **negative**;
     * a pair sharing a long verbatim content-word span, or fully satisfying
       several Tier-1 classes, is a **positive**.
   These are proxies, not gold. The script reports their sizes and refuses to
   recommend anything below a minimum, so a sweep cannot silently tune on noise.

2. **Sweep.** Given a JSONL of precomputed candidate sets
   (``{"doc_id", "sentence", "candidates": [{index, label, text, ce_score,
   position}]}``) it runs the real decision function over a grid of
   ``(C, M, contradiction_mode)`` and reports, per grid point:
   supported / ambiguous / unsupported / not-summary-content rates,
   ``hard_evidence_mismatch_rate`` (share of attached pairs the corpus labelled
   negative), ``records_retained`` and ``unassigned_rate``.

Everything here runs **offline**: it never loads the Cross-Encoder, the BART
tokenizer, spaCy or the network. Scores and candidate text come from the input
file. That is deliberate - the layer must be auditable without a model download.

Usage
-----
    # headline mismatch rate + corpus sizes on an existing build
    python scripts/audit_alignment_evidence.py \\
        --from-jsonl ../../../training\\ data/train.jsonl \\
        --report artifacts/evidence_audit.json

    # full sweep (needs a precomputed candidate matrix, see --dump-candidates)
    python scripts/audit_alignment_evidence.py \\
        --from-candidates artifacts/evidence_candidates.jsonl \\
        --out artifacts/evidence_sweep.json --report artifacts/evidence_audit.json

Exit codes: 0 = report produced; 2 = a recommendation was requested but the
corpus was below the minimum size (the report is still written).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.data import evidence_alignment as ev  # noqa: E402
from src.data.section_summary_alignment import evidence_settings_from_config  # noqa: E402

#: A recommendation is only produced when the corpus is at least this large.
MIN_POSITIVES = 200
MIN_NEGATIVES = 200

#: Proxy-label thresholds (definitional, not tuned): a pair is a positive when it
#: shares at least this many contiguous content words, or fully satisfies this
#: many Tier-1 evidence classes.
POSITIVE_SPAN = 6
POSITIVE_CLASSES = 2

GRID_COVERAGE = (0.30, 0.40, 0.50, 0.60, 0.70)
GRID_MARGIN = (0.0, 0.05, 0.10, 0.20)
GRID_CONTRADICTION = (ev.CONTRADICTION_MODE_HIGH_CONFIDENCE, ev.CONTRADICTION_MODE_OFF)

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_UNCLEAR = "unclear"


# --------------------------------------------------------------------------- #
# Weak supervision
# --------------------------------------------------------------------------- #

@dataclass
class LabelledPair:
    """One (sentence, chunk) pair with its proxy label and evidence features."""

    doc_id: str
    sentence: str
    chunk_index: int
    section: str
    label: str
    contradicted: bool
    contradiction_classes: list[str] = field(default_factory=list)
    supported_classes: int = 0
    longest_span: int = 0
    coverage: float = 0.0
    rare_shared: int = 0
    has_checkable_tier1: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def has_checkable_tier1(sentence_evidence: ev.Evidence) -> bool:
    """True when the sentence carries at least one Tier-1 token worth checking.

    Fractions are always reported against this denominator, so a sentence with no
    dates or amounts cannot dilute the mismatch rate.
    """
    return any(sentence_evidence.tier1(name) for name in ev.CONTRADICTION_CLASSES)


def label_pair(
    pair: ev.CandidateEvidence,
    *,
    positive_span: int = POSITIVE_SPAN,
    positive_classes: int = POSITIVE_CLASSES,
) -> str:
    """Proxy label. Negative is high precision; positive is a strong-signal proxy."""
    if pair.contradicted:
        return LABEL_NEGATIVE
    if pair.supported_classes >= positive_classes:
        return LABEL_POSITIVE
    if pair.longest_span >= positive_span:
        return LABEL_POSITIVE
    return LABEL_UNCLEAR


def build_corpus_from_jsonl(
    records,
    *,
    settings: ev.EvidenceSettings | None = None,
    positive_span: int = POSITIVE_SPAN,
    positive_classes: int = POSITIVE_CLASSES,
) -> dict:
    """Build the weak-supervision corpus from an existing built dataset.

    Only the chunk each sentence was actually attached to is scored, because that
    is what the built dataset records. Document-local rarity is computed per
    ``doc_id`` from that document's chunk texts.
    """
    settings = settings or ev.EvidenceSettings()

    by_doc: dict[str, list[dict]] = {}
    for record in records:
        by_doc.setdefault(str(record.get("doc_id")), []).append(record)

    pairs: list[LabelledPair] = []
    checkable = 0
    for doc_id, doc_records in by_doc.items():
        df = ev.document_frequency([r.get("chunk_text") or "" for r in doc_records])
        for record in doc_records:
            chunk_text = record.get("chunk_text") or ""
            scores = record.get("alignment_scores") or []
            for position, sentence in enumerate(record.get("matched_sentences") or []):
                sentence_evidence = ev.extract_evidence(sentence)
                if ev.is_not_summary_content(sentence_evidence):
                    continue
                # NOT_SUMMARY_CONTENT fragments are counted separately below; they
                # are not alignment evidence of anything.
                ce_score = float(scores[position]) if position < len(scores) else 0.0
                pair = ev.evaluate_candidate(
                    sentence_evidence,
                    ev.Candidate(
                        index=int(record.get("chunk_index", 0)),
                        label=str(record.get("section") or record.get("section_label") or ""),
                        text=chunk_text,
                        ce_score=ce_score,
                    ),
                    settings=settings,
                    document_frequency=df,
                )
                checkable_flag = has_checkable_tier1(sentence_evidence)
                checkable += int(checkable_flag)
                pairs.append(LabelledPair(
                    doc_id=doc_id,
                    sentence=sentence,
                    chunk_index=pair.index,
                    section=pair.label,
                    label=label_pair(pair, positive_span=positive_span, positive_classes=positive_classes),
                    contradicted=pair.contradicted,
                    contradiction_classes=pair.contradiction_classes,
                    supported_classes=pair.supported_classes,
                    longest_span=pair.longest_span,
                    coverage=round(pair.coverage, 4),
                    rare_shared=pair.rare_shared,
                    has_checkable_tier1=checkable_flag,
                ))

    counts = Counter(pair.label for pair in pairs)
    contradicted = sum(1 for pair in pairs if pair.contradicted)
    # Headline metric: of the pairs the *current* build attached that we could
    # check, how many are contradicted by their own Tier-1 evidence?
    return {
        "source": "already_built_dataset",
        "pairs": len(pairs),
        "positives": counts.get(LABEL_POSITIVE, 0),
        "negatives": counts.get(LABEL_NEGATIVE, 0),
        "unclear": counts.get(LABEL_UNCLEAR, 0),
        "contradicted_pairs": contradicted,
        "pairs_with_checkable_tier1": checkable,
        "hard_evidence_mismatch_rate": round(contradicted / checkable, 4) if checkable else 0.0,
        "coverage_distribution": _percentiles([pair.coverage for pair in pairs]),
        "span_distribution": _percentiles([float(pair.longest_span) for pair in pairs]),
        "examples": {
            "contradicted": [p.as_dict() for p in pairs if p.contradicted][:5],
            "positive": [p.as_dict() for p in pairs if p.label == LABEL_POSITIVE][:5],
        },
    }


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    ordered = sorted(values)

    def pick(pct: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
        return round(float(ordered[index]), 4)

    return {f"p{p}": pick(p) for p in (10, 25, 50, 75, 90)}


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def build_case(doc_id: str, sentence: str, candidates: list[dict]) -> dict:
    """Normalise one precomputed candidate record."""
    return {
        "doc_id": doc_id,
        "sentence": sentence,
        "candidates": [
            ev.Candidate(
                index=int(c["index"]),
                label=str(c.get("label", "")),
                text=str(c.get("text", "")),
                ce_score=float(c.get("ce_score", 0.0)),
                position=(None if c.get("position") is None else float(c["position"])),
            )
            for c in candidates
        ],
    }


def sweep_case(case: dict, settings: ev.EvidenceSettings) -> ev.EvidenceDecision:
    """Run the real decision function on one precomputed case."""
    document_frequency = ev.document_frequency([c.text for c in case["candidates"]])
    return ev.decide(
        case["sentence"],
        case["candidates"],
        settings=settings,
        document_frequency=document_frequency,
    )


def sweep(cases: list[dict], *, grid_coverage=GRID_COVERAGE, grid_margin=GRID_MARGIN,
          grid_contradiction=GRID_CONTRADICTION, settings_base: ev.EvidenceSettings | None = None) -> list[dict]:
    """Grid search over ``(C, M, contradiction_mode)``.

    For each point: decision rates, ``hard_evidence_mismatch_rate`` (attached
    pairs whose candidate is contradicted by Tier-1 evidence),
    ``records_retained`` (distinct chunks that would receive a target - an upper
    bound, since the target-budget layer can still drop a record) and the
    unassigned rate.
    """
    base = settings_base or ev.EvidenceSettings()
    points: list[dict] = []

    prepared = []
    for case in cases:
        prepared.append((case, ev.extract_evidence(case["sentence"]),
                         ev.document_frequency([c.text for c in case["candidates"]])))

    for coverage in grid_coverage:
        for margin in grid_margin:
            for contradiction in grid_contradiction:
                settings = ev.EvidenceSettings(
                    coverage_bound=coverage,
                    ambiguity_margin=margin,
                    min_common_span=base.min_common_span,
                    max_rare_df=base.max_rare_df,
                    contradiction_mode=contradiction,
                )
                counts: Counter = Counter()
                retained: set = set()
                attached_pairs = 0
                attached_mismatch = 0
                attached_positive = 0

                for case, sentence_evidence, df in prepared:
                    decision = ev.decide(
                        case["sentence"],
                        case["candidates"],
                        settings=settings,
                        document_frequency=df,
                        sentence_evidence=sentence_evidence,
                    )
                    counts[decision.decision] += 1
                    if not decision.attached:
                        continue
                    retained.add((case["doc_id"], decision.primary_index))
                    attached_pairs += 1
                    winner = next(
                        (c for c in decision.candidates if c.index == decision.primary_index), None
                    )
                    if winner is not None:
                        attached_mismatch += int(winner.contradicted)
                        if label_pair(winner) == LABEL_POSITIVE:
                            attached_positive += 1

                total = len(cases) or 1
                points.append({
                    "coverage_bound": coverage,
                    "ambiguity_margin": margin,
                    "contradiction_mode": contradiction,
                    "cases": len(cases),
                    "supported": counts.get(ev.DECISION_SUPPORTED, 0),
                    "ambiguous": counts.get(ev.DECISION_AMBIGUOUS, 0),
                    "unsupported": counts.get(ev.DECISION_UNSUPPORTED, 0),
                    "not_summary_content": counts.get(ev.DECISION_NOT_SUMMARY_CONTENT, 0),
                    "supported_rate": round(counts.get(ev.DECISION_SUPPORTED, 0) / total, 4),
                    "ambiguous_rate": round(counts.get(ev.DECISION_AMBIGUOUS, 0) / total, 4),
                    "unsupported_rate": round(counts.get(ev.DECISION_UNSUPPORTED, 0) / total, 4),
                    "not_summary_content_rate": round(
                        counts.get(ev.DECISION_NOT_SUMMARY_CONTENT, 0) / total, 4
                    ),
                    "unassigned_rate": round(
                        (counts.get(ev.DECISION_UNSUPPORTED, 0)
                         + counts.get(ev.DECISION_NOT_SUMMARY_CONTENT, 0)) / total, 4
                    ),
                    "records_retained": len(retained),
                    "attached_pairs": attached_pairs,
                    "hard_evidence_mismatch_rate": (
                        round(attached_mismatch / attached_pairs, 4) if attached_pairs else 0.0
                    ),
                    "positive_recovery_rate": (
                        round(attached_positive / attached_pairs, 4) if attached_pairs else 0.0
                    ),
                })
    return points


def recommend(points: list[dict], corpus: dict, *,
              min_positives: int = MIN_POSITIVES, min_negatives: int = MIN_NEGATIVES) -> tuple[dict | None, dict]:
    """Pick an operating point, or refuse when the corpus is too small.

    Two-step objective, stated plainly: **minimise
    ``hard_evidence_mismatch_rate``**, then among the points that achieve it take
    the one with the **highest supported_rate** (least collapse). No composite
    score is invented, and nothing is recommended from an empty/small corpus.
    """
    status = {
        "positives": corpus.get("positives", 0),
        "negatives": corpus.get("negatives", 0),
        "min_positives": min_positives,
        "min_negatives": min_negatives,
    }
    if not points:
        status["reason"] = "no_sweep_points"
        return None, status
    if corpus.get("positives", 0) < min_positives or corpus.get("negatives", 0) < min_negatives:
        status["reason"] = "insufficient_corpus"
        status["message"] = (
            "Refusing to recommend thresholds: the weak-supervision corpus is below the "
            "minimum size, so any operating point would be tuned on noise. Grow the "
            "corpus (more documents), or lower --min-positives/--min-negatives "
            "deliberately."
        )
        return None, status

    best_mismatch = min(point["hard_evidence_mismatch_rate"] for point in points)
    finalists = [p for p in points if p["hard_evidence_mismatch_rate"] == best_mismatch]
    chosen = max(finalists, key=lambda p: (p["supported_rate"], -p["ambiguous_rate"]))
    status["reason"] = "ok"
    status["objective"] = "minimise hard_evidence_mismatch_rate, then maximise supported_rate"
    return chosen, status


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline calibration audit for the evidence-aware alignment layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-jsonl", type=Path,
                        help="Existing built dataset JSONL (corpus + headline mismatch rate).")
    source.add_argument("--from-candidates", type=Path,
                        help="Precomputed candidate sets JSONL (runs the full sweep).")
    parser.add_argument("--out", type=Path, default=Path("artifacts/evidence_sweep.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/evidence_audit.json"))
    parser.add_argument("--min-positives", type=int, default=MIN_POSITIVES)
    parser.add_argument("--min-negatives", type=int, default=MIN_NEGATIVES)
    parser.add_argument("--positive-span", type=int, default=POSITIVE_SPAN)
    parser.add_argument("--positive-classes", type=int, default=POSITIVE_CLASSES)
    parser.add_argument("--limit", type=int, default=None, help="Read only the first N lines.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = evidence_settings_from_config()
    report: dict = {"evidence_settings_used_for_scoring": settings.as_dict()}
    exit_code = 0

    if args.from_jsonl:
        if not args.from_jsonl.exists():
            raise SystemExit(f"Not found: {args.from_jsonl}")
        records = _read_jsonl(args.from_jsonl)
        if args.limit:
            records = records[: args.limit]
        corpus = build_corpus_from_jsonl(
            records, settings=settings,
            positive_span=args.positive_span, positive_classes=args.positive_classes,
        )
        report["corpus"] = corpus
        report["sweep"] = None
        recommendation, status = recommend(
            [], corpus, min_positives=args.min_positives, min_negatives=args.min_negatives
        )
        report["recommendation"] = recommendation
        report["corpus_status"] = status
        if status.get("reason") == "no_sweep_points":
            report["note"] = (
                "Corpus-only audit: an existing dataset records the score of the chunk each "
                "sentence was attached to, not the full candidate matrix, so the (C, M) grid "
                "cannot be swept from it. Use --from-candidates with a precomputed candidate "
                "file for the sweep and the threshold recommendation."
            )
        print(json.dumps({
            "pairs": corpus["pairs"],
            "positives": corpus["positives"],
            "negatives": corpus["negatives"],
            "hard_evidence_mismatch_rate": corpus["hard_evidence_mismatch_rate"],
            "pairs_with_checkable_tier1": corpus["pairs_with_checkable_tier1"],
            "coverage": corpus["coverage_distribution"],
        }, indent=2))
    else:
        if not args.from_candidates.exists():
            raise SystemExit(f"Not found: {args.from_candidates}")
        rows = _read_jsonl(args.from_candidates)
        if args.limit:
            rows = rows[: args.limit]
        cases = [build_case(r.get("doc_id", ""), r["sentence"], r.get("candidates") or []) for r in rows]

        # Corpus labels come from the strongest pair in each case, so the
        # recommendation has a positive/negative basis even in sweep mode.
        corpus = Counter()
        for case in cases:
            sentence_evidence = ev.extract_evidence(case["sentence"])
            df = ev.document_frequency([c.text for c in case["candidates"]])
            labels = {
                label_pair(ev.evaluate_candidate(sentence_evidence, c, settings=settings, document_frequency=df))
                for c in case["candidates"]
            }
            if LABEL_NEGATIVE in labels:
                corpus[LABEL_NEGATIVE] += 1
            if LABEL_POSITIVE in labels:
                corpus[LABEL_POSITIVE] += 1
        corpus_summary = {
            "source": "precomputed_candidates",
            "positives": corpus.get(LABEL_POSITIVE, 0),
            "negatives": corpus.get(LABEL_NEGATIVE, 0),
        }

        points = sweep(cases, settings_base=settings)
        report["corpus"] = corpus_summary
        report["sweep"] = points
        recommendation, status = recommend(
            points, corpus_summary,
            min_positives=args.min_positives, min_negatives=args.min_negatives,
        )
        report["recommendation"] = recommendation
        report["corpus_status"] = status

        if recommendation is None:
            exit_code = 2
            print(status.get("message", status.get("reason")), file=sys.stderr)
        else:
            print(json.dumps({"recommended": recommendation}, indent=2))

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {args.report}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""Trace operative-order (outcome) sentences through the dataset build, stage by stage.

For every reference-summary sentence that states a court's final order, this
reports where it ends up in the section-aligned training data:

    segmented -> aligned (assigned to a section) -> primary chunk
              -> retained in that chunk's aligned_target
              -> or lost: unassigned | budget_overflow | primary_chunk_dropped

Outcome sentences are found with ``OUTCOME_CUES``, an audit-side pattern set
that is deliberately independent of ``section_detector.is_disposition`` (the
training-side detector). Each sentence also records whether ``is_disposition``
catches it, so detector misses are visible rather than hidden. Both detectors
are heuristics: every number produced here is provisional until the review
sample (``--review_sample``) has been read by a person.

Usage (run from legal-doc-intelligence/):

    python scripts/audit_outcome_survival.py --sample 100 --split validation \\
        --label baseline --output_dir reports/outcome_audit/baseline

Writes ``outcome_audit.json`` (summary), ``outcome_sentences.csv`` (one row per
outcome sentence) and ``outcome_audit.md`` into ``--output_dir``. Nothing is
written into a dataset directory and no training artifact is touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import (  # noqa: E402
    OFFICIAL_SPLITS,
    _tokenizer_measure,
    build_document,
    judgement_outcome,
    load_split,
    resolve_columns,
    require_bart_tokenizer,
    require_real_encoder,
)
from config import MAX_TARGET_LENGTH  # noqa: E402
from src.data.section_detector import is_disposition  # noqa: E402
from src.data.section_summary_alignment import summary_sentences  # noqa: E402

# Audit-side outcome cues. Independent of section_detector.is_disposition so the
# audit cannot grade the detector with the detector's own rules. Covers appeal,
# petition, criminal, civil-decree and generic operative wording, plus remand /
# modification / costs. Lower-court context is flagged, not excluded: "the
# appeal was dismissed by the High Court" is recorded with lower_court=True so a
# reviewer can see the ambiguity.
_OUTCOME_VERBS = (
    r"(?:partly\s+|partially\s+|hereby\s+)?"
    r"(?:allowed|dismissed|disposed(?:\s+of)?|decreed|set\s+aside|quashed|"
    r"remanded|remitted|modified|upheld|restored|affirmed|reversed|"
    r"acquitted|convicted|confirmed|maintained|varied|granted|rejected)"
)
OUTCOME_CUES = [
    re.compile(r"\b(?:appeal|petition|revision|application|suit|claim|reference|matter)s?\b[^.]{0,60}?\b(?:is|are|stands?|be|stand|was|were)\s+" + _OUTCOME_VERBS + r"\b", re.I),
    re.compile(r"\b(?:order|judgment|judgement|decree|conviction|sentence|verdict|acquittal|impugned\s+\w+)s?\b[^.]{0,40}?\b(?:is|are|stands?|be|was|were)\s+" + _OUTCOME_VERBS + r"\b", re.I),
    re.compile(r"\b(?:we|court|this\s+court|i)\s+(?:hereby\s+)?(?:allow|dismiss|set\s+aside|quash|remand|remit|acquit|convict|modify|decree|partly\s+allow)\b", re.I),
    re.compile(r"\b(?:is|are|be|stands?)\s+(?:hereby\s+)?(?:acquitted|convicted|remanded|discharged)\b", re.I),
    re.compile(r"\bis\s+(?:hereby\s+)?(?:ordered|directed|declared)\b|\b(?:ordered|directed)\s+to\s+(?:pay|refund|return|deposit|compensate)\b", re.I),
    re.compile(r"\bheld\s+(?:the\s+\w+\s+)?liable\b|\bliable\s+to\s+(?:pay|refund|compensate)\b", re.I),
    re.compile(r"\bno\s+order\s+as\s+to\s+costs\b|\bwith\s+costs\b|\bno\s+costs\b", re.I),
    re.compile(r"\bmatter\s+is\s+remanded\b|\bremanded\s+(?:back\s+)?to\b", re.I),
]
_LOWER_COURT = re.compile(
    r"\b(?:high\s+court|trial\s+court|sessions\s+(?:court|judge)|magistrate|lower\s+court|"
    r"tribunal|single\s+judge|division\s+bench|learned\s+(?:judge|magistrate)|district\s+court)\b",
    re.I,
)


def is_outcome_cue(sentence: str) -> bool:
    """True when the sentence states a final-order cue under the audit's independent rules."""
    return any(p.search(sentence) for p in OUTCOME_CUES)


def _find_alignment_record(alignment, sentence: str):
    if alignment is None:
        return None
    for record in alignment.per_sentence:
        if record.sentence == sentence:
            return record
    return None


def trace_outcome_sentence(build, sentence: str) -> dict:
    """Follow one reference outcome sentence through every stage of one document's build."""
    row = {
        "doc_id": build.doc_id,
        "split": build.split,
        "sentence": sentence,
        "lower_court_context": bool(_LOWER_COURT.search(sentence)),
        "training_detector_hit": is_disposition(sentence),
        "stage": None,
        "assigned_labels": "",
        "primary_chunk": None,
        "primary_chunk_section": "",
        "retained_in_target": False,
        "best_label": "",
        "best_score": None,
    }
    record = _find_alignment_record(build.alignment, sentence)
    if record is None:
        row["stage"] = "not_segmented"
        return row

    row["best_label"] = record.best_label or ""
    row["best_score"] = round(record.best_score, 4)
    row["assigned_labels"] = ";".join(record.assigned)
    if not record.assigned:
        row["stage"] = "unassigned"
        return row

    if record.primary_chunk is None:
        row["stage"] = "assigned_no_primary_chunk"
        return row

    chunk_index = record.primary_chunk
    row["primary_chunk"] = chunk_index
    row["primary_chunk_section"] = build.chunk_sections[chunk_index] if chunk_index < len(build.chunk_sections) else ""

    target_records = [r for r in build.records if r.get("chunk_index") == chunk_index]
    if not target_records:
        row["stage"] = "primary_chunk_dropped"
        return row
    target = target_records[0]
    if sentence in target["matched_sentences"]:
        row["stage"] = "retained"
        row["retained_in_target"] = True
    elif sentence in target["overflow_sentences"]:
        row["stage"] = "budget_overflow"
    else:
        row["stage"] = "lost_unknown"
    return row


def audit_builds(builds) -> tuple[dict, list[dict]]:
    """Return ``(summary, per_sentence_rows)`` for a list of DocumentBuild objects."""
    rows: list[dict] = []
    docs_with_outcome = 0
    docs_with_retained_outcome = 0
    docs_with_judgement_section = 0
    doc_outcomes = Counter()
    per_doc_outcome_status: Counter = Counter()
    failed = 0

    for build in builds:
        if build.error:
            failed += 1
            continue
        if build.judgement_section_detected:
            docs_with_judgement_section += 1
        if build.reference is None:
            continue
        sentences = summary_sentences(build.reference["original_reference_summary"])
        outcome_sentences = [s for s in sentences if is_outcome_cue(s)]
        if not outcome_sentences:
            doc_outcomes["no_outcome_in_reference"] += 1
            continue
        docs_with_outcome += 1
        traced = [trace_outcome_sentence(build, s) for s in outcome_sentences]
        rows.extend(traced)
        if any(t["retained_in_target"] for t in traced):
            docs_with_retained_outcome += 1
            per_doc_outcome_status["at_least_one_outcome_retained"] += 1
        else:
            per_doc_outcome_status["no_outcome_retained"] += 1
        for t in traced:
            doc_outcomes[t["stage"]] += 1

    stage_counts = Counter(r["stage"] for r in rows)
    total = len(rows)
    detector_misses = [r for r in rows if not r["training_detector_hit"]]
    lower_court = [r for r in rows if r["lower_court_context"]]
    judgement_states = Counter(judgement_outcome(b) for b in builds if not b.error)

    def pct(n: int, d: int) -> float | None:
        return round(100 * n / d, 2) if d else None

    finals = [r for r in rows if not r["lower_court_context"]]
    final_stage_counts = Counter(r["stage"] for r in finals)
    final_summary = {
        "total": len(finals),
        "stages": dict(final_stage_counts),
        "retained_pct": pct(final_stage_counts.get("retained", 0), len(finals)),
        "assigned_judgement_label": sum(1 for r in finals if "judgement" in r["assigned_labels"].split(";")),
        "in_judgement_chunk": sum(1 for r in finals if r["primary_chunk_section"] == "judgement"),
        "training_detector_misses": sum(1 for r in finals if not r["training_detector_hit"]),
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": len(builds),
        "documents_failed": failed,
        "documents_with_judgement_section_detected": docs_with_judgement_section,
        "judgement_outcome_states": dict(judgement_states),
        "documents_with_reference_outcome": docs_with_outcome,
        "documents_with_reference_outcome_pct": pct(docs_with_outcome, len(builds)),
        "documents_with_any_outcome_retained_in_target": docs_with_retained_outcome,
        "documents_with_any_outcome_retained_pct_of_outcome_docs": pct(docs_with_retained_outcome, docs_with_outcome),
        "outcome_sentences_total": total,
        "outcome_sentence_stages": dict(stage_counts),
        "outcome_sentence_stage_pct": {k: pct(v, total) for k, v in stage_counts.items()},
        "outcome_sentences_retained_pct": pct(stage_counts.get("retained", 0), total),
        "training_detector_misses": len(detector_misses),
        "training_detector_miss_examples": [r["sentence"][:200] for r in detector_misses[:10]],
        "lower_court_context_sentences": len(lower_court),
        "non_lower_court_outcome_sentences": final_summary,
        "max_target_length": MAX_TARGET_LENGTH,
    }
    return summary, rows


_CSV_FIELDS = [
    "doc_id", "split", "stage", "retained_in_target", "primary_chunk",
    "primary_chunk_section", "assigned_labels", "best_label", "best_score",
    "training_detector_hit", "lower_court_context", "sentence",
]


def write_outputs(output_dir: Path, label: str, summary: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"label": label, **summary}
    (output_dir / "outcome_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "outcome_sentences.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    stages = summary["outcome_sentence_stages"]
    total = summary["outcome_sentences_total"]
    lines = [
        f"# Outcome survival audit: {label}",
        "",
        f"- Documents audited: {summary['documents']} ({summary['documents_failed']} failed)",
        f"- Documents with a judgement section detected: {summary['documents_with_judgement_section_detected']}",
        f"- Documents with a reference outcome (audit cues): {summary['documents_with_reference_outcome']} "
        f"({summary['documents_with_reference_outcome_pct']}% of documents)",
        f"- Outcome documents with >=1 outcome retained in a training target: "
        f"{summary['documents_with_any_outcome_retained_in_target']} "
        f"({summary['documents_with_any_outcome_retained_pct_of_outcome_docs']}%)",
        f"- Outcome sentences: {total}",
        "",
        "| Stage | Count | Share |",
        "|---|---:|---:|",
    ]
    for stage, count in sorted(stages.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {stage} | {count} | {summary['outcome_sentence_stage_pct'][stage]}% |")
    lines += [
        "",
        f"- Training-detector (`is_disposition`) misses among outcome sentences: {summary['training_detector_misses']}",
        f"- Outcome sentences with lower-court context (ambiguous, flagged): {summary['lower_court_context_sentences']}",
        f"- Excluding lower-court context ({summary['non_lower_court_outcome_sentences']['total']} sentences): "
        f"retained {summary['non_lower_court_outcome_sentences']['retained_pct']}%, "
        f"labelled judgement {summary['non_lower_court_outcome_sentences']['assigned_judgement_label']}, "
        f"primary chunk in a judgement section {summary['non_lower_court_outcome_sentences']['in_judgement_chunk']}",
        "",
        "Heuristic counts. Stage labels come from exact sentence matching against the",
        "alignment record; outcome sentences come from audit-side cues. Provisional until",
        "a person has reviewed a sample.",
        "",
    ]
    (output_dir / "outcome_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=100, help="First N documents of each split.")
    parser.add_argument("--split", action="append", choices=OFFICIAL_SPLITS, help="Repeatable; default validation.")
    parser.add_argument("--label", default="audit", help="Name written into the report.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_real_encoder(False)
    tokenizer = require_bart_tokenizer()
    measure = _tokenizer_measure(tokenizer)
    splits = args.split or ["validation"]

    builds = []
    for split in splits:
        dataset = load_split(split, limit=args.sample)
        text_col, summary_col, _ = resolve_columns(dataset)
        for index, row in enumerate(dataset):
            builds.append(build_document(
                f"{split}_{index:05d}", split,
                row.get(text_col) or "", row.get(summary_col) or "",
                token_measure=measure, max_target_tokens=MAX_TARGET_LENGTH,
            ))
        print(f"{split}: audited {len(dataset)} document(s)")

    summary, rows = audit_builds(builds)
    write_outputs(Path(args.output_dir), args.label, summary, rows)
    print(json.dumps({k: summary[k] for k in (
        "documents_with_reference_outcome", "outcome_sentences_total",
        "outcome_sentence_stages", "outcome_sentences_retained_pct")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

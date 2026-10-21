#!/usr/bin/env python3
"""Offline quality diagnosis of a built section-aligned dataset.

Why this exists
---------------
A build reports its own statistics (``build_stats.json``), but those describe what
the builder *intended*: counts, configuration, the alignment backend. They do not
tell you whether the records are trainable. This script reads only the written
artifacts and answers the questions a reviewer actually has:

* Is ``aligned_target`` really about its ``chunk_text``?
* How much of each reference summary was turned into supervision?
* Which sections - above all ``judgement`` - are represented, and how thinly?
* Do any records carry a target that could not have come from their chunk?
* Are the alignment scores usable as confidence, or saturated?

It never loads a model, a tokenizer or the network: everything is computed from
the JSONL the build already wrote, so it can run on a laptop against a build made
on a GPU box.

Usage
-----
    python scripts/diagnose_dataset.py --dir artifacts/training_data/full
    python scripts/diagnose_dataset.py --dir artifacts/training_data/full \
        --json artifacts/diagnosis_full.json --md artifacts/diagnosis_full.md
    python scripts/diagnose_dataset.py --dir old_build --baseline artifacts/training_data/full

Exit code: 0 = every hard gate passed, 1 = at least one hard gate failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "validation", "test")

# Hard gates. These are failure conditions, not preferences: a build that trips
# one of them is not trainable as-is.
GATE_UNASSIGNED_RATE = 0.20      # documented band in the project guide is 10-20%
GATE_DEGENERATE_TARGETS = 0      # a target longer than its chunk cannot exist
GATE_JUDGEMENT_SHARE = 0.10      # share of docs whose summary keeps judgement signal

_CONTENT_WORD = re.compile(r"[a-z]{4,}")

# Mirrors config.MAX_INPUT_LENGTH. Kept local so this script stays importable
# without the project's dependency set.
MAX_TARGET_LENGTH_HINT = 1024
_SENTENCE_END = re.compile(r"[.!?]\s*$")


def _words(text: str) -> int:
    return len(text.split())


def _content_words(text: str) -> set[str]:
    return set(_CONTENT_WORD.findall(text.lower()))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _shingles(text: str, size: int = 8) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + size]) for i in range(max(0, len(words) - size + 1))}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return float(ordered[index])


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p10": round(_percentile(values, 10), 4),
        "p90": round(_percentile(values, 90), 4),
        "max": round(max(values), 4),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def diagnose(build_dir: Path, examples: int = 3) -> dict:
    """Compute the full diagnosis for one build directory."""
    records_by_split = {split: _read_jsonl(build_dir / f"{split}.jsonl") for split in SPLITS}
    per_document_records: dict[str, list[dict]] = defaultdict(list)
    records = [row for rows in records_by_split.values() for row in rows]
    references = _read_jsonl(build_dir / "reference_summaries.jsonl")
    unassigned = _read_jsonl(build_dir / "unassigned.jsonl")
    build_stats = {}
    stats_path = build_dir / "build_stats.json"
    if stats_path.exists():
        build_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    for row in (row for rows in records_by_split.values() for row in rows):
        per_document_records[row["doc_id"]].append(row)

    report: dict = {
        "build_dir": str(build_dir),
        "success": bool(records),
        "counts": {
            "records": len(records),
            "records_per_split": {split: len(rows) for split, rows in records_by_split.items()},
            "documents": len(references),
            "unique_doc_ids": len({r["doc_id"] for r in records}),
        },
        "sections": dict(Counter(r["section"] for r in records)),
        "findings": {},
        "gates": {},
    }
    if not records:
        report["success"] = False
        return report

    # --- per-section shape -------------------------------------------------- #
    per_section = {}
    for section, count in Counter(r["section"] for r in records).most_common():
        subset = [r for r in records if r["section"] == section]
        scores = [s for r in subset for s in r.get("alignment_scores", [])]
        per_section[section] = {
            "records": count,
            "share": round(count / len(records), 4),
            "target_words": _stats([_words(r["aligned_target"]) for r in subset]),
            "chunk_words": _stats([_words(r["chunk_text"]) for r in subset]),
            "alignment_score": _stats([float(s) for s in scores]),
        }
    report["per_section"] = per_section

    # --- the two ways a target can be impossible ----------------------------- #
    longer = [
        r for r in records if _words(r["aligned_target"]) > _words(r["chunk_text"])
    ]
    wordless = [
        r for r in records if not re.search(r"[a-z]{3,}", r["aligned_target"])
    ]
    report["findings"]["target_longer_than_chunk"] = {
        "records": len(longer),
        "share": round(len(longer) / len(records), 4),
        "examples": [
            {
                "doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "section": r["section"],
                "chunk_words": _words(r["chunk_text"]),
                "target_words": _words(r["aligned_target"]),
                "selection": "highest-CE chunk for the sentence",
            }
            for r in longer[:examples]
        ],
    }
    report["findings"]["target_without_prose"] = {
        "records": len(wordless),
        "share": round(len(wordless) / len(records), 4),
        "examples": [
            {"doc_id": r["doc_id"], "section": r["section"], "target": r["aligned_target"]}
            for r in wordless[:examples]
        ],
    }

    # --- target/chunk lexical grounding ------------------------------------- #
    coverage = [
        len(_content_words(r["aligned_target"]) & _content_words(r["chunk_text"]))
        / max(1, len(_content_words(r["aligned_target"])))
        for r in records
    ]
    report["findings"]["target_content_word_coverage_of_chunk"] = _stats(coverage)
    report["findings"]["records_below_20pct_coverage"] = sum(1 for c in coverage if c < 0.2)

    matched_sentences = [s for r in records for s in r.get("matched_sentences", [])]
    verbatim = sum(
        1 for r in records for s in r.get("matched_sentences", []) if s.strip() in r["chunk_text"]
    )
    report["findings"]["matched_sentences"] = {
        "total": len(matched_sentences),
        "unique": len(set(matched_sentences)),
        "verbatim_in_chunk": verbatim,
        "verbatim_share": round(verbatim / len(matched_sentences), 4) if matched_sentences else 0.0,
        "under_5_words": sum(1 for s in matched_sentences if _words(s) < 5),
        "under_5_words_share": (
            round(sum(1 for s in matched_sentences if _words(s) < 5) / len(matched_sentences), 4)
            if matched_sentences else 0.0
        ),
        "fragment_only_records": sum(
            1 for r in records
            if r.get("matched_sentences")
            and all(_words(s) < 5 for s in r["matched_sentences"])
        ),
    }

    # --- alignment score distribution: confidence or saturation? ------------- #
    matched_scores = [float(s) for r in records for s in r.get("alignment_scores", [])]
    unassigned_scores = [float(u.get("best_score", 0.0)) for u in unassigned]
    report["findings"]["alignment_scores"] = {
        "matched": _stats(matched_scores),
        "matched_share_above_0_9": (
            round(sum(1 for s in matched_scores if s >= 0.9) / len(matched_scores), 4)
            if matched_scores else 0.0
        ),
        "unassigned_best": _stats(unassigned_scores),
        "unassigned_share_below_0_05": (
            round(sum(1 for s in unassigned_scores if s <= 0.05) / len(unassigned_scores), 4)
            if unassigned_scores else 0.0
        ),
        "unassigned_reasons": dict(Counter(u.get("reason") for u in unassigned)),
    }

    # --- coverage of the reference summaries --------------------------------- #
    used_by_doc: dict[str, set[str]] = defaultdict(set)
    for record in records:
        used_by_doc[record["doc_id"]].update(record.get("matched_sentences", []))
    coverage_per_doc = []
    reference_sentence_lengths = []
    for reference in references:
        sentences = _sentences(reference["original_reference_summary"])
        reference_sentence_lengths.extend(_words(s) for s in sentences)
        if not sentences:
            continue
        coverage_per_doc.append(len(used_by_doc.get(reference["doc_id"], set())) / len(sentences))
    report["findings"]["reference_summary_coverage"] = {
        "documents_measured": len(coverage_per_doc),
        "coverage": _stats(coverage_per_doc),
        "reference_sentence_words": _stats(reference_sentence_lengths),
        "reference_fragments_under_5_words": (
            round(sum(1 for n in reference_sentence_lengths if n < 5) / len(reference_sentence_lengths), 4)
            if reference_sentence_lengths else 0.0
        ),
    }

    # --- judgement signal ---------------------------------------------------- #
    docs = {reference["doc_id"] for reference in references}
    docs_with_judgement = {
        record["doc_id"] for record in records if record["section"] == "judgement"
    }
    report["findings"]["judgement"] = {
        "documents": len(docs),
        "documents_keeping_judgement": len(docs_with_judgement),
        "share_of_documents": round(len(docs_with_judgement) / len(docs), 4) if docs else 0.0,
        "records": sum(1 for r in records if r["section"] == "judgement"),
        "documented_but_lost": sorted(docs - docs_with_judgement)[:examples],
    }

    # --- section coverage, per document and per section ---------------------- #
    # "Judgement is missing" has several distinct causes (the detector found no
    # judgement section, the summary has no judgement sentence, the sentence was
    # attached to another chunk). This separates them instead of reporting a single
    # coverage number that hides the cause.
    docs_by_section: dict[str, set[str]] = defaultdict(set)
    for record in records:
        docs_by_section[record["section"]].add(record["doc_id"])
    docs_with_any_record = {record["doc_id"] for record in records}
    documents_without_records = sorted(docs - docs_with_any_record)
    report["findings"]["section_coverage"] = {
        "documents": len(docs),
        "documents_without_any_record": len(documents_without_records),
        "documents_without_any_record_examples": documents_without_records[:examples],
        "documents_per_section": {
            section: {
                "documents": len(docs_by_section.get(section, set())),
                "share_of_documents": (
                    round(len(docs_by_section.get(section, set())) / len(docs), 4) if docs else 0.0
                ),
            }
            for section in sorted(set(docs_by_section) | set(SPLITS))
        },
        "records_per_document": _stats(
            [len(rows) for rows in per_document_records.values()]
        ),
        "structure_fallback_documents": (
            build_stats.get("dataset", {}).get("documents_structure_fallback")
        ),
        "judgement_outcomes": build_stats.get("judgement", {}).get("documents_by_outcome"),
    }

    # --- duplicate / overlapping supervision --------------------------------- #
    chunk_hashes = Counter(
        hashlib.sha1(r["chunk_text"].encode("utf-8")).hexdigest() for r in records
    )
    cross_split: set[str] = set()
    by_split_text = {
        split: {r["doc_id"] + "|" + r["chunk_text"] for r in rows}
        for split, rows in records_by_split.items()
    }
    split_texts = {split: {r["chunk_text"] for r in rows} for split, rows in records_by_split.items()}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            cross_split |= split_texts[left] & split_texts[right]
    overlapping_pairs = 0
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_doc[record["doc_id"]].append(record)
    for rows in by_doc.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = _shingles(rows[i]["chunk_text"]), _shingles(rows[j]["chunk_text"])
                if a and b and len(a & b) / len(a | b) > 0.10:
                    overlapping_pairs += 1
    report["findings"]["duplication"] = {
        "exact_duplicate_chunks": sum(count - 1 for count in chunk_hashes.values() if count > 1),
        "identical_chunk_text_across_splits": len(cross_split),
        "same_document_chunks_over_10pct_overlap_pairs": overlapping_pairs,
        "duplicate_doc_id_chunk_id": sum(
            count - 1
            for count in Counter((r["doc_id"], r["chunk_id"]) for r in records).values()
            if count > 1
        ),
        "by_split_text_sets": {split: len(texts) for split, texts in by_split_text.items()},
    }

    # --- what the builder itself recorded ------------------------------------ #
    report["builder_stats"] = {
        key: build_stats.get(key)
        for key in ("mode", "generated_at", "alignment", "summary_sentences",
                    "judgement", "quality_flags", "timings", "split_isolation")
        if key in build_stats
    }

    # --- gates ---------------------------------------------------------------- #
    unassigned_rate = build_stats.get("summary_sentences", {}).get("unassigned_rate")
    if unassigned_rate is None:
        total_sentences = len(matched_sentences) + len(unassigned)
        unassigned_rate = len(unassigned) / total_sentences if total_sentences else 0.0
    over_window = sum(
        1 for record in records
        if record.get("n_target_tokens_est", 0) > MAX_TARGET_LENGTH_HINT
    )
    report["gates"] = {
        "no_duplicate_record_keys": {
            "value": report["findings"]["duplication"]["duplicate_doc_id_chunk_id"],
            "limit": 0,
            "passed": report["findings"]["duplication"]["duplicate_doc_id_chunk_id"] == 0,
            "why": "(doc_id, chunk_id) must identify exactly one training record",
        },
        "no_cross_split_duplicate_text": {
            "value": report["findings"]["duplication"]["identical_chunk_text_across_splits"],
            "limit": 0,
            "passed": report["findings"]["duplication"]["identical_chunk_text_across_splits"] == 0,
            "why": "identical chunk text in two splits is leakage",
        },
        "no_target_over_model_window": {
            "value": over_window,
            "limit": 0,
            "passed": over_window == 0,
            "why": f"a target estimated over {MAX_TARGET_LENGTH_HINT} tokens is untrainable",
        },
        "unassigned_rate_within_band": {
            "value": round(unassigned_rate, 4),
            "limit": GATE_UNASSIGNED_RATE,
            "passed": unassigned_rate <= GATE_UNASSIGNED_RATE,
            "why": "the share of each reference summary that never becomes a target",
        },
        "no_impossible_targets": {
            "value": len(longer) + len(wordless),
            "limit": GATE_DEGENERATE_TARGETS,
            "passed": (len(longer) + len(wordless)) == 0,
            "why": "a target longer than its chunk, or with no prose, is not supervision",
        },
        "judgement_signal_present": {
            "value": round(len(docs_with_judgement) / len(docs), 4) if docs else 0.0,
            "limit": GATE_JUDGEMENT_SHARE,
            "passed": bool(docs) and len(docs_with_judgement) / len(docs) >= GATE_JUDGEMENT_SHARE,
            "why": "judgement is the section the product is primarily judged on",
        },
    }
    report["passed"] = all(gate["passed"] for gate in report["gates"].values())
    return report


def _fmt_stats(stats: dict) -> str:
    return (f"n={stats['n']} mean={stats['mean']} median={stats['median']} "
            f"p10={stats['p10']} p90={stats['p90']} max={stats['max']}")


def render_markdown(report: dict) -> str:
    counts = report["counts"]
    findings = report["findings"]
    lines = [
        f"# Dataset diagnosis - `{report['build_dir']}`",
        "",
        f"* records: **{counts['records']}** {counts['records_per_split']}",
        f"* documents: **{counts['documents']}**",
        "",
        "## Gate results",
        "",
        "| Gate | Value | Limit | Result |",
        "| --- | --- | --- | --- |",
    ]
    for name, gate in report["gates"].items():
        lines.append(
            f"| `{name}` | {gate['value']} | {gate['limit']} | "
            f"{'PASS' if gate['passed'] else '**FAIL**'} |"
        )
    lines += ["", "## Sections", "", "| Section | Records | Share | Target words | Chunk words | Best score |",
              "| --- | --- | --- | --- | --- | --- |"]
    for section, data in report["per_section"].items():
        lines.append(
            f"| `{section}` | {data['records']} | {data['share']:.1%} | "
            f"{data['target_words']['median']:.0f} | {data['chunk_words']['median']:.0f} | "
            f"{data['alignment_score']['median']:.3f} |"
        )
    lines += [
        "",
        "## Findings",
        "",
        f"* targets longer than their chunk: **{findings['target_longer_than_chunk']['records']}**"
        f" ({findings['target_longer_than_chunk']['share']:.1%})",
        f"* targets with no prose: **{findings['target_without_prose']['records']}**",
        f"* target content-word coverage of chunk: {_fmt_stats(findings['target_content_word_coverage_of_chunk'])}",
        f"* records below 20% coverage: **{findings['records_below_20pct_coverage']}**",
        f"* matched sentences under 5 words: {findings['matched_sentences']['under_5_words']}"
        f" ({findings['matched_sentences']['under_5_words_share']:.1%})",
        f"* fragment-only records: **{findings['matched_sentences']['fragment_only_records']}**",
        f"* alignment scores (matched): {_fmt_stats(findings['alignment_scores']['matched'])}",
        f"* share of matched scores >= 0.9: {findings['alignment_scores']['matched_share_above_0_9']:.1%}",
        f"* unassigned best scores: {_fmt_stats(findings['alignment_scores']['unassigned_best'])}",
        f"* reference-summary coverage: {_fmt_stats(findings['reference_summary_coverage']['coverage'])}",
        f"* documents keeping judgement signal: "
        f"{findings['judgement']['documents_keeping_judgement']}/"
        f"{findings['judgement']['documents']}"
        f" ({findings['judgement']['share_of_documents']:.1%})",
        f"* identical chunk text across splits: "
        f"{findings['duplication']['identical_chunk_text_across_splits']}",
        f"* same-document chunk pairs over 10% overlap: "
        f"{findings['duplication']['same_document_chunks_over_10pct_overlap_pairs']}",
        "",
    ]
    coverage = findings.get("section_coverage")
    if coverage:
        lines += [
            "## Section coverage (documents, not records)",
            "",
            "| Section | Documents | Share of documents |",
            "| --- | --- | --- |",
        ]
        for section, data in coverage["documents_per_section"].items():
            if data["documents"] == 0 and section in ("train", "validation", "test"):
                continue
            lines.append(
                f"| `{section}` | {data['documents']} | {data['share_of_documents']:.1%} |"
            )
        lines += [
            "",
            f"* documents with no records at all: "
            f"**{coverage['documents_without_any_record']}**",
            f"* structure-fallback documents (no section signal): "
            f"{coverage.get('structure_fallback_documents')}",
            f"* judgement outcomes recorded by the builder: {coverage.get('judgement_outcomes')}",
            "",
        ]

    timings = report.get("builder_stats", {}).get("timings")
    if timings:
        lines += [
            "## Build cost",
            "",
            f"* total: {timings['total_seconds']}s "
            f"({timings.get('seconds_per_document')}s/document)",
            f"* alignment: {timings['alignment_seconds']}s "
            f"({timings.get('alignment_share', 0):.0%} of the run), "
            f"{timings.get('pairs_scored')} pairs at {timings.get('pairs_per_second')} pairs/s",
            "",
        ]
    if findings["target_longer_than_chunk"]["examples"]:
        lines += ["## Impossible targets (first examples)", ""]
        for example in findings["target_longer_than_chunk"]["examples"]:
            lines.append(
                f"* `{example['doc_id']}` chunk {example['chunk_id']} "
                f"(`{example['section']}`): {example['target_words']} target words "
                f"vs {example['chunk_words']} chunk words"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose a built training dataset offline.")
    parser.add_argument("--dir", required=True, help="Build directory (contains train.jsonl ...)")
    parser.add_argument("--json", default=None, help="Write the full report as JSON here.")
    parser.add_argument("--md", default=None, help="Write a Markdown report here.")
    parser.add_argument("--examples", type=int, default=3, help="Examples per finding (default 3).")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Another build directory to print a headline comparison against.",
    )
    args = parser.parse_args(argv)

    build_dir = Path(args.dir)
    if not (build_dir / "build_stats.json").exists() and not any(
        (build_dir / f"{split}.jsonl").exists() for split in SPLITS
    ):
        raise SystemExit(f"{build_dir} does not look like a build directory.")

    report = diagnose(build_dir, examples=args.examples)
    print(render_markdown(report))

    if args.baseline:
        baseline = diagnose(Path(args.baseline), examples=1)
        print("\n## Comparison\n")
        print("| Metric | " + args.baseline + " | " + args.dir + " |")
        print("| --- | --- | --- |")
        for label, a, b in (
            ("records", baseline["counts"]["records"], report["counts"]["records"]),
            ("documents", baseline["counts"]["documents"], report["counts"]["documents"]),
            ("impossible targets",
             baseline["findings"]["target_longer_than_chunk"]["records"],
             report["findings"]["target_longer_than_chunk"]["records"]),
            ("judgement records",
             baseline["findings"]["judgement"]["records"],
             report["findings"]["judgement"]["records"]),
            ("fragment-only records",
             baseline["findings"]["matched_sentences"]["fragment_only_records"],
             report["findings"]["matched_sentences"]["fragment_only_records"]),
        ):
            print(f"| {label} | {a} | {b} |")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.json}")
    if args.md:
        Path(args.md).write_text(render_markdown(report) + "\n", encoding="utf-8")
        print(f"Wrote {args.md}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

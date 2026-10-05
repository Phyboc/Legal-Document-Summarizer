#!/usr/bin/env python3
"""Validate section-summary alignment on a sample of real documents.

Week 2 validation task: run on 15-20 documents from the training set,
manually inspect alignment quality, and log the unassigned rate.

Each document goes through ``build_dataset.build_document`` - the exact code
path that produces the training records - so what is inspected here is what
ends up in the dataset, not a parallel re-implementation of it.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import build_document, detect_alignment_backend, load_split, resolve_columns
from config import ALIGNMENT_RELATIVE_MARGIN, ALIGNMENT_THRESHOLD


def _preview(text: str, limit: int) -> str:
    return (text[:limit] + "...") if len(text) > limit else text


def validate_single_document(
    doc_text: str,
    reference_summary: str,
    doc_id: str = "sample",
    split: str = "train",
    verbose: bool = True,
) -> dict:
    """Validate alignment for a single document."""
    build = build_document(doc_id, split, doc_text, reference_summary)
    if build.error:
        return {"success": False, "doc_id": doc_id, "error": build.error}

    total = build.summary_sentences
    unassigned = len(build.unassigned)
    per_section = Counter(record["section"] for record in build.records)
    sentences_per_section = Counter()
    for record in build.records:
        sentences_per_section[record["section"]] += len(record["matched_sentences"])

    metrics = {
        "total_sentences": total,
        "assigned_sentences": total - unassigned,
        "unassigned_sentences": unassigned,
        "unassigned_rate": round(unassigned / total, 3) if total else 0.0,
        "multi_label_sentences": build.multi_label_sentences,
        "chunks_created": build.chunks_created,
        "records": len(build.records),
        "structure_fallback": build.structure_fallback,
        "target_overflow_sentences": build.target_overflow_sentences,
        "assignment_counts": dict(sentences_per_section),
    }

    if verbose:
        print(f"\n📋 Document: {doc_id}")
        print("=" * 70)
        print(f"Chunks: {build.chunks_created} | records: {len(build.records)}"
              f" | structure fallback: {build.structure_fallback}")
        print(f"Reference summary length: {len(reference_summary)} chars")
        print(f"Total sentences in reference: {total}")
        print()

        for record in build.records:
            print(f"  [chunk {record['chunk_index']} · {record['section']}]"
                  f" → {len(record['matched_sentences'])} sentence(s)"
                  f" (scores {record['alignment_scores']})")
            print(f"    Chunk:  {_preview(record['chunk_text'].replace(chr(10), ' '), 120)}")
            print(f"    Target: {_preview(record['aligned_target'], 120)}")
        print()

        if build.unassigned:
            print(f"⚠️  Unassigned sentences ({unassigned}):")
            for item in build.unassigned[:3]:
                print(f"    [{item.get('best_label', '?')}: {item.get('best_score', 0):.2f}]"
                      f" {_preview(item['sentence'], 80)}")
            if unassigned > 3:
                print(f"    ... and {unassigned - 3} more")
        else:
            print("✓ All sentences assigned!")

        print()
        print(f"Metrics: {metrics}")
        print("=" * 70)

    return {
        "success": True,
        "doc_id": doc_id,
        "sections": sorted(per_section),
        "metrics": metrics,
        "unassigned": build.unassigned,
        "records": build.records,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-document metrics and print the summary block."""
    total_sentences = sum(r["metrics"]["total_sentences"] for r in results)
    total_unassigned = sum(r["metrics"]["unassigned_sentences"] for r in results)
    total_multi = sum(r["metrics"]["multi_label_sentences"] for r in results)
    unassigned_rate = total_unassigned / total_sentences if total_sentences else 0.0
    multi_label_rate = total_multi / total_sentences if total_sentences else 0.0

    per_section = Counter()
    for r in results:
        per_section.update(r["metrics"]["assignment_counts"])

    print("\n" + "=" * 70)
    print("📈 AGGREGATED VALIDATION RESULTS")
    print("=" * 70)
    print(f"Valid documents processed: {len(results)}")
    print(f"Total sentences: {total_sentences}")
    print(f"Assigned: {total_sentences - total_unassigned}")
    print(f"Unassigned: {total_unassigned} ({unassigned_rate*100:.1f}%)")
    print(f"Multi-label: {total_multi} ({multi_label_rate*100:.1f}%)")
    print(f"Structure fallbacks: {sum(r['metrics']['structure_fallback'] for r in results)}")
    print()
    print("Sentences written per section (primary chunk only):")
    for section, count in per_section.most_common():
        print(f"  {section:25s}: {count:4d}")

    print()
    if unassigned_rate > 0.4:
        print("⚠️  High unassigned rate (>40%). Consider:")
        print("    - Lowering ALIGNMENT_THRESHOLD in config.py")
        print("    - Checking section detection quality")
        print("    - Validating reference summary quality")
    elif unassigned_rate > 0.2:
        print("⚠️  Moderate unassigned rate (20-40%). This is expected with a general-domain")
        print("    Cross-Encoder on legal documents. Document this limitation in the report.")
    else:
        print("✓ Alignment quality looks good!")
    print("=" * 70)

    return {
        "valid_documents": len(results),
        "total_sentences": total_sentences,
        "assigned": total_sentences - total_unassigned,
        "unassigned": total_unassigned,
        "unassigned_rate": round(unassigned_rate, 3),
        "multi_label_rate": round(multi_label_rate, 3),
        "per_section_sentences": dict(per_section),
        "threshold": ALIGNMENT_THRESHOLD,
        "relative_margin": ALIGNMENT_RELATIVE_MARGIN,
    }


def validate_sample(sample_size: int = 15, split: str = "train", verbose: bool = True) -> dict:
    """Validate alignment on the first ``sample_size`` documents of an IN-ABS split."""
    dataset = load_split(split, limit=sample_size)
    text_col, summary_col, file_col = resolve_columns(dataset)

    print(f"📊 Validating alignment on {len(dataset)} documents from '{split}'...")

    results = []
    for idx, example in enumerate(dataset):
        doc_id = str(example.get(file_col) or idx) if file_col else str(idx)
        doc_text = example.get(text_col) or ""
        reference_summary = example.get(summary_col) or ""
        if not doc_text or not reference_summary:
            print(f"  [{idx + 1}] Skipped (missing data)")
            continue

        result = validate_single_document(doc_text, reference_summary, doc_id, split, verbose)
        if result["success"]:
            results.append(result)
        else:
            print(f"  [{idx + 1}] ❌ Error: {result['error']}")

    if not results:
        return {"success": False, "error": "No documents could be validated"}

    return {"success": True, **summarize(results), "sample_results": results}


def validate_demo() -> dict:
    """Demo validation with a small hardcoded example."""
    print("\n📝 Running DEMO validation with example text...\n")

    doc_text = """FACTS OF THE CASE
The appellant, a 45-year-old male, filed a petition before this Court challenging the order dated 15th March, 2020 passed by the High Court of Delhi. The respondent, the State of Delhi, contested the petition on the grounds that the appellant's claims were baseless and contrary to law.

ARGUMENTS
Learned counsel for the appellant submitted that the impugned order was passed without considering the material evidence on record. He further contended that the High Court failed to appreciate the case law laid down in landmark decisions.

On behalf of the respondent, it was argued that the appellant had failed to exhaust all remedies available under law before approaching this Court.

REASONING
In our opinion, the contention raised by the appellant is meritorious. We have considered the statute and relevant case law. Section 101 of the Indian Evidence Act, 1872 places the burden of proof on the person asserting. We find that the High Court overlooked critical evidence. Reliance is placed on the decision in ABC v. XYZ.

JUDGEMENT
The appeal is allowed. The impugned order is set aside. We direct the High Court to reconsider the matter in light of the evidence. Costs are imposed.
"""

    reference_summary = (
        "The appellant challenged a High Court order from March 2020 filed by the respondent, the State. "
        "Counsel for the appellant argued the order was passed without considering evidence and failed to apply landmark case law. "
        "The respondent contended the appellant had not exhausted available remedies. "
        "The Court found the appellant's contentions meritorious and held that the High Court overlooked critical evidence. "
        "The appeal is allowed and the impugned order is set aside."
    )

    return validate_single_document(doc_text, reference_summary, doc_id="demo_case")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate section-summary alignment")
    parser.add_argument("--sample-size", type=int, default=15, help="Number of documents to validate")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--demo", action="store_true", help="Run demo validation instead")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-document output")
    parser.add_argument("--output", type=Path, help="Write the aggregated results to this JSON file")
    args = parser.parse_args()

    backend = detect_alignment_backend()
    if backend == "stub":
        print("⚠️  Cross-Encoder stub in use (sentence-transformers missing) - scores are meaningless.")

    if args.demo:
        outcome = validate_demo()
    else:
        outcome = validate_sample(args.sample_size, args.split, verbose=not args.quiet)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"backend": backend, **outcome}, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

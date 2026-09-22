#!/usr/bin/env python3
"""Validate section-summary alignment on a sample of real documents.

Week 2 validation task: run on 15-20 documents from the training set,
manually inspect alignment quality, and log the unassigned rate.
"""
import sys
import json
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.data.section_detector import detect_sections, group_by_section
from src.data.cleaning import split_paragraphs
from src.data.chunker import chunk_section, Chunk
from src.data.section_summary_alignment import align_summary_to_sections, batch_align_dataset
from config import SECTIONS


def detect_sections_as_text(doc_text: str) -> dict[str, str]:
    """Split a raw document into paragraphs, tag each with a section, and
    join back into {section_label: full_section_text}."""
    paragraphs = split_paragraphs(doc_text)
    if not paragraphs:
        return {}
    tagged = detect_sections(paragraphs)
    grouped = group_by_section(tagged)
    return {label: " ".join(p.text for p in paras) for label, paras in grouped.items()}


def load_in_abs_dataset(split: str = "train") -> Optional:
    """Load the IN-Abs dataset from Hugging Face Hub."""
    try:
        dataset = load_dataset("percins/IN-ABS")
        return dataset
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("   Falling back to MARRO dataset...")
        try:
            from datasets import DatasetDict, Dataset
            import os

            marro_path = Path(__file__).parent.parent / "artifacts" / "marro"
            if marro_path.exists():
                print(f"   Using MARRO data from {marro_path}")
                # This would need custom loading logic for MARRO
                return None
        except:
            pass
        return None


def validate_single_document(
    doc_text: str,
    reference_summary: str,
    doc_id: str = "sample",
    verbose: bool = True,
) -> dict:
    """Validate alignment for a single document."""
    # Detect sections
    sections = detect_sections_as_text(doc_text)
    if not sections:
        return {"success": False, "error": "No sections detected"}

    # Align
    result = align_summary_to_sections(
        doc_id=doc_id,
        section_texts=sections,
        reference_summary=reference_summary,
    )

    if verbose:
        print(f"\n📋 Document: {doc_id}")
        print("=" * 70)
        print(f"Sections detected: {list(sections.keys())}")
        print(f"Reference summary length: {len(reference_summary)} chars")
        print(f"Total sentences in reference: {result.metrics['total_sentences']}")
        print()

        for section, aligned_text in result.aligned_summaries.items():
            num_sents = len(aligned_text.split(".")) if aligned_text else 0
            print(f"  [{section}] → {num_sents} sentence(s) aligned")
            if aligned_text:
                preview = (aligned_text[:120] + "...") if len(aligned_text) > 120 else aligned_text
                print(f"    Preview: {preview}")
        print()

        if result.unassigned_sentences:
            print(f"⚠️  Unassigned sentences ({len(result.unassigned_sentences)}):")
            for item in result.unassigned_sentences[:3]:  # Show first 3
                best_section = item.get("best_section", "?")
                score = item.get("best_score", 0)
                preview = (item["sentence"][:80] + "...") if len(item["sentence"]) > 80 else item["sentence"]
                print(f"    [{best_section}: {score:.2f}] {preview}")
            if len(result.unassigned_sentences) > 3:
                print(f"    ... and {len(result.unassigned_sentences) - 3} more")
        else:
            print("✓ All sentences assigned!")

        print()
        print(f"Metrics: {result.metrics}")
        print("=" * 70)

    return {
        "success": True,
        "doc_id": doc_id,
        "sections": list(sections.keys()),
        "metrics": result.metrics,
        "unassigned": result.unassigned_sentences,
        "aligned": result.aligned_summaries,
    }


def validate_sample(
    sample_size: int = 15,
    verbose: bool = True,
) -> dict:
    """Validate alignment on a sample of documents from the dataset."""
    dataset = load_in_abs_dataset(split="train")

    if not dataset:
        print("⚠️  Could not load IN-Abs dataset. Running demo validation instead...")
        # Demo with hardcoded example
        return validate_demo()

    print(f"📊 Validating alignment on {sample_size} documents...")
    print()

    results = []
    valid_count = 0

    for idx in range(min(sample_size, len(dataset["train"]))):
        example = dataset["train"][idx]
        doc_id = example.get("file") or str(idx)
        doc_text = example.get("text", "")
        reference_summary = example.get("summary", "")

        if not doc_text or not reference_summary:
            print(f"  [{idx + 1}] Skipped (missing data)")
            continue

        try:
            result = validate_single_document(
                doc_text=doc_text,
                reference_summary=reference_summary,
                doc_id=doc_id,
                verbose=verbose,
            )

            if result["success"]:
                results.append(result)
                valid_count += 1
        except Exception as e:
            print(f"  [{idx + 1}] ❌ Error: {e}")
            continue

    # Aggregate metrics
    if results:
        total_sentences = sum(r["metrics"]["total_sentences"] for r in results)
        total_assigned = sum(r["metrics"]["assigned_sentences"] for r in results)
        total_unassigned = sum(r["metrics"]["unassigned_sentences"] for r in results)
        avg_unassigned_rate = total_unassigned / total_sentences if total_sentences > 0 else 0

        print("\n" + "=" * 70)
        print("📈 AGGREGATED VALIDATION RESULTS")
        print("=" * 70)
        print(f"Valid documents processed: {valid_count}")
        print(f"Total sentences: {total_sentences}")
        print(f"Assigned: {total_assigned}")
        print(f"Unassigned: {total_unassigned} ({avg_unassigned_rate*100:.1f}%)")
        print()

        # Per-section analysis
        per_section = {}
        for section in SECTIONS:
            per_section[section] = sum(r["metrics"]["assignment_counts"].get(section, 0) for r in results)
        print("Assignments per section:")
        for section, count in per_section.items():
            print(f"  {section:25s}: {count:4d}")

        print()
        if avg_unassigned_rate > 0.4:
            print("⚠️  High unassigned rate (>40%). Consider:")
            print("    - Lowering ALIGNMENT_THRESHOLD in config.py")
            print("    - Checking section detection quality")
            print("    - Validating reference summary quality")
        elif avg_unassigned_rate > 0.2:
            print("⚠️  Moderate unassigned rate (20-40%). This is expected with a general-domain")
            print("    Cross-Encoder on legal documents. Document this limitation in the report.")
        else:
            print("✓ Alignment quality looks good!")

        print("=" * 70)

        return {
            "success": True,
            "valid_documents": valid_count,
            "total_sentences": total_sentences,
            "assigned": total_assigned,
            "unassigned": total_unassigned,
            "unassigned_rate": round(avg_unassigned_rate, 3),
            "per_section": per_section,
            "sample_results": results,
        }

    return {"success": False, "error": "No documents could be validated"}


def validate_demo() -> dict:
    """Demo validation with a small hardcoded example."""
    print("\n📝 Running DEMO validation with example text...\n")

    doc_text = """
    FACTS OF THE CASE
    The appellant, a 45-year-old male, filed a petition before this Court
    challenging the order dated 15th March, 2020 passed by the High Court of Delhi.
    The respondent, the State of Delhi, contested the petition on the grounds that
    the appellant's claims were baseless and contrary to law.

    ARGUMENTS
    Learned counsel for the appellant submitted that the impugned order was
    passed without considering the material evidence on record. He further contended
    that the High Court failed to appreciate the case law laid down in landmark decisions.

    On behalf of the respondent, it was argued that the appellant had failed to exhaust
    all remedies available under law before approaching this Court.

    REASONING
    In our opinion, the contention raised by the appellant is meritorious. We have
    considered the statute and relevant case law. Section 101 of the Indian Evidence Act,
    1872 places the burden of proof on the person asserting. We find that the High Court
    overlooked critical evidence. Reliance is placed on the decision in ABC v. XYZ.

    JUDGEMENT
    The appeal is allowed. The impugned order is set aside. We direct the High Court
    to reconsider the matter in light of the evidence. Costs are imposed.
    """

    reference_summary = """
    The appellant challenged a High Court order from March 2020 filed by the respondent, the State.
    Counsel for the appellant argued the order was passed without considering evidence and failed to apply landmark case law.
    The respondent contended the appellant had not exhausted available remedies.
    The Court found the appellant's contentions meritorious and held that the High Court overlooked critical evidence.
    The appeal is allowed and the impugned order is set aside.
    """

    result = validate_single_document(
        doc_text=doc_text,
        reference_summary=reference_summary,
        doc_id="demo_case",
        verbose=True,
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate section-summary alignment")
    parser.add_argument("--sample-size", type=int, default=15, help="Number of documents to validate")
    parser.add_argument("--demo", action="store_true", help="Run demo validation instead")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if args.demo:
        validate_demo()
    else:
        validate_sample(sample_size=args.sample_size, verbose=not args.quiet)

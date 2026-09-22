"""Section-Summary Alignment: derive section-specific training targets from full reference summaries.

Uses Cross-Encoder to determine which sentences of the reference summary
correspond to which judgment sections. Implements multi-label (sentences
may belong to multiple sections) and conservative (unassigned if below threshold)
assignment strategy.
"""
import sys
sys.path.insert(0, ".")

from dataclasses import dataclass, field
from typing import Optional
from config import (
    ALIGNMENT_THRESHOLD,
    ALIGNMENT_RELATIVE_MARGIN,
    SECTIONS,
)
from src.shared.cross_encoder import CrossEncoderSingleton
from src.postprocess.cite_verify import split_sentences as sent_tokenize


@dataclass
class AlignmentResult:
    """Result of aligning a document's summary to its sections."""
    doc_id: str
    aligned_summaries: dict[str, str]  # {section_label: aligned_text}
    unassigned_sentences: list[dict] = field(default_factory=list)  # [{"sentence": ..., "best_score": ...}, ...]
    metrics: dict = field(default_factory=dict)  # unassigned_rate, assignment_counts, etc.


def align_summary_to_sections(
    doc_id: str,
    section_texts: dict[str, str],
    reference_summary: str,
    threshold: float = ALIGNMENT_THRESHOLD,
    allow_multi_label: bool = True,
    relative_margin: float = ALIGNMENT_RELATIVE_MARGIN,
) -> AlignmentResult:
    """
    Align reference summary sentences to judgment sections.

    Args:
        doc_id: Document identifier for logging
        section_texts: {section_label: full_section_text}
        reference_summary: The full reference summary from the dataset
        threshold: Absolute similarity threshold (0.0-1.0)
        allow_multi_label: If True, assign to multiple sections if threshold met
        relative_margin: If set, assign if score >= best_score * relative_margin

    Returns:
        AlignmentResult with aligned summaries per section, unassigned sentences, and metrics
    """
    cross_encoder = CrossEncoderSingleton()

    summary_sentences = sent_tokenize(reference_summary)
    if not summary_sentences:
        return AlignmentResult(
            doc_id=doc_id,
            aligned_summaries={s: "" for s in section_texts.keys()},
            unassigned_sentences=[],
            metrics={"total_sentences": 0, "unassigned_rate": 0.0},
        )

    section_labels = list(section_texts.keys())
    aligned = {label: [] for label in section_labels}
    unassigned = []
    assignment_counts = {label: 0 for label in section_labels}

    for sent in summary_sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Score this sentence against each section
        pairs = [[sent, section_texts[label]] for label in section_labels]
        scores = cross_encoder.score(pairs, normalize=True)
        max_score = max(scores)

        # Conservative gate: a section must clear the absolute threshold to
        # be eligible at all. This is the ONLY thing that decides unassigned
        # vs assigned — nothing widens the net below `threshold`.
        qualifying = [
            (section_labels[i], score) for i, score in enumerate(scores)
            if score >= threshold
        ]

        if not qualifying:
            unassigned.append({
                "sentence": sent,
                "best_score": float(max_score),
                "best_section": section_labels[scores.index(max_score)],
            })
            continue

        if allow_multi_label:
            # Among qualifying sections, keep the dominant one plus any
            # others within `relative_margin` of the best qualifying score —
            # this narrows multi-label spread rather than widening eligibility.
            best_qualifying = max(s for _, s in qualifying)
            to_assign = [
                (label, score) for label, score in qualifying
                if score >= best_qualifying * relative_margin
            ]
            for label, score in to_assign:
                aligned[label].append(sent)
                assignment_counts[label] += 1
        else:
            best_label = max(qualifying, key=lambda x: x[1])[0]
            aligned[best_label].append(sent)
            assignment_counts[best_label] += 1

    # Build final aligned summaries (join sentences back into text)
    aligned_text = {
        label: " ".join(sents)
        for label, sents in aligned.items()
        if sents
    }

    # Metrics
    total_assigned = sum(len(sents) for sents in aligned.values())
    unassigned_rate = len(unassigned) / len(summary_sentences) if summary_sentences else 0.0

    metrics = {
        "total_sentences": len(summary_sentences),
        "assigned_sentences": total_assigned,
        "unassigned_sentences": len(unassigned),
        "unassigned_rate": round(unassigned_rate, 3),
        "assignment_counts": assignment_counts,
    }

    return AlignmentResult(
        doc_id=doc_id,
        aligned_summaries=aligned_text,
        unassigned_sentences=unassigned,
        metrics=metrics,
    )


def batch_align_dataset(
    dataset,
    section_detector,
    chunker,
    split: str = "train",
    sample_size: Optional[int] = None,
    verbose: bool = True,
) -> tuple[list[AlignmentResult], dict]:
    """
    Align all (or a sample of) documents in a dataset.

    Args:
        dataset: HF Dataset with 'document_text' and 'summary' columns
        section_detector: Function(doc_text) -> {section_label: section_text}
        chunker: Function(section_dict) -> [Chunk, ...]
        split: Dataset split to process ("train", "test", "validation")
        sample_size: If set, process only first N documents
        verbose: Print progress

    Returns:
        (list of AlignmentResult, aggregated metrics dict)
    """
    from src.data.section_detector import detect_sections, group_by_section
    from src.data.cleaning import split_paragraphs

    split_data = dataset[split] if split in dataset else dataset
    total_docs = len(split_data)

    if sample_size:
        total_docs = min(sample_size, total_docs)

    results = []
    all_metrics = {
        "total_documents": total_docs,
        "total_sentences": 0,
        "assigned_sentences": 0,
        "unassigned_sentences": 0,
        "avg_unassigned_rate": 0.0,
        "per_section_counts": {s: 0 for s in SECTIONS},
    }

    for idx in range(total_docs):
        example = split_data[idx]
        doc_id = example.get("file") or str(idx)
        doc_text = example.get("text", "")
        reference_summary = example.get("summary", "")

        if not doc_text or not reference_summary:
            if verbose and idx % max(1, total_docs // 10) == 0:
                print(f"  [{idx + 1}/{total_docs}] Skipped (missing text/summary)")
            continue

        try:
            # Split into paragraphs, tag each with a section, then group
            paragraphs = split_paragraphs(doc_text)
            if not paragraphs:
                if verbose and idx % max(1, total_docs // 10) == 0:
                    print(f"  [{idx + 1}/{total_docs}] Skipped (no paragraphs)")
                continue

            tagged = detect_sections(paragraphs)
            grouped = group_by_section(tagged)
            sections = {
                label: " ".join(p.text for p in paras)
                for label, paras in grouped.items()
            }
            if not sections:
                if verbose and idx % max(1, total_docs // 10) == 0:
                    print(f"  [{idx + 1}/{total_docs}] Skipped (no sections detected)")
                continue

            # Align
            result = align_summary_to_sections(
                doc_id=doc_id,
                section_texts=sections,
                reference_summary=reference_summary,
            )

            results.append(result)

            # Accumulate metrics
            all_metrics["total_sentences"] += result.metrics["total_sentences"]
            all_metrics["assigned_sentences"] += result.metrics["assigned_sentences"]
            all_metrics["unassigned_sentences"] += result.metrics["unassigned_sentences"]

            for section, count in result.metrics["assignment_counts"].items():
                if section in all_metrics["per_section_counts"]:
                    all_metrics["per_section_counts"][section] += count

            if verbose and (idx + 1) % max(1, total_docs // 10) == 0:
                print(
                    f"  [{idx + 1}/{total_docs}] {result.metrics['unassigned_rate']*100:.1f}% "
                    f"unassigned ({result.metrics['unassigned_sentences']} sentences)"
                )

        except Exception as e:
            if verbose:
                print(f"  [{idx + 1}/{total_docs}] ERROR: {e}")
            continue

    if results:
        all_metrics["avg_unassigned_rate"] = round(
            all_metrics["unassigned_sentences"] / all_metrics["total_sentences"]
            if all_metrics["total_sentences"] > 0
            else 0.0,
            3,
        )

    if verbose:
        print(f"\n✓ Aligned {len(results)} documents")
        print(f"  Total sentences: {all_metrics['total_sentences']}")
        print(f"  Assigned: {all_metrics['assigned_sentences']}")
        print(f"  Unassigned: {all_metrics['unassigned_sentences']} "
              f"({all_metrics['avg_unassigned_rate']*100:.1f}%)")
        print(f"  Per section: {all_metrics['per_section_counts']}")

    return results, all_metrics


def create_training_data_from_alignment(
    alignment_results: list[AlignmentResult],
    chunks: dict,  # {doc_id: [Chunk, ...]}
) -> list[dict]:
    """
    Convert alignment results into training examples.

    For each chunk, find the aligned summary text for its section,
    creating {chunk_text, section_label, aligned_target_summary} examples.

    Args:
        alignment_results: Output from batch_align_dataset
        chunks: {doc_id: [Chunk objects]}

    Returns:
        List of training examples: {
            "doc_id": str,
            "chunk_id": int,
            "section_label": str,
            "chunk_text": str,
            "aligned_target": str
        }
    """
    training_examples = []

    for result in alignment_results:
        doc_id = result.doc_id
        if doc_id not in chunks:
            continue

        doc_chunks = chunks[doc_id]
        aligned_summaries = result.aligned_summaries

        for chunk in doc_chunks:
            section = chunk.section
            if section not in aligned_summaries:
                continue

            target_summary = aligned_summaries[section]
            if not target_summary.strip():
                continue

            training_examples.append({
                "doc_id": doc_id,
                "chunk_id": chunk.chunk_id,
                "section_label": section,
                "chunk_text": chunk.text,
                "aligned_target": target_summary,
            })

    return training_examples

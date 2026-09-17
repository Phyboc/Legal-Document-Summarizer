"""Build the section-aligned IN-ABS training dataset for chunk-level BART fine-tuning.

What this script does
---------------------
Walks the official IN-ABS splits document by document, runs the judgement
through the *existing* inference pipeline stages, aligns each reference-summary
sentence to the chunk(s) it came from with the shared Cross-Encoder, and writes
chunk-level supervised records plus build statistics. Optionally validates and
pushes the result to the Hugging Face Dataset Hub.

What this script deliberately does not do
-----------------------------------------
No BART training, no model loading, no changes to inference/API/UI behaviour.
Dataset *creation* only.

Key invariants
--------------
* **Document-level splitting.** The official IN-ABS ``train`` / ``validation`` /
  ``test`` splits are preserved as-is. Sampling selects whole documents, never
  chunks. A document belongs to exactly one split, and the split boundary is
  drawn *before* chunk expansion, so no chunk of one document can leak into
  another split. This is verified programmatically on every build.
* **``--sample N`` means N complete source documents**, per selected split. It
  never means N chunks, N summary sentences or N training records: the number of
  records that results is whatever the documents yield.
* **No invented targets.** A chunk with no aligned target is excluded, not
  padded. A sentence that clears no section threshold is left unassigned.

Three objects that must stay distinct
-------------------------------------
* ``chunk_text`` - source input to the model.
* ``aligned_target`` - Cross-Encoder-derived *pseudo-target* for chunk-level
  supervised training. Not ground truth.
* ``original_reference_summary`` - the IN-ABS summary as published, the
  **document-level evaluation reference**. Written to
  ``reference_summaries.jsonl`` once per document; deliberately not duplicated
  into every chunk record.

Usage
-----
    python scripts/build_dataset.py --sample 10  --output_dir artifacts/training_data/sample10
    python scripts/build_dataset.py --sample 100 --output_dir artifacts/training_data/val100
    python scripts/build_dataset.py --full --output_dir artifacts/training_data/full
    python scripts/build_dataset.py --push-to-hub \
        --repo-id <HF_USERNAME>/in-abs-section-aligned \
        --output_dir artifacts/training_data/full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Make ``config`` and ``src.*`` importable no matter which directory the script
# is invoked from. The library modules themselves use ``sys.path.insert(0, ".")``
# (the repository convention), which only works when the CWD is the package
# directory - this makes the builder robust to that.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config import MAX_CHUNKS_PER_DOC, MAX_INPUT_LENGTH  # noqa: E402
from src.data.chunker import approximate_tokens, chunk_document  # noqa: E402
from src.data.cleaning import preprocess, split_paragraphs  # noqa: E402
from src.data.pii_redaction import redact  # noqa: E402
from src.data.section_detector import (  # noqa: E402
    detect_sections,
    fallback_to_single_section,
    group_by_section,
)
from src.data.section_summary_alignment import align_summary_to_sections  # noqa: E402

logger = logging.getLogger("build_dataset")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATASET_ID = "percins/IN-ABS"
OFFICIAL_SPLITS = ("train", "validation", "test")

# Column resolution: prefer the documented IN-ABS names, then common aliases, so
# a schema drift produces a clear error instead of a silently wrong dataset.
TEXT_COLUMN_CANDIDATES = ("text", "judgment", "judgement", "document", "content")
SUMMARY_COLUMN_CANDIDATES = ("summary", "reference_summary", "headnote", "abstract")
FILE_COLUMN_CANDIDATES = ("file", "filename", "doc_name", "id", "doc_id")

# Label used for documents whose structure the detector cannot resolve. The
# validated build (artifacts/alignment_validation.json) uses ``full_document``;
# consistently labelling the whole document is honest, whereas reusing a real
# section name such as ``reasoning`` would claim a structure the detector never
# found.
FALLBACK_SECTION_LABEL = "full_document"

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------------------------------------------------------- #
# Per-document build result
# --------------------------------------------------------------------------- #

@dataclass
class DocumentBuild:
    """Everything one source document contributes to the dataset."""

    doc_id: str
    split: str
    records: list[dict] = field(default_factory=list)
    unassigned: list[dict] = field(default_factory=list)
    reference: dict | None = None
    chunks_created: int = 0
    structure_fallback: bool = False
    at_chunk_cap: bool = False
    summary_sentences: int = 0
    multi_label_sentences: int = 0
    assigned_labels: int = 0
    pii_counts: dict = field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------------- #
# IN-ABS loading
# --------------------------------------------------------------------------- #

def _import_datasets():
    """Import the ``datasets`` library, or explain exactly what to install."""
    try:
        import datasets  # noqa: F401
        return datasets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The `datasets` library is required to read IN-ABS and to push to the Hub.\n"
            "Install the project's declared dependencies first:\n"
            "    python -m pip install -r legal-doc-intelligence/requirements.txt\n"
            f"(import error was: {exc})"
        ) from exc


def resolve_columns(dataset) -> tuple[str, str, str | None]:
    """Find the judgement, summary and (optional) file columns in an IN-ABS row set."""
    names = list(dataset.column_names)

    def pick(candidates):
        for candidate in candidates:
            if candidate in names:
                return candidate
        return None

    text_col = pick(TEXT_COLUMN_CANDIDATES)
    summary_col = pick(SUMMARY_COLUMN_CANDIDATES)

    missing = [
        label
        for label, col in (("judgement text", text_col), ("reference summary", summary_col))
        if col is None
    ]
    if missing:
        raise SystemExit(
            f"Could not resolve {' and '.join(missing)} column(s) in {DATASET_ID}. "
            f"Available columns are: {names}. "
            "Update TEXT_COLUMN_CANDIDATES / SUMMARY_COLUMN_CANDIDATES in this script."
        )

    return text_col, summary_col, pick(FILE_COLUMN_CANDIDATES)


def load_split(split: str, limit: int | None = None):
    """Load one IN-ABS split, or its first ``limit`` documents.

    ``limit`` selects complete leading documents in the official split order -
    deterministic, so ``--sample N`` reproduces byte-for-byte across runs.
    """
    datasets = _import_datasets()
    split_spec = f"{split}[:{limit}]" if limit else split
    logger.info("Loading %s split='%s'", DATASET_ID, split_spec)
    try:
        return datasets.load_dataset(DATASET_ID, split=split_spec)
    except Exception as exc:  # pragma: no cover - network/dataset dependent
        raise SystemExit(
            f"Failed to load {DATASET_ID} (split={split_spec}): {exc}\n"
            "The first run downloads the parquet files from the Hub, so it needs network access."
        ) from exc


# --------------------------------------------------------------------------- #
# Alignment backend guard
# --------------------------------------------------------------------------- #

def detect_alignment_backend() -> str:
    """Report which Cross-Encoder backend is actually in use.

    ``CrossEncoderSingleton`` silently substitutes ``_MockCrossEncoder`` (a
    bag-of-words overlap stub) when ``sentence-transformers`` is unavailable.
    Alignment produced by that stub is meaningless, so the builder refuses to
    use it by default rather than emitting a dataset whose ``aligned_target``
    values look plausible and are not.
    """
    from src.shared.cross_encoder import get_cross_encoder, _MockCrossEncoder

    try:
        model = get_cross_encoder().get_model()
    except Exception as exc:
        raise SystemExit(
            "Could not load the Cross-Encoder "
            f"({CROSS_ENCODER_MODEL}): {exc}\n"
            "The first run downloads the model from the Hub, so it needs network access."
        ) from exc

    if isinstance(model, _MockCrossEncoder):
        return "stub"
    return type(model).__name__


def require_real_encoder(allow_stub: bool) -> str:
    """Block the build when only the stub encoder is available."""
    backend = detect_alignment_backend()
    if backend == "stub" and not allow_stub:
        raise SystemExit(
            "Refusing to build: sentence-transformers is not installed, so the shared\n"
            "Cross-Encoder fell back to its bag-of-words _MockCrossEncoder stub. Aligned\n"
            "targets from that stub are semantically meaningless.\n\n"
            "Install the encoder and retry:\n"
            "    python -m pip install sentence-transformers\n\n"
            "Or, to exercise the pipeline plumbing only (NOT a valid dataset), pass\n"
            "    --allow-stub-encoder\n"
            "which records alignment_backend='stub' in build_stats.json and blocks any upload."
        )
    if backend == "stub":
        logger.warning(
            "USING THE MOCK CROSS-ENCODER STUB - the aligned targets are NOT meaningful. "
            "This build is for pipeline plumbing only and cannot be uploaded."
        )
    return backend


# --------------------------------------------------------------------------- #
# Per-document processing
# --------------------------------------------------------------------------- #

def build_document(
    doc_id: str,
    split: str,
    text: str,
    summary: str,
    *,
    redact_names: bool = False,
    strip_preamble: bool = True,
    fallback_label: str = FALLBACK_SECTION_LABEL,
    alignment_kwargs: dict | None = None,
) -> DocumentBuild:
    """Run one source document through the existing pipeline and align its summary.

    Stage order mirrors ``src.pipeline.process_document`` so the training data is
    built from exactly the same text the inference path would see:

        preprocess -> redact -> split_paragraphs -> detect_sections ->
        (group_by_section | fallback) -> chunk_document -> align
    """
    result = DocumentBuild(doc_id=doc_id, split=split)

    try:
        # 1-2. Cleaning and preamble stripping (existing implementation).
        clean = preprocess(text or "", strip_preamble_flag=strip_preamble)

        # 3. PII redaction (existing implementation). Pattern redaction always
        #    runs, which is the project's established behaviour; name redaction
        #    is opt-in.
        redacted, result.pii_counts = redact(clean, redact_names_flag=redact_names)

        # 4-5. Paragraph split and section detection (existing implementations).
        paragraphs = split_paragraphs(redacted)
        paragraph_objects = detect_sections(paragraphs)

        # 6. Fallback behaviour when the detector cannot resolve a structure.
        result.structure_fallback = fallback_to_single_section(paragraph_objects)
        if result.structure_fallback:
            section_groups = {fallback_label: paragraph_objects}
        else:
            section_groups = group_by_section(paragraph_objects)

        # 7. Section-aware chunking (existing implementation).
        chunks = chunk_document(section_groups)
        result.chunks_created = len(chunks)
        result.at_chunk_cap = len(chunks) >= MAX_CHUNKS_PER_DOC

        # 8-10. Split the reference summary into sentences and align it to the
        #       chunks with the shared Cross-Encoder (existing implementation).
        section_chunks = [(chunk.section, chunk.text) for chunk in chunks]
        alignment = align_summary_to_sections(
            section_chunks, summary or "", **(alignment_kwargs or {})
        )

        result.summary_sentences = alignment.total_sentences
        result.multi_label_sentences = sum(1 for r in alignment.per_sentence if len(r.assigned) > 1)
        result.assigned_labels = sum(len(r.assigned) for r in alignment.per_sentence)
        result.unassigned = [
            {"doc_id": doc_id, **entry} for entry in alignment.unassigned
        ]

        # The published IN-ABS summary, kept once per document for later
        # document-level evaluation.
        result.reference = {
            "doc_id": doc_id,
            "split": split,
            "original_reference_summary": summary or "",
        }

        # 11-12. Chunk-level pseudo-targets; chunks with no target are dropped.
        assigned_by_sentence = {r.sentence: r.assigned for r in alignment.per_sentence}

        for index, chunk in enumerate(chunks):
            matched = alignment.chunk_sentences.get(index) or []
            target = (alignment.aligned_by_chunk.get(index) or "").strip()

            # Both guards are needed: a chunk can be the best match for a
            # sentence yet receive no text, and vice versa.
            if not matched or not target:
                continue

            matched_labels = sorted({
                label
                for sentence in matched
                for label in assigned_by_sentence.get(sentence, [])
            })

            result.records.append({
                # --- required training fields -------------------------------- #
                "doc_id": doc_id,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "chunk_text": chunk.text,
                "aligned_target": target,
                "split": split,
                "para_range": list(chunk.para_range),
                # --- alignment provenance ----------------------------------- #
                "matched_sentences": matched,
                "alignment_scores": [round(s, 4) for s in alignment.chunk_pair_scores.get(index, [])],
                "matched_labels": matched_labels,
                "structure_fallback": result.structure_fallback,
                # --- length metadata (estimates use chunker.approximate_tokens) -- #
                "chunk_index": index,
                "n_chunk_tokens_est": chunk.token_count,
                "n_target_tokens_est": approximate_tokens(target),
            })

    except Exception as exc:  # pragma: no cover - defensive: one bad doc must not kill a 5k-doc build
        result.error = f"{type(exc).__name__}: {exc}"
        logger.error("Failed to build %s: %s", doc_id, result.error)

    return result


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, so the value is always an observed one."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return float(ordered[index])


def _length_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "mean": round(sum(values) / len(values), 1),
        "median": round(float(median), 1),
        "p90": round(_percentile(values, 90), 1),
        "max": round(float(max(values)), 1),
    }


def quality_flags(builds: list[DocumentBuild]) -> dict:
    """Diagnostics that make an unhealthy build impossible to miss.

    Not cosmetic: the chunker does not sentence-split a paragraph that is larger
    than ``CHUNK_SIZE`` (``chunker.split_by_tokens`` exists for that but is never
    called), so a judgment whose paragraphs did not survive blank-line splitting
    can collapse to a single oversized chunk. That chunk then collects every
    assigned summary sentence and its pseudo-target becomes the whole reference
    summary - longer than the model's input window and not trainable. These
    counts expose that instead of silently shipping it.
    """
    retained = [r for b in builds for r in b.records]
    return {
        "max_input_length_limit": MAX_INPUT_LENGTH,
        "records_with_target_over_model_input": sum(
            1 for r in retained if r["n_target_tokens_est"] > MAX_INPUT_LENGTH
        ),
        "records_with_target_longer_than_chunk": sum(
            1 for r in retained if r["n_target_tokens_est"] > r["n_chunk_tokens_est"]
        ),
        "documents_with_single_chunk": sum(1 for b in builds if b.chunks_created == 1),
    }


def verify_split_isolation(doc_ids_by_split: dict[str, set[str]], content_hash_by_split: dict[str, dict]) -> dict:
    """Prove no document - and no chunk of one - crosses a split boundary.

    Two independent checks:

    1. Document IDs are disjoint across splits. This is the hard invariant; a
       failure raises.
    2. Source-text content hashes are disjoint across splits. IN-ABS is built
       from published judgments, so the same judgment *can* legitimately appear
       in two official splits; that would be real leakage, so it is reported
       rather than hidden.
    """
    splits = list(doc_ids_by_split)
    overlapping: dict[str, list[str]] = {}
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            shared = doc_ids_by_split[left] & doc_ids_by_split[right]
            if shared:
                overlapping[f"{left}\u2229{right}"] = sorted(shared)

    if overlapping:
        raise SystemExit(
            "SPLIT ISOLATION FAILED - the same document id appears in more than one split: "
            f"{overlapping}"
        )

    duplicate_contents = []
    hashes = {split: set(values.values()) for split, values in content_hash_by_split.items()}
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            for digest in sorted(hashes[left] & hashes[right]):
                duplicate_contents.append({
                    "hash": digest[:12],
                    "splits": [left, right],
                    "doc_ids": sorted({
                        doc_id
                        for split in (left, right)
                        for doc_id, value in content_hash_by_split[split].items()
                        if value == digest
                    }),
                })

    if duplicate_contents:
        logger.warning(
            "Found %d source document(s) whose text appears in more than one official split. "
            "This is leakage inherited from %s, not introduced by the builder - report it. Example: %s",
            len(duplicate_contents),
            DATASET_ID,
            duplicate_contents[0],
        )

    return {
        "verified": True,
        "method": "official IN-ABS splits preserved; documents partitioned before chunking",
        "documents_per_split": {split: len(ids) for split, ids in doc_ids_by_split.items()},
        "doc_id_overlaps": overlapping,
        "cross_split_content_duplicates": len(duplicate_contents),
        "cross_split_content_duplicate_examples": duplicate_contents[:10],
    }


def build_stats(
    *,
    mode: str,
    backend: str,
    builds: list[DocumentBuild],
    records_by_split: dict[str, list[dict]],
    doc_ids_by_split: dict[str, set[str]],
    content_hash_by_split: dict[str, dict],
    split_isolation: dict,
    args: argparse.Namespace,
) -> dict:
    """Assemble ``build_stats.json``. No expected count is hard-coded anywhere."""
    chunks_created = sum(b.chunks_created for b in builds)
    chunks_retained = sum(len(b.records) for b in builds)

    target_tokens = [r["n_target_tokens_est"] for b in builds for r in b.records]
    chunk_tokens = [r["n_chunk_tokens_est"] for b in builds for r in b.records]
    pair_scores = [s for b in builds for r in b.records for s in r["alignment_scores"]]

    sentences_total = sum(b.summary_sentences for b in builds)
    sentences_multi = sum(b.multi_label_sentences for b in builds)
    labels_total = sum(b.assigned_labels for b in builds)
    unassigned_total = sum(len(b.unassigned) for b in builds)

    retained_by_section: Counter = Counter()
    for build in builds:
        retained_by_section.update(r["section"] for r in build.records)

    pii_counts: Counter = Counter()
    for build in builds:
        pii_counts.update(build.pii_counts)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "source_dataset": DATASET_ID,
        "dataset": {
            "documents_processed": len(builds),
            "documents_per_split": {split: len(ids) for split, ids in doc_ids_by_split.items()},
            "documents_failed": sum(1 for b in builds if b.error),
            "documents_structure_fallback": sum(1 for b in builds if b.structure_fallback),
            "documents_at_chunk_cap": sum(1 for b in builds if b.at_chunk_cap),
            "documents_without_records": sum(1 for b in builds if not b.records),
        },
        "chunks": {
            "created": chunks_created,
            "retained": chunks_retained,
            "dropped_no_aligned_target": chunks_created - chunks_retained,
            "retained_per_split": {split: len(recs) for split, recs in records_by_split.items()},
            "retained_by_section": dict(retained_by_section),
        },
        "summary_sentences": {
            "total": sentences_total,
            "aligned": sentences_total - unassigned_total,
            "unassigned": unassigned_total,
            "unassigned_rate": round(unassigned_total / sentences_total, 4) if sentences_total else 0.0,
            "multi_label": sentences_multi,
            "multi_label_rate": round(sentences_multi / sentences_total, 4) if sentences_total else 0.0,
            "labels_per_sentence": round(labels_total / sentences_total, 4) if sentences_total else 0.0,
        },
        "alignment": {
            "cross_encoder_model": CROSS_ENCODER_MODEL,
            "backend": backend,
            "threshold": args.threshold,
            "relative_margin": args.relative_margin,
            "allow_multi_label": not args.no_multi_label,
            "score_stats": {
                "count": len(pair_scores),
                "mean": round(sum(pair_scores) / len(pair_scores), 4) if pair_scores else 0.0,
                "median": round(float(_percentile(pair_scores, 50)), 4),
                "p90": round(_percentile(pair_scores, 90), 4),
                "max": round(max(pair_scores), 4) if pair_scores else 0.0,
            },
        },
        "preprocessing": {
            "strip_preamble": not args.no_strip_preamble,
            "redact_names": args.redact_names,
            "fallback_section_label": FALLBACK_SECTION_LABEL,
            "pii_pattern_counts": dict(pii_counts),
            "chunk_size": None,  # recorded by config; see config.CHUNK_SIZE
            "max_chunks_per_doc": MAX_CHUNKS_PER_DOC,
        },
        "lengths_estimated_with_approximate_tokens": {
            "aligned_target": _length_stats(target_tokens),
            "chunk_text": _length_stats(chunk_tokens),
        },
        "quality_flags": quality_flags(builds),
        "split_isolation": split_isolation,
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows: list[dict]) -> bool:
    """Write Parquet alongside the JSONL when pyarrow is available."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    try:
        pq.write_table(pa.Table.from_pylist(rows), path)
        return True
    except Exception as exc:  # pragma: no cover - schema dependent
        logger.warning("Could not write %s: %s", path.name, exc)
        return False


def write_inspection_sample(path: Path, records: list[dict], limit: int = 5) -> None:
    """Human-readable side-by-side dump so a target can be eyeballed against its source.

    The point of this file is to make it possible to check that ``aligned_target``
    is semantically *about* ``chunk_text``. A non-empty target proves nothing.
    """
    lines = [
        "INSPECTION SAMPLE",
        "Check that each aligned_target is semantically about the chunk_text above it.",
        "A non-empty target is not evidence by itself.",
        "=" * 100,
        "",
    ]
    for record in records[:limit]:
        lines += [
            f"doc_id           : {record['doc_id']}",
            f"chunk_id         : {record['chunk_id']}  (index {record['chunk_index']})",
            f"split            : {record['split']}",
            f"section          : {record['section']}"
            f"{'   [structure fallback]' if record['structure_fallback'] else ''}",
            f"para_range       : {record['para_range']}",
            f"alignment_scores : {record['alignment_scores']}",
            f"matched_labels   : {record['matched_labels']}",
            f"chunk tokens~    : {record['n_chunk_tokens_est']}",
            f"target tokens~   : {record['n_target_tokens_est']}",
            "",
            "--- CHUNK_TEXT (source input) " + "-" * 70,
            record["chunk_text"],
            "",
            "--- ALIGNED_TARGET (pseudo-target) " + "-" * 66,
            record["aligned_target"],
            "",
        ]
        for i, sentence in enumerate(record["matched_sentences"], 1):
            lines.append(f"    [{i}] {sentence}")
        lines += ["=" * 100, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_card(path: Path, stats: dict, records_by_split: dict[str, list[dict]]) -> None:
    """Dataset documentation that states the provenance and the caveats."""
    counts = ", ".join(f"`{split}`: {len(recs)}" for split, recs in records_by_split.items())
    alignment = stats["alignment"]
    card = f"""---
language:
- en
task_categories:
- summarization
source_datasets:
- {DATASET_ID}
---

# IN-ABS section-aligned (chunk-level training targets)

Chunk-level supervised training data derived from
[{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID}), for fine-tuning a
BART summarizer that produces structured, section-aware judgments summaries.

## Provenance and what the fields mean

* Each example is **one chunk** of a judgment.
* `aligned_target` is a **pseudo-target**, not ground truth. It is generated by
  decomposing the document-level IN-ABS reference summary into sentences and
  matching each sentence to the chunk(s) it most plausibly came from, using a
  Cross-Encoder.
* **The original IN-ABS summary remains the document-level evaluation
  reference.** It is not the training target, and it is not duplicated into
  every chunk record.
* `chunk_text` is the source input to the model; `aligned_target` is what the
  model is trained to produce.

## Build methodology

* Judgement text: the existing inference preprocessing pipeline is reused
  (`preprocess` -> PII redaction -> paragraph split -> `detect_sections` ->
  `chunk_document`).
* Reference summary: split into sentences, then each sentence is scored against
  every chunk with the shared Cross-Encoder; a section is scored by its best
  chunk, because the Cross-Encoder truncates its input at 512 tokens and long
  sections would otherwise be systematically under-matched.
* A summary sentence is assigned to every section scoring at or above the
  absolute threshold **and** within the relative margin of its best section
  (multi-label), and is attached to that section's single best-matching chunk.
* Sentences clearing no threshold are **left unassigned** rather than pushed
  into a nearest neighbour.
* Chunks with no aligned target are **excluded** from the supervised dataset.

| Setting | Value |
| --- | --- |
| Alignment threshold | `{alignment['threshold']}` |
| Relative margin | `{alignment['relative_margin']}` |
| Multi-label | `{alignment['allow_multi_label']}` |
| Cross-Encoder | `{alignment['cross_encoder_model']}` |
| Alignment backend | `{alignment['backend']}` |
| Source documents | `{stats['dataset']['documents_processed']}` |
| Records | {counts} |
| Chunks dropped (no target) | `{stats['chunks']['dropped_no_aligned_target']}` |
| Reference-summary sentences unassigned | `{stats['summary_sentences']['unassigned']}` ({stats['summary_sentences']['unassigned_rate']:.1%}) |

## Leakage control

**Documents are split before chunking.** The official IN-ABS `train` /
`validation` / `test` splits are preserved unchanged; sampling selects whole
documents only, so every chunk of a document lives in exactly one split and a
document never appears in two. The builder asserts document-id disjointness on
every run and records the result in `build_stats.json` under `split_isolation`.

Known caveat: if the same published judgment appears in two official IN-ABS
splits, that leak is inherited from the source dataset. The builder detects and
reports it (`split_isolation.cross_split_content_duplicates`) rather than hiding
it. This build reported
`{stats['split_isolation']['cross_split_content_duplicates']}` such document(s).

## Not trained here

No BART model has been fine-tuned as part of producing this dataset. It contains
training data only. `aligned_target` values are alignment-derived pseudo-targets
and should not be reported as model quality.

Licence: inherited from `{DATASET_ID}` - check the source dataset card.
"""
    path.write_text(card, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Build driver
# --------------------------------------------------------------------------- #

def run_build(args: argparse.Namespace) -> tuple[dict[str, list[dict]], dict, list[DocumentBuild]]:
    """Process documents and return ``(records_by_split, stats, builds)``."""
    backend = require_real_encoder(args.allow_stub_encoder)
    logger.info("Alignment backend: %s", backend)

    splits = args.split or list(OFFICIAL_SPLITS)
    limit = args.sample  # --sample N == N complete documents per selected split

    # Overrides are resolved from config once, then applied to every alignment
    # so build_stats.json reports exactly what was used.
    alignment_kwargs: dict = {}
    if args.threshold is not None:
        alignment_kwargs["threshold"] = args.threshold
    if args.relative_margin is not None:
        alignment_kwargs["relative_margin"] = args.relative_margin
    if args.no_multi_label:
        alignment_kwargs["allow_multi_label"] = False

    builds: list[DocumentBuild] = []
    records_by_split: dict[str, list[dict]] = {}
    doc_ids_by_split: dict[str, set[str]] = {}
    content_hash_by_split: dict[str, dict] = {}
    all_unassigned: list[dict] = []
    references: list[dict] = []

    for split in splits:
        dataset = load_split(split, limit=limit)
        text_col, summary_col, file_col = resolve_columns(dataset)
        logger.info(
            "%s: %d document(s) [text=%r summary=%r file=%r]",
            split, len(dataset), text_col, summary_col, file_col,
        )

        records_by_split[split] = []
        doc_ids_by_split[split] = set()
        content_hash_by_split[split] = {}

        for index, row in enumerate(dataset):
            # Document ids are <split>_<position in that split>, so a document
            # is addressable and provably single-split.
            doc_id = f"{split}_{index:05d}"
            doc_ids_by_split[split].add(doc_id)

            text = row.get(text_col) or ""
            summary = row.get(summary_col) or ""
            content_hash_by_split[split][doc_id] = hashlib.sha1(
                text.encode("utf-8", errors="ignore")
            ).hexdigest()

            build = build_document(
                doc_id,
                split,
                text,
                summary,
                redact_names=args.redact_names,
                strip_preamble=not args.no_strip_preamble,
                alignment_kwargs=alignment_kwargs,
            )
            builds.append(build)
            records_by_split[split].extend(build.records)
            all_unassigned.extend(build.unassigned)
            if build.reference is not None:
                references.append(build.reference)

            if (index + 1) % 25 == 0:
                logger.info(
                    "%s: %d/%d documents, %d records so far",
                    split, index + 1, len(dataset), len(records_by_split[split]),
                )

    split_isolation = verify_split_isolation(doc_ids_by_split, content_hash_by_split)
    stats = build_stats(
        mode=args.mode,
        backend=backend,
        builds=builds,
        records_by_split=records_by_split,
        doc_ids_by_split=doc_ids_by_split,
        content_hash_by_split=content_hash_by_split,
        split_isolation=split_isolation,
        args=args,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_written = []
    for split, records in records_by_split.items():
        if not records:
            logger.warning("Split '%s' produced no records - nothing written for it.", split)
            continue
        write_jsonl(output_dir / f"{split}.jsonl", records)
        if write_parquet(output_dir / f"{split}.parquet", records):
            parquet_written.append(split)

    write_jsonl(output_dir / "reference_summaries.jsonl", references)
    write_jsonl(output_dir / "unassigned.jsonl", all_unassigned)

    stats["outputs"] = {
        "directory": str(output_dir),
        "splits": {split: f"{split}.jsonl" for split, recs in records_by_split.items() if recs},
        "parquet_splits": parquet_written,
        "reference_summaries": "reference_summaries.jsonl",
        "unassigned_report": "unassigned.jsonl",
        "inspection_sample": "inspect_sample.txt" if args.sample else None,
        "dataset_card": "README.md",
    }
    (output_dir / "build_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    inspection_records = []
    for split in records_by_split:
        inspection_records.extend(records_by_split[split])
    if args.sample:
        write_inspection_sample(output_dir / "inspect_sample.txt", inspection_records)

    write_dataset_card(output_dir / "README.md", stats, records_by_split)

    return records_by_split, stats, builds


def report(stats: dict, output_dir: Path) -> None:
    dataset = stats["dataset"]
    chunks = stats["chunks"]
    sentences = stats["summary_sentences"]
    target = stats["lengths_estimated_with_approximate_tokens"]["aligned_target"]

    lines = [
        "",
        "=" * 72,
        "BUILD SUMMARY",
        "=" * 72,
        f"  documents processed    : {dataset['documents_processed']}"
        f"  {dataset['documents_per_split']}",
        f"  structure fallbacks    : {dataset['documents_structure_fallback']}",
        f"  documents at chunk cap : {dataset['documents_at_chunk_cap']} (cap = {MAX_CHUNKS_PER_DOC})",
        f"  documents with 0 records: {dataset['documents_without_records']}",
        f"  chunks created         : {chunks['created']}",
        f"  chunks retained        : {chunks['retained']}"
        f"   ({chunks['dropped_no_aligned_target']} dropped: no aligned target)",
        f"  records per split      : {chunks['retained_per_split']}",
        f"  records by section     : {chunks['retained_by_section']}",
        f"  summary sentences      : {sentences['total']}"
        f"  ({sentences['aligned']} aligned / {sentences['unassigned']} unassigned"
        f" = {sentences['unassigned_rate']:.1%})",
        f"  multi-label rate       : {sentences['multi_label_rate']:.1%}"
        f"  (labels per sentence {sentences['labels_per_sentence']})",
        f"  aligned target tokens~ : mean {target['mean']} median {target['median']}"
        f" p90 {target['p90']} max {target['max']}",
        f"  records w/ target over {stats['quality_flags']['max_input_length_limit']} tok: "
        f"{stats['quality_flags']['records_with_target_over_model_input']}"
        f"  (single-chunk docs: {stats['quality_flags']['documents_with_single_chunk']})",
        f"  split isolation        : {'VERIFIED' if stats['split_isolation']['verified'] else 'FAILED'}"
        f"  (cross-split duplicate text: {stats['split_isolation']['cross_split_content_duplicates']})",
        f"  alignment backend      : {stats['alignment']['backend']}",
        f"  output                 : {output_dir}",
        "=" * 72,
    ]
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Hugging Face upload
# --------------------------------------------------------------------------- #

def resolve_hf_token() -> str | None:
    """Token from the environment only - never hard-coded, never on disk here."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def require_hf_auth(token: str | None) -> str:
    """Fail early, with instructions, when the Hub credentials are missing or bad."""
    try:
        from huggingface_hub import whoami
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"huggingface_hub is required to upload: {exc}") from exc

    try:
        identity = whoami(token=token)
    except Exception as exc:
        raise SystemExit(
            "Not authenticated with the Hugging Face Hub, so the dataset cannot be uploaded.\n\n"
            "Do one of the following:\n"
            "  1. Log in once (recommended); the cached token is then reused:\n"
            "         hf auth login            # huggingface_hub >= 0.23\n"
            "         huggingface-cli login    # older installs\n"
            "  2. Or export a token for this shell / Colab secret:\n"
            "         export HF_TOKEN=hf_xxx\n\n"
            "Tokens are read from the environment or the Hugging Face login cache;\n"
            "they are never written into this repository.\n"
            f"(error was: {exc})"
        ) from exc

    name = identity.get("name") if isinstance(identity, dict) else getattr(identity, "name", None)
    logger.info("Authenticated with the Hub as '%s'", name)
    return name or "unknown"


def push_to_hub(args: argparse.Namespace, stats: dict | None) -> None:
    """Validate locally, then upload the splits and the dataset card."""
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise SystemExit(
            f"Nothing to upload: {output_dir} does not exist. Build the dataset first "
            "(e.g. --full --output_dir ...) or point --output_dir at an existing build."
        )

    if stats is None:
        stats_path = output_dir / "build_stats.json"
        if not stats_path.exists():
            raise SystemExit(f"Missing {stats_path}; cannot verify the build before uploading.")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    # Hard gate: a stub-encoder build must never reach the Hub.
    backend = stats.get("alignment", {}).get("backend")
    if backend == "stub":
        raise SystemExit(
            "Refusing to upload: this build used the mock Cross-Encoder stub, so its "
            "aligned_target values are meaningless. Rebuild with sentence-transformers installed."
        )

    require_hf_auth(resolve_hf_token())

    split_files = {
        split: output_dir / f"{split}.jsonl"
        for split in OFFICIAL_SPLITS
        if (output_dir / f"{split}.jsonl").exists()
    }
    if not split_files:
        raise SystemExit(f"No split JSONL files found in {output_dir}.")

    datasets = _import_datasets()

    # Load locally first: this is the same code path a consumer will use, and it
    # fails here rather than after a partial upload.
    dataset_dict = datasets.DatasetDict({
        split: datasets.Dataset.from_json(str(path)) for split, path in split_files.items()
    })
    logger.info("Loaded %s locally", {s: len(d) for s, d in dataset_dict.items()})

    dataset_dict.push_to_hub(
        args.repo_id,
        token=resolve_hf_token(),
        commit_message="Upload section-aligned IN-ABS chunk-level training dataset",
    )
    logger.info("Pushed splits to https://huggingface.co/datasets/%s", args.repo_id)

    card = output_dir / "README.md"
    if card.exists():
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=resolve_hf_token(),
            commit_message="Add dataset card",
        )
        logger.info("Uploaded dataset card")

    # Read back what the Hub now serves, so the upload is confirmed rather than assumed.
    try:
        remote = datasets.load_dataset(args.repo_id, token=resolve_hf_token())
        logger.info(
            "Verified remote dataset: %s",
            {split: len(remote[split]) for split in remote},
        )
        logger.info("Remote columns: %s", remote[list(remote)[0]].column_names)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Uploaded, but could not read the dataset back for verification: %s", exc)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build (and optionally upload) the section-aligned IN-ABS training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  --sample 10  --output_dir artifacts/training_data/sample10\n"
            "  --sample 100 --output_dir artifacts/training_data/val100\n"
            "  --full       --output_dir artifacts/training_data/full\n"
            "  --push-to-hub --repo-id <user>/in-abs-section-aligned --output_dir artifacts/training_data/full\n\n"
            "--sample N selects N COMPLETE documents per selected split (first N in the\n"
            "official split order, so builds are reproducible). It never means N chunks,\n"
            "N summary sentences or N training records.\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sample", type=int, metavar="N",
        help="Process the first N complete documents of each selected split.",
    )
    mode.add_argument(
        "--full", action="store_true",
        help="Process every document in the selected splits.",
    )

    parser.add_argument("--output_dir", required=True, help="Where to write the dataset.")
    parser.add_argument(
        "--split", action="append", choices=list(OFFICIAL_SPLITS), default=None,
        help="Restrict to a split (repeatable). Defaults to all three official splits.",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Upload to the Hub. Without --sample/--full, uploads an existing build from --output_dir.",
    )
    parser.add_argument("--repo-id", default=None, help="Target Hub repo, e.g. <user>/in-abs-section-aligned.")
    parser.add_argument("--redact-names", action="store_true", help="Also redact PERSON entities (spaCy).")
    parser.add_argument("--no-strip-preamble", action="store_true", help="Keep the case-caption preamble.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the alignment threshold.")
    parser.add_argument("--relative-margin", type=float, default=None, help="Override the relative margin.")
    parser.add_argument("--no-multi-label", action="store_true", help="Force a single label per sentence.")
    parser.add_argument(
        "--allow-stub-encoder", action="store_true",
        help="Allow the meaningless mock encoder (plumbing tests only; blocks --push-to-hub).",
    )

    args = parser.parse_args(argv)

    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be a positive number of documents.")
    if not args.sample and not args.full and not args.push_to_hub:
        parser.error("Choose one of --sample N, --full, or --push-to-hub.")
    if args.push_to_hub and not args.repo_id:
        parser.error("--push-to-hub requires --repo-id, e.g. --repo-id <user>/in-abs-section-aligned.")

    if args.sample:
        args.mode = f"sample{args.sample}"
    elif args.full:
        args.mode = "full"
    else:
        args.mode = "push-only"

    # Record the settings the alignment will actually use. Resolved through the
    # alignment module's own accessor so the reported values cannot drift from
    # the values the aligner applies.
    from src.data.section_summary_alignment import alignment_settings
    cfg_threshold, _cfg_multi_label, cfg_margin = alignment_settings()
    if args.threshold is None:
        args.threshold = cfg_threshold
    if args.relative_margin is None:
        args.relative_margin = cfg_margin

    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    stats = None
    if args.sample or args.full:
        _, stats, _builds = run_build(args)
        report(stats, Path(args.output_dir))

    if args.push_to_hub:
        push_to_hub(args, stats)
        print(f"\nUploaded to https://huggingface.co/datasets/{args.repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Section-summary alignment: turn an IN-ABS reference summary into chunk-level pseudo-targets.

IN-ABS gives one whole-document reference summary per judgment and no per-section
ground truth. Training every chunk against that whole summary would teach the
model to produce content its input never contained, so the reference summary is
decomposed instead: each summary sentence is matched, by the shared
Cross-Encoder, against the chunks it plausibly came from.

Three objects must not be confused, and this module only ever produces the
middle one:

    ``original_reference_summary``
        the IN-ABS summary as published - **document-level evaluation ground
        truth**. Never modified, never chunked by position, never used as a
        per-chunk training target.
    ``aligned_target`` (``AlignmentResult.aligned_by_chunk``)
        Cross-Encoder-derived **chunk-level pseudo-target**, used only as a
        supervised training target for per-chunk summarization.
    ``chunk_text``
        the source input.

Two properties of the alignment, both deliberate and both reported as metrics:

  * **Multi-label** - a summary sentence may belong to more than one section.
    Headnotes routinely fuse a fact and a holding into one sentence; forcing a
    single winner throws away half of it. Multi-label is governed by *two*
    tests, not one: an absolute threshold, and a relative margin. On its own the
    absolute threshold turns out to assign ~80% of sentences to nearly every
    section, because all sections of a judgment share vocabulary and the
    Cross-Encoder rates them all plausible - which is the mirror image of the
    over-forcing the plan set out to avoid. The relative margin keeps a
    secondary section only when it scores close to the best one, so genuinely
    fused sentences stay multi-label and merely-similar ones do not.
  * **Conservative** - a sentence that clears the threshold for no section is
    left unassigned rather than pushed into its nearest neighbour. The
    unassigned rate is tracked and reported: it is a measure of how much of the
    reference summary this pipeline can honestly account for, and a high rate
    is a finding, not a bug to hide.

Implementation note that differs from the original plan's sketch, deliberately:
the plan scores a summary sentence against ``" ".join(all chunks of the
section)``. The Cross-Encoder truncates its input at 512 tokens, so for any
section longer than that - which is most of them - everything past the first
~512 tokens would be silently invisible to the comparison, and long sections
would be systematically under-matched. Here each sentence is scored against
each *chunk* and the section takes the best of its chunks. Same idea, actually
applied to the whole section.

This module reuses, and does not reimplement:
  * ``config.ALIGNMENT_THRESHOLD`` / ``ALIGNMENT_RELATIVE_MARGIN`` - the
    validated configuration produced by the threshold sweep in
    ``artifacts/alignment_sweep.json`` (0.40 / 0.90).
  * ``src.shared.cross_encoder.get_cross_encoder`` - the single shared
    Cross-Encoder singleton (``cross-encoder/ms-marco-MiniLM-L-6-v2``).

It does **not** use ``src.postprocess.cite_verify.split_sentences``: that helper
quietly drops sentences with fewer than three words, which would discard terse
operative orders. ``cite_verify`` itself is untouched; this module keeps its own
non-empty-sentence splitter (``summary_sentences``).

Each sentence has at most **one primary chunk** - the single highest-scoring
chunk among its plausible sections - so a sentence is never duplicated across
training records. ``SentenceAlignment.assigned`` still records every plausible
section label as metadata.

The pseudo-targets produced here are a training aid, never an evaluation
target. Final evaluation runs against the true IN-ABS summary.
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, ".")

import config
from src.data import evidence_alignment as evidence_layer
from src.data.chunker import approximate_tokens
from src.shared.cross_encoder import get_cross_encoder

logger = logging.getLogger(__name__)

# Every non-empty reference-summary sentence is retained. The previous behaviour
# dropped sentences shorter than 25 characters, which silently removed terse
# operative orders ("The appeal is dismissed.") - exactly the sentences that
# carry the judgement signal. Conservative rejection is now left to the
# Cross-Encoder threshold instead of a length filter.
#
# ``cite_verify.split_sentences`` is deliberately NOT used here: it also drops
# sentences with fewer than three words, so a 25-char filter was not the only
# thing discarding short orders ("Appeal dismissed." is two words).
# ``cite_verify.split_sentences`` is left unchanged because cite-and-verify
# depends on its existing behaviour.
def summary_sentences(full_summary: str) -> list[str]:
    """Split a reference summary into sentences, keeping every non-empty one."""
    if not full_summary:
        return []
    parts = re.split(r"(?<=[.!?])\s+", full_summary)
    return [part.strip() for part in parts if part.strip()]

# The historical config module carried these alongside ALIGNMENT_THRESHOLD and
# ALIGNMENT_RELATIVE_MARGIN. They are read from config when present so the
# validated values are always the single source of truth, and fall back to the
# documented defaults otherwise. config.py is intentionally left untouched, so
# the two knobs it no longer declares are supplied here.
_DEFAULT_THRESHOLD = 0.40
_DEFAULT_RELATIVE_MARGIN = 0.90
_ALLOW_MULTI_LABEL_DEFAULT = True
_UNASSIGNED_WARN_RATE_DEFAULT = 0.25


def _config_value(name: str, default):
    """Read an alignment knob from config, falling back to the documented default."""
    return getattr(config, name, default)


def evidence_settings_from_config():
    """Resolve the evidence-layer knobs from config, with documented fallbacks.

    The fallbacks are the module's own (uncalibrated) placeholders, so a missing
    config entry cannot silently change behaviour - and ``build_stats.json``
    records whichever values were actually used.
    """
    defaults = evidence_layer.EvidenceSettings()
    return evidence_layer.EvidenceSettings(
        coverage_bound=_config_value("ALIGNMENT_EVIDENCE_COVERAGE_BOUND", defaults.coverage_bound),
        ambiguity_margin=_config_value(
            "ALIGNMENT_EVIDENCE_AMBIGUITY_MARGIN", defaults.ambiguity_margin
        ),
        min_common_span=_config_value("ALIGNMENT_EVIDENCE_MIN_SPAN", defaults.min_common_span),
        max_rare_df=_config_value("ALIGNMENT_EVIDENCE_MAX_RARE_DF", defaults.max_rare_df),
        contradiction_mode=_config_value(
            "ALIGNMENT_EVIDENCE_CONTRADICTION_MODE", defaults.contradiction_mode
        ),
    )


def alignment_settings() -> tuple[float, bool, float]:
    """Resolve the validated alignment configuration.

    Returns ``(threshold, allow_multi_label, relative_margin)``. This is the
    single source of truth for the settings, so a caller that needs to *report*
    them (e.g. ``scripts/build_dataset.py`` recording them in build_stats.json)
    cannot drift from the values the aligner actually uses.
    """
    return (
        _config_value("ALIGNMENT_THRESHOLD", _DEFAULT_THRESHOLD),
        _config_value("ALIGNMENT_ALLOW_MULTI_LABEL", _ALLOW_MULTI_LABEL_DEFAULT),
        _config_value("ALIGNMENT_RELATIVE_MARGIN", _DEFAULT_RELATIVE_MARGIN),
    )


@dataclass
class SentenceAlignment:
    """One reference-summary sentence and the sections it was matched against."""

    sentence: str
    scores: dict[str, float]
    # Plausible section labels (metadata / multi-label statistics).
    assigned: list[str] = field(default_factory=list)
    # The single chunk this sentence is actually written to, if any. A sentence
    # is never copied into one chunk per assigned section.
    primary_chunk: int | None = None
    primary_chunk_score: float = 0.0
    # --- evidence layer (only populated when evidence mode is enabled) -------- #
    # One of SUPPORTED / AMBIGUOUS / UNSUPPORTED / NOT_SUMMARY_CONTENT, or None
    # when the sentence never reached the evidence layer (below the CE threshold).
    decision: str | None = None
    evidence: dict | None = None
    evidence_reason: str | None = None
    ambiguity: dict | None = None

    @property
    def best_label(self) -> str | None:
        """Section with the highest score, or None when nothing was scored."""
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.get)

    @property
    def best_score(self) -> float:
        """Highest section score, or 0.0 when nothing was scored."""
        if not self.scores:
            return 0.0
        return max(self.scores.values())


@dataclass
class AlignmentResult:
    """Outcome of aligning one document's reference summary to its chunks.

    ``aligned`` and ``aligned_sentences`` are keyed by section label.
    ``aligned_by_chunk`` / ``chunk_sentences`` / ``chunk_pair_scores`` are keyed
    by the chunk's index within ``section_chunks`` (which is the same index
    space as the document's chunk list).
    """

    aligned: dict[str, str] = field(default_factory=dict)
    aligned_sentences: dict[str, list[str]] = field(default_factory=dict)
    aligned_by_chunk: dict[int, str] = field(default_factory=dict)
    chunk_sentences: dict[int, list[str]] = field(default_factory=dict)
    chunk_pair_scores: dict[int, list[float]] = field(default_factory=dict)
    unassigned: list[dict] = field(default_factory=list)
    per_sentence: list[SentenceAlignment] = field(default_factory=list)
    threshold: float = 0.0
    multi_label: bool = True
    # --- evidence layer ------------------------------------------------------ #
    # Ambiguous sentences (evidence split across distant chunks, or a genuine
    # tie). Never written to a training record.
    ambiguous: list[dict] = field(default_factory=list)
    # Counts per evidence decision, plus "below_threshold" for sentences the
    # Cross-Encoder itself refused.
    decision_counts: dict = field(default_factory=dict)
    evidence_mode: bool = False
    evidence_settings: dict = field(default_factory=dict)

    @property
    def total_sentences(self) -> int:
        return len(self.per_sentence)

    @property
    def unassigned_rate(self) -> float:
        """Share of reference-summary sentences that no section claimed."""
        if not self.total_sentences:
            return 0.0
        return len(self.unassigned) / self.total_sentences

    @property
    def multi_label_rate(self) -> float:
        """Share of sentences assigned to more than one section."""
        if not self.total_sentences:
            return 0.0
        return sum(1 for r in self.per_sentence if len(r.assigned) > 1) / self.total_sentences

    @property
    def labels_per_sentence(self) -> float:
        """Mean number of assigned sections per sentence."""
        if not self.total_sentences:
            return 0.0
        return sum(len(r.assigned) for r in self.per_sentence) / self.total_sentences

    def summary(self) -> str:
        per_label = " | ".join(f"{label}:{len(sents)}" for label, sents in self.aligned_sentences.items())
        return (
            f"{per_label} | {self.total_sentences} summary sentences"
            f" | unassigned {len(self.unassigned)} ({self.unassigned_rate:.0%})"
            f" | multi-label {self.multi_label_rate:.0%}"
        )


def _score_pairs(pairs: list[list[str]]) -> list[float]:
    """Score (sentence, chunk) pairs with the shared Cross-Encoder.

    ``CrossEncoderSingleton`` exposes ``score`` (this replaced the older
    ``predict`` delegate). ``normalize=True`` applies the singleton's sigmoid,
    which is the space the validated thresholds live in - the recorded
    unassigned scores in
    ``artifacts/training_data/validation_unassigned_log.jsonl`` (0.03, 0.0086,
    ...) are all in [0, 1], and a 0.40 / 0.90 threshold pair is only meaningful
    there.
    """
    return get_cross_encoder().score(pairs, normalize=True)


def score_summary_sentences(
    section_chunks: list[tuple[str, str]], full_summary: str
) -> tuple[list[str], list[dict[str, float]], list[list[float]]]:
    """Score every summary sentence against every chunk.

    Returns ``(sentences, per_label_scores, per_chunk_scores)`` where
    ``per_label_scores[i][label]`` is the best score sentence *i* achieved
    against any chunk carrying that label, and ``per_chunk_scores[i][j]`` is the
    raw score for sentence *i* against ``section_chunks[j]``.

    All sentence x chunk pairs are scored in a single batch: the per-pair
    call was the dominant cost of the alignment pass.
    """
    sentences = summary_sentences(full_summary or "")
    if not sentences or not section_chunks:
        return sentences, [], []

    pairs = [[sentence, chunk] for sentence in sentences for _, chunk in section_chunks]
    flat_scores = _score_pairs(pairs)

    n_chunks = len(section_chunks)
    per_label_scores: list[dict[str, float]] = []
    per_chunk_scores: list[list[float]] = []

    for i in range(len(sentences)):
        row = [float(s) for s in flat_scores[i * n_chunks:(i + 1) * n_chunks]]

        # A section is represented by several chunks; score it by its best one
        # so that long sections are not penalised for having no single chunk
        # that covers the whole sentence.
        best: dict[str, float] = {}
        for (label, _), score in zip(section_chunks, row):
            if score > best.get(label, -1.0):
                best[label] = score

        per_label_scores.append(best)
        per_chunk_scores.append(row)

    return sentences, per_label_scores, per_chunk_scores


def align_summary_to_sections(
    section_chunks: list[tuple[str, str]],
    full_summary: str,
    threshold: float | None = None,
    allow_multi_label: bool | None = None,
    relative_margin: float | None = None,
    precomputed: tuple[list[str], list[dict[str, float]], list[list[float]]] | None = None,
    evidence_mode: bool | None = None,
    evidence_settings=None,
    chunk_positions: list[float] | None = None,
) -> AlignmentResult:
    """Align each sentence of ``full_summary`` to section label(s) and to a chunk.

    ``section_chunks`` is ``[(section_label, chunk_text), ...]`` - one entry per
    chunk, in the document's chunk order.

    A sentence is assigned to every section that (a) scores at or above
    ``threshold`` and (b) is within ``relative_margin`` of the sentence's best
    section. A sentence that satisfies neither is left unassigned. Within each
    assigned section the sentence is attached to that section's best-matching
    chunk, which is what makes a chunk-level target possible at all.

    ``precomputed`` allows reusing an existing ``score_summary_sentences``
    result instead of paying for the Cross-Encoder pass twice.

    ``evidence_mode`` (default OFF, from ``config.ALIGNMENT_EVIDENCE_MODE``) adds
    the evidence layer between the existing candidate selection and the primary
    chunk. With it off, this function behaves exactly as before. With it on, the
    plausible candidate chunks are passed through ``evidence_alignment``: only a
    SUPPORTED sentence is written to a chunk, and AMBIGUOUS / UNSUPPORTED /
    NOT_SUMMARY_CONTENT sentences are attached nowhere and reported instead.
    ``chunk_positions`` are optional document positions (0..1) derived from
    ``Chunk.para_range`` midpoints, used only as the evidence layer's soft final
    tie-break - never as a hard rule.
    """
    if evidence_mode is None:
        evidence_mode = bool(_config_value("ALIGNMENT_EVIDENCE_MODE", False))
    if evidence_settings is None:
        evidence_settings = evidence_settings_from_config()
    if threshold is None or allow_multi_label is None or relative_margin is None:
        cfg_threshold, cfg_multi_label, cfg_margin = alignment_settings()
        if threshold is None:
            threshold = cfg_threshold
        if allow_multi_label is None:
            allow_multi_label = cfg_multi_label
        if relative_margin is None:
            relative_margin = cfg_margin

    if precomputed is not None:
        sentences, all_scores, chunk_scores = precomputed
    else:
        sentences, all_scores, chunk_scores = score_summary_sentences(section_chunks, full_summary)

    labels = sorted({label for label, _ in section_chunks})

    if not sentences or not labels:
        return AlignmentResult(
            aligned={},
            aligned_sentences={},
            threshold=threshold,
            multi_label=allow_multi_label,
        )

    aligned_sentences: dict[str, list[str]] = {label: [] for label in labels}
    chunk_sentences: dict[int, list[str]] = {}
    chunk_pair_scores: dict[int, list[float]] = {}
    unassigned: list[dict] = []
    ambiguous: list[dict] = []
    per_sentence: list[SentenceAlignment] = []
    decision_counts: dict = {}

    # Document-local rarity for Tier-2 evidence: how many chunks of THIS document
    # contain each content word. Computed once, only when evidence mode is on.
    document_df = (
        evidence_layer.document_frequency([text for _, text in section_chunks])
        if evidence_mode else None
    )

    for sentence, per_label, row in zip(sentences, all_scores, chunk_scores):
        record = SentenceAlignment(sentence=sentence, scores=per_label)
        above = [label for label, score in per_label.items() if score >= threshold]

        if not above:
            # Conservative: no nearest-neighbour fallback. This sentence is
            # simply not part of the supervised dataset.
            unassigned.append({
                "sentence": sentence,
                "best_score": record.best_score,
                "best_label": record.best_label,
                "reason": "below_threshold",
            })
        elif allow_multi_label:
            cutoff = record.best_score * relative_margin
            record.assigned = [label for label in above if per_label[label] >= cutoff]
        else:
            record.assigned = [max(above, key=lambda lbl: per_label[lbl])]

        candidate_indices = [
            j for j, (label, _) in enumerate(section_chunks) if label in record.assigned
        ]

        # Attach at most ONE chunk. In evidence mode the evidence layer decides
        # whether this sentence is supported by any candidate at all; in the
        # default mode the previous highest-CE-chunk rule is kept verbatim.
        attach_index: int | None = None
        if candidate_indices:
            if not evidence_mode:
                attach_index = max(candidate_indices, key=lambda j: row[j])
            else:
                candidates = [
                    evidence_layer.Candidate(
                        index=j,
                        label=section_chunks[j][0],
                        text=section_chunks[j][1],
                        ce_score=float(row[j]),
                        position=(chunk_positions[j] if chunk_positions else None),
                    )
                    for j in candidate_indices
                ]
                decision = evidence_layer.decide(
                    sentence,
                    candidates,
                    settings=evidence_settings,
                    document_frequency=document_df,
                )
                record.decision = decision.decision
                record.evidence_reason = decision.reason
                record.ambiguity = decision.ambiguity
                record.evidence = {
                    "sentence": (
                        decision.sentence_evidence.as_dict()
                        if decision.sentence_evidence is not None else None
                    ),
                    "candidates": [c.as_dict() for c in decision.candidates],
                }
                decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1

                if decision.decision == evidence_layer.DECISION_AMBIGUOUS:
                    ambiguous.append({
                        "sentence": sentence,
                        "reason": decision.reason,
                        "assigned_labels": list(record.assigned),
                        "ambiguity": decision.ambiguity,
                        "evidence": record.evidence,
                        "candidates": [
                            {
                                "chunk_index": c.index,
                                "section": c.label,
                                "ce_score": round(float(c.ce_score), 4),
                                "chunk_preview": section_chunks[c.index][1][:300],
                            }
                            for c in decision.candidates
                        ],
                    })
                elif decision.decision != evidence_layer.DECISION_SUPPORTED:
                    unassigned.append({
                        "sentence": sentence,
                        "best_score": record.best_score,
                        "best_label": record.best_label,
                        "reason": decision.reason,
                        "decision": decision.decision,
                    })
                if decision.attached:
                    attach_index = decision.primary_index

        if attach_index is not None:
            # ``assigned`` remains the plausible section labels (build statistics
            # and metadata depend on it), but the sentence has exactly ONE
            # training destination. It is never copied into one chunk per section.
            for label in record.assigned:
                aligned_sentences[label].append(sentence)
            record.primary_chunk = attach_index
            record.primary_chunk_score = float(row[attach_index])
            chunk_sentences.setdefault(attach_index, []).append(sentence)
            chunk_pair_scores.setdefault(attach_index, []).append(float(row[attach_index]))

        per_sentence.append(record)

    if evidence_mode:
        decision_counts["below_threshold"] = sum(
            1 for entry in unassigned if entry.get("reason") == "below_threshold"
        )

    aligned = {label: " ".join(sents) for label, sents in aligned_sentences.items() if sents}
    aligned_by_chunk = {j: " ".join(sents) for j, sents in chunk_sentences.items()}

    result = AlignmentResult(
        aligned=aligned,
        aligned_sentences={label: sents for label, sents in aligned_sentences.items() if sents},
        aligned_by_chunk=aligned_by_chunk,
        chunk_sentences=chunk_sentences,
        chunk_pair_scores=chunk_pair_scores,
        unassigned=unassigned,
        ambiguous=ambiguous,
        decision_counts=decision_counts,
        evidence_mode=bool(evidence_mode),
        evidence_settings=evidence_settings.as_dict() if evidence_mode else {},
        per_sentence=per_sentence,
        threshold=threshold,
        multi_label=allow_multi_label,
    )

    warn_rate = _config_value("ALIGNMENT_UNASSIGNED_WARN_RATE", _UNASSIGNED_WARN_RATE_DEFAULT)
    if result.unassigned_rate > warn_rate:
        logger.warning(
            "Unassigned rate %.0f%% exceeds the %.0f%% review line - revisit the "
            "threshold, or state the general-domain Cross-Encoder's limits in the report.",
            result.unassigned_rate * 100,
            warn_rate * 100,
        )

    return result


def build_aligned_target(
    sentences: list[str],
    max_tokens: int | None = None,
    measure=None,
) -> tuple[str, list[str], list[str]]:
    """Join ``sentences`` into one target without ever splitting a sentence.

    Returns ``(target, kept, overflow)``. Sentences are added in the given
    (deterministic reference-summary) order while the accumulated target stays
    within ``max_tokens`` as measured by ``measure`` (``chunker.approximate_tokens``
    by default). As soon as adding the next complete sentence would exceed the
    budget, construction stops; that sentence and every sentence after it are
    returned as ``overflow`` so the caller can report them rather than discard
    them silently. ``max_tokens=None`` means no budget.
    """
    measure = measure or approximate_tokens
    kept: list[str] = []
    current = ""
    for position, sentence in enumerate(sentences):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if max_tokens is not None and measure(candidate) > max_tokens:
            return current, kept, list(sentences[position:])
        kept.append(sentence)
        current = candidate
    return current, kept, []


def validate_token_lengths(
    records,
    tokenizer,
    max_input_tokens: int,
    max_target_tokens: int,
) -> dict:
    """Measure every record with the real BART tokenizer and report violations.

    This is the only place real tokenizer counts are computed. Nothing is
    truncated here: a record that exceeds a limit is counted and reported, and
    the caller decides what to do. ``tokenizer`` is any callable returning a
    mapping with an ``input_ids`` entry, so the check is unit-testable without
    ``transformers`` installed.
    """
    report = {
        "records": 0,
        "max_input_tokens": 0,
        "max_target_tokens": 0,
        "input_violations": 0,
        "target_violations": 0,
        "violation_examples": [],
        "ok": True,
    }
    for record in records:
        report["records"] += 1
        n_input = len(tokenizer(record["chunk_text"])["input_ids"])
        n_target = len(tokenizer(record["aligned_target"])["input_ids"])
        report["max_input_tokens"] = max(report["max_input_tokens"], n_input)
        report["max_target_tokens"] = max(report["max_target_tokens"], n_target)
        for field_name, tokens, limit, key in (
            ("chunk_text", n_input, max_input_tokens, "input_violations"),
            ("aligned_target", n_target, max_target_tokens, "target_violations"),
        ):
            if tokens > limit:
                report[key] += 1
                if len(report["violation_examples"]) < 10:
                    report["violation_examples"].append({
                        "doc_id": record.get("doc_id"),
                        "chunk_id": record.get("chunk_id"),
                        "field": field_name,
                        "tokens": tokens,
                        "limit": limit,
                    })
    report["ok"] = report["input_violations"] == 0 and report["target_violations"] == 0
    return report

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
  * ``src.postprocess.cite_verify.split_sentences`` - the repository's existing
    sentence splitter.

The pseudo-targets produced here are a training aid, never an evaluation
target. Final evaluation runs against the true IN-ABS summary.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

sys.path.insert(0, ".")

import config
from src.postprocess.cite_verify import split_sentences
from src.shared.cross_encoder import get_cross_encoder

logger = logging.getLogger(__name__)

# Sentences shorter than this are dropped before scoring: fragments like
# "Hence the appeal." carry no alignment signal, and the shared Cross-Encoder
# rates them as plausible matches for almost any chunk.
_MIN_SENTENCE_CHARS = 25

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
    assigned: list[str] = field(default_factory=list)

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
    sentences = [s for s in split_sentences(full_summary or "") if len(s) >= _MIN_SENTENCE_CHARS]
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
    """
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
    per_sentence: list[SentenceAlignment] = []

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
            })
        elif allow_multi_label:
            cutoff = record.best_score * relative_margin
            record.assigned = [label for label in above if per_label[label] >= cutoff]
        else:
            record.assigned = [max(above, key=lambda lbl: per_label[lbl])]

        for label in record.assigned:
            aligned_sentences[label].append(sentence)

            # Attach the sentence to the single best-matching chunk of this
            # section. ``default=None`` guards the degenerate case where a label
            # was declared but has no chunk.
            best_chunk = max(
                (j for j, (lbl, _) in enumerate(section_chunks) if lbl == label),
                key=lambda j: row[j],
                default=None,
            )
            if best_chunk is not None:
                chunk_sentences.setdefault(best_chunk, []).append(sentence)
                chunk_pair_scores.setdefault(best_chunk, []).append(float(row[best_chunk]))

        per_sentence.append(record)

    aligned = {label: " ".join(sents) for label, sents in aligned_sentences.items() if sents}
    aligned_by_chunk = {j: " ".join(sents) for j, sents in chunk_sentences.items()}

    result = AlignmentResult(
        aligned=aligned,
        aligned_sentences={label: sents for label, sents in aligned_sentences.items() if sents},
        aligned_by_chunk=aligned_by_chunk,
        chunk_sentences=chunk_sentences,
        chunk_pair_scores=chunk_pair_scores,
        unassigned=unassigned,
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

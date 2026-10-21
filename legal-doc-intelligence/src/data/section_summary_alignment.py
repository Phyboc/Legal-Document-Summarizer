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
from collections.abc import Callable
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

# Trailing-period tokens that do NOT end a sentence. IN-ABS summaries are dense
# with reporter citations (``A.I.R.``, ``P.C.``, ``I.L.R.``), section references
# (``s.``, ``cl.``, ``art.``) and party abbreviations (``vs.``, ``Ltd.``). A naive
# ``(?<=[.!?])\s+`` split breaks inside them and produces citation fragments
# (``"Ry."``, ``"1936 P.C."``) that then became whole training targets in the
# first full build. Abbreviations are stored lower-case, without the final dot.
_ABBREVIATIONS = frozenset({
    "a", "anr", "art", "arts", "bom", "cal", "ch", "cj", "cl", "cls", "co",
    "corp", "cr", "crlj", "del", "distt", "dr", "edn", "etc", "govt", "guj",
    "hc", "hon", "ibid", "ilr", "j", "jj", "ker", "ltd", "lr", "lrs", "mad",
    "mr", "mrs", "ms", "no", "nos", "ors", "p", "para", "paras", "pat", "pc",
    "pp", "prof", "r", "raj", "rs", "ry", "s", "sc", "sch", "scr", "scc",
    "sec", "secs", "shri", "smt", "st", "v", "vol", "vs",
})

# ``A.I.R.`` / ``P.C.`` / ``P.W.`` / ``M.R.`` / ``B.`` - an initialism, not a
# sentence end. The trailing dot is part of the match.
_INITIALISM = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]?")
# Sub-section / pin cite such as ``56(j).`` or ``265(1)(a).``
_SUBSECTION = re.compile(r"\d+\([A-Za-z0-9]+\)\.?")
_TERMINAL = re.compile(r"[.!?][\"'\])]*$")
# SCR headnote pinpoint markers - "[473 B C]", "[184E H; 185A]" - stripped from
# reference summaries before splitting (see summary_sentences).
_PAGE_MARKER = re.compile(r"[ \t]*\[\s*\d{1,4}\s*[A-H]\d?(?:[\s,;-]*(?:\d{1,4}\s*)?[A-H]\d?)*\s*\]")


def _ends_with_abbreviation(text: str, period_index: int) -> bool:
    """True when the period at ``period_index`` belongs to an abbreviation."""
    token = text[: period_index + 1].rsplit(None, 1)[-1].lstrip("\"'([")
    core = token.rstrip(".!?")
    if not core:
        return False
    if _INITIALISM.fullmatch(token):
        return True
    if _SUBSECTION.fullmatch(token):
        return True
    return core.lower() in _ABBREVIATIONS


def _looks_like_fragment(piece: str) -> bool:
    """True for a citation stub such as ``"Ry."`` or ``"1936 P.C."``.

    A real operative order can be two words long (``"Appeal dismissed."``), so
    length alone is not the test: the piece must also contain no lower-case
    word of four or more letters, which every genuine English sentence does.
    """
    return len(piece.split()) <= 2 and not re.search(r"[a-z]{4,}", piece)


def _split_on_periods(text: str) -> list[str]:
    """Split at real sentence boundaries, protecting abbreviations."""
    boundaries: list[int] = []
    for match in re.finditer(r"[.!?]", text):
        end = match.end()
        if end < len(text) and not text[end].isspace():
            continue
        rest = text[end:].lstrip()
        if rest and not re.match(r"[\"'(\[]?[A-Z0-9]", rest):
            continue
        if _ends_with_abbreviation(text, match.start()):
            continue
        boundaries.append(end)

    pieces: list[str] = []
    start = 0
    for boundary in boundaries:
        piece = text[start:boundary].strip()
        if piece:
            pieces.append(piece)
        start = boundary
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _merge_fragments(pieces: list[str]) -> list[str]:
    """Reattach pieces that cannot stand as sentences.

    Two repairs: a piece that does not end in terminal punctuation is glued to
    the next one (the split happened *after* it, so it is still mid-sentence),
    and a leading citation stub is glued to the sentence that follows it. The
    original characters are preserved; only whitespace at the join is normalised.
    """
    merged: list[str] = []
    for piece in pieces:
        if merged and (
            _looks_like_fragment(piece) or not _TERMINAL.search(merged[-1])
        ):
            merged[-1] = f"{merged[-1]} {piece}".strip()
        else:
            merged.append(piece)

    repaired: list[str] = []
    for piece in merged:
        if repaired and _looks_like_fragment(repaired[-1]):
            repaired[-1] = f"{repaired[-1]} {piece}".strip()
        else:
            repaired.append(piece)
    return repaired


def summary_sentences(full_summary: str) -> list[str]:
    """Split a reference summary into sentences, keeping every non-empty one.

    Abbreviation-aware: ``"1936 P.C."`` and ``"Ry."`` are no longer emitted as
    standalone sentences, so they cannot become stand-alone training targets.
    Genuine short orders (``"Appeal dismissed."``) are still retained.
    """
    if not full_summary or not full_summary.strip():
        return []
    text = re.sub(r"\s+", " ", full_summary).strip()
    # SCR headnote pinpoint markers - "[473 B C]", "[184E H; 185A]" - point at
    # pages of the printed report, not content; stripped so a model never learns
    # to emit page references it cannot know.
    text = _PAGE_MARKER.sub("", text)
    return _merge_fragments(_split_on_periods(text))

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
    # The window inside the primary chunk that produced the score. This is what
    # makes a match auditable: a reviewer can read the exact span the decision
    # rests on instead of re-reading a 600-word chunk.
    evidence_span: str | None = None
    evidence_location: dict | None = None
    # --- evidence layer (only populated when evidence mode is enabled) -------- #
    # One of SUPPORTED / AMBIGUOUS / UNSUPPORTED / NOT_SUMMARY_CONTENT, or None
    # when the sentence never reached the evidence layer (below the CE threshold).
    decision: str | None = None
    evidence: dict | None = None
    evidence_reason: str | None = None
    ambiguity: dict | None = None
    # True when the sentence states the court's operative order and was routed to
    # a judgement chunk (see align_summary_to_sections). Provenance only.
    operative: bool = False

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
    # Real (sentence x window) comparison count, so throughput is measured rather
    # than assumed: it is the number that decides how long a build takes.
    pairs_scored: int = 0

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
    """Score (sentence, window) pairs with the shared Cross-Encoder.

    ``CrossEncoderSingleton`` exposes ``score`` (this replaced the older
    ``predict`` delegate). ``normalize=True`` applies the singleton's sigmoid,
    which is the space the validated thresholds live in - the recorded
    unassigned scores in
    ``artifacts/training_data/validation_unassigned_log.jsonl`` (0.03, 0.0086,
    ...) are all in [0, 1], and a 0.40 / 0.90 threshold pair is only meaningful
    there. The batch size comes from ``config`` so a CPU/GPU benchmark is
    reproducible.
    """
    return get_cross_encoder().score(
        pairs, normalize=True, batch_size=config.CROSS_ENCODER_BATCH_SIZE
    )


# --------------------------------------------------------------------------- #
# Windows and candidate selection
# --------------------------------------------------------------------------- #

# A character span covering one sentence (period-terminated, or the tail).
_SENTENCE_SPAN = re.compile(r"\S(?:.*?)(?:[.!?](?=\s|$)|$)", re.DOTALL)
_CONTENT_WORD = re.compile(r"[a-z]{4,}")


def _content_words(text: str) -> list[str]:
    return _CONTENT_WORD.findall(text.lower())


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences in ``text``, whitespace-trimmed."""
    spans: list[tuple[int, int]] = []
    for match in _SENTENCE_SPAN.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def chunk_windows(
    text: str, max_tokens: int | None = None, overlap_tokens: int | None = None
) -> list[tuple[int, int, str]]:
    """Split one chunk into overlapping windows that fit the Cross-Encoder.

    Returns ``(char_start, char_end, window_text)`` triples relative to the chunk,
    which is the location a reviewer needs to audit a match: chunk id and
    ``para_range`` give the document position, the character range gives the exact
    evidence inside it.

    The encoder truncates at 512 sub-word tokens. Windows are bounded in
    *estimated* tokens well below that (350 by default) because the chunker's
    estimate under-counts dense legal text - the same reason the builder measures
    chunks with the real tokenizer before scoring them. Building windows costs
    nothing next to the encoder pass and makes the whole chunk visible to it.
    """
    max_tokens = max_tokens or config.ALIGNMENT_WINDOW_TOKENS
    overlap_tokens = overlap_tokens if overlap_tokens is not None else config.ALIGNMENT_WINDOW_OVERLAP
    spans = _sentence_spans(text)
    if not spans:
        return [(0, len(text), text)] if text else []

    windows: list[tuple[int, int, str]] = []
    current: list[tuple[int, int]] = [spans[0]]
    for span in spans[1:]:
        candidate = text[current[0][0]:span[1]]
        if approximate_tokens(candidate) > max_tokens:
            windows.append((current[0][0], current[-1][1], text[current[0][0]:current[-1][1]]))
            # Carry the trailing sentences over as overlap so a sentence split
            # across the boundary is still fully visible in one window.
            carry: list[tuple[int, int]] = []
            carried = 0
            for previous in reversed(current):
                carried += approximate_tokens(text[previous[0]:previous[1]])
                if carried > overlap_tokens:
                    break
                carry.append(previous)
            current = list(reversed(carry))
        current.append(span)

    if current:
        windows.append((current[0][0], current[-1][1], text[current[0][0]:current[-1][1]]))
    return windows


def rank_candidate_chunks(
    sentence: str,
    chunk_texts: list[str],
    top_k: int | None = None,
    document_df: dict | None = None,
    section_labels: list[str] | None = None,
) -> list[int]:
    """Cheap IDF-weighted lexical prefilter: the chunks worth the encoder's time.

    Deliberately not a hard rule: it only decides *which* chunks are scored, never
    what is assigned - the Cross-Encoder still makes that decision, and a sentence
    sharing no content word with any chunk still gets its ``top_k`` candidates.

    With ``section_labels``, the best-ranked chunk of every section present is added
    to the selection whatever its rank. Without that, a section whose chunks happen
    to rank below ``top_k`` would be unscoreable for this sentence and could never
    be assigned - the prefilter would be silently editing the label set, which is
    exactly the kind of quiet bias this pipeline is supposed to avoid.

    Returns chunk indices ascending, so the result is deterministic and stable.
    """
    top_k = config.ALIGNMENT_TOP_K_CANDIDATES if top_k is None else top_k
    # ``top_k <= 0`` means "do not prefilter at all": every chunk is a candidate.
    if top_k <= 0 or len(chunk_texts) <= top_k:
        return list(range(len(chunk_texts)))

    if document_df is None:
        document_df = evidence_layer.document_frequency(chunk_texts)
    total = max(1, len(chunk_texts))
    import math

    sentence_words = set(_content_words(sentence))
    scored: list[tuple[float, int]] = []
    for index, text in enumerate(chunk_texts):
        words = set(_content_words(text))
        overlap = sentence_words & words
        score = sum(
            math.log((1 + total) / (1 + document_df.get(word, 0)))
            for word in overlap
        )
        scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [index for _, index in scored]

    selected = set(ordered[:top_k])
    if section_labels is not None:
        covered: set[str] = set()
        for index in ordered:
            label = section_labels[index]
            if label not in covered:
                covered.add(label)
                selected.add(index)
    return sorted(selected)


def score_summary_sentences(
    section_chunks: list[tuple[str, str]],
    full_summary: str,
    *,
    window_tokens: int | None = None,
    overlap_tokens: int | None = None,
    top_k_candidates: int | None = None,
    use_prefilter: bool | None = None,
) -> tuple[
    list[str],
    list[dict[str, float]],
    list[list[float]],
    list[dict[int, dict]],
    int,
]:
    """Score every summary sentence against the evidence windows of its candidates.

    Returns ``(sentences, per_label_scores, per_chunk_scores, window_hits)``:

    * ``per_label_scores[i][label]`` - best score sentence *i* reached against any
      chunk carrying that label;
    * ``per_chunk_scores[i][j]`` - best score for sentence *i* against chunk *j*
      (the best of that chunk's windows);
    * ``window_hits[i][j]`` - which window produced that score, as
      ``{"window_index", "char_start", "char_end"}``, so the matched evidence span
      can be recorded and audited.

    Two changes from scoring whole chunks, both forced by the encoder's 512-token
    input limit:

    1. a chunk is scored as overlapping windows that fit the encoder, with the
       chunk taking its best window. A whole-chunk call silently discarded
       everything past ~512 tokens, which is most of an 850-token chunk - the
       alignment comment claimed this was handled, but only at *section* level,
       so a sentence could win a chunk whose relevant text the encoder never saw;
    2. a cheap lexical prefilter picks the ``top_k_candidates`` chunks per
       sentence. This bounds the cost as sentences x candidates rather than
       sentences x chunks, which is what made a handful of long judgments take
       tens of minutes each on CPU.
    """
    sentences = summary_sentences(full_summary or "")
    if not sentences or not section_chunks:
        return sentences, [], [], [], 0

    chunk_texts = [text for _, text in section_chunks]
    windows = [
        chunk_windows(text, max_tokens=window_tokens, overlap_tokens=overlap_tokens)
        for text in chunk_texts
    ]
    if use_prefilter is None:
        use_prefilter = config.ALIGNMENT_USE_LEXICAL_PREFILTER
    document_df = evidence_layer.document_frequency(chunk_texts)

    # One flat pair list keeps the encoder call batched; ``slots`` remembers which
    # (sentence, chunk, window) each row of the result belongs to.
    pairs: list[list[str]] = []
    slots: list[tuple[int, int, int]] = []
    section_labels = [label for label, _ in section_chunks]
    for i, sentence in enumerate(sentences):
        candidates = (
            rank_candidate_chunks(
                sentence, chunk_texts, top_k_candidates, document_df, section_labels
            )
            if use_prefilter
            else list(range(len(chunk_texts)))
        )
        for j in candidates:
            for window_index, (_, _, window_text) in enumerate(windows[j]):
                pairs.append([sentence, window_text])
                slots.append((i, j, window_index))

    flat_scores = [float(value) for value in _score_pairs(pairs)] if pairs else []

    per_chunk_scores: list[list[float]] = [[] for _ in sentences]
    for i in range(len(sentences)):
        per_chunk_scores[i] = [0.0] * len(chunk_texts)
    window_hits: list[dict[int, dict]] = [{} for _ in sentences]

    for (i, j, window_index), score in zip(slots, flat_scores):
        previous = window_hits[i].get(j)
        if previous is None or score > previous["score"]:
            window_hits[i][j] = {
                "window_index": window_index,
                "char_start": windows[j][window_index][0],
                "char_end": windows[j][window_index][1],
                "score": round(score, 4),
            }
        if score > per_chunk_scores[i][j]:
            per_chunk_scores[i][j] = score

    per_label_scores: list[dict[str, float]] = []
    for i in range(len(sentences)):
        best: dict[str, float] = {}
        for j, (label, _) in enumerate(section_chunks):
            if j not in window_hits[i]:
                continue
            score = per_chunk_scores[i][j]
            if score > best.get(label, -1.0):
                best[label] = score
        per_label_scores.append(best)

    return sentences, per_label_scores, per_chunk_scores, window_hits, len(pairs)


def _unpack_precomputed(precomputed):
    """Accept the 3-, 4- or 5-tuple forms of precomputed scores.

    Three elements is the original ``(sentences, per_label, per_chunk)`` shape that
    callers and tests already use; window hits and the pair count are optional
    additions, so an older caller keeps working unchanged.
    """
    sentences, per_label, per_chunk = precomputed[0], precomputed[1], precomputed[2]
    window_hits = precomputed[3] if len(precomputed) > 3 else [{} for _ in sentences]
    pair_count = precomputed[4] if len(precomputed) > 4 else 0
    return sentences, per_label, per_chunk, window_hits, pair_count


def align_summary_to_sections(
    section_chunks: list[tuple[str, str]],
    full_summary: str,
    threshold: float | None = None,
    allow_multi_label: bool | None = None,
    relative_margin: float | None = None,
    precomputed: tuple | None = None,
    is_operative: Callable[[str], bool] | None = None,
    evidence_mode: bool | None = None,
    evidence_settings=None,
    chunk_positions: list[float] | None = None,
    window_tokens: int | None = None,
    overlap_tokens: int | None = None,
    top_k_candidates: int | None = None,
    use_prefilter: bool | None = None,
) -> AlignmentResult:
    """Align each sentence of ``full_summary`` to section label(s) and to a chunk.

    ``section_chunks`` is ``[(section_label, chunk_text), ...]`` - one entry per
    chunk, in the document's chunk order.

    A sentence is assigned to every section that (a) scores at or above
    ``threshold`` and (b) is within ``relative_margin`` of the sentence's best
    section. A sentence that satisfies neither is left unassigned. Within each
    assigned section the sentence is attached to that section's best-matching
    chunk, which is what makes a chunk-level target possible at all.

    ``is_operative`` marks sentences that state the court's final order. Relevance
    alone does not place them: a Cross-Encoder scores a restated fact about as
    highly as the order itself, so most orders landed in facts or reasoning chunks.
    For an operative sentence the ``judgement`` label is added whenever a judgement
    chunk clears ``threshold`` (the source must support it), and its primary chunk is
    the best judgement chunk. Without a judgement chunk above threshold, nothing changes.

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

    ``window_tokens`` / ``overlap_tokens`` / ``top_k_candidates`` / ``use_prefilter``
    are forwarded to ``score_summary_sentences`` and control how chunks are windowed
    for the 512-token Cross-Encoder and how many candidate chunks each sentence is
    compared against. They are ignored when ``precomputed`` scores are supplied.
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
        (
            sentences, all_scores, chunk_scores, window_hits, pair_count
        ) = _unpack_precomputed(precomputed)
    else:
        (
            sentences, all_scores, chunk_scores, window_hits, pair_count
        ) = score_summary_sentences(
            section_chunks,
            full_summary,
            window_tokens=window_tokens,
            overlap_tokens=overlap_tokens,
            top_k_candidates=top_k_candidates,
            use_prefilter=use_prefilter,
        )

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

    for sentence_index, (sentence, per_label, row) in enumerate(
        zip(sentences, all_scores, chunk_scores)
    ):
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

        # Operative-order routing (outcome preservation). When the sentence states
        # the court's operative order and the judgement section cleared the
        # threshold, force its training destination to a judgement chunk rather
        # than wherever it happens to score best among its plausible sections.
        record.operative = bool(
            is_operative and "judgement" in above and is_operative(sentence)
        )
        if record.operative and "judgement" not in record.assigned:
            record.assigned.append("judgement")

        candidate_indices = [
            j for j, (label, _) in enumerate(section_chunks) if label in record.assigned
        ]
        if record.operative:
            judgement_candidates = [
                j for j in candidate_indices if section_chunks[j][0] == "judgement"
            ]
            if judgement_candidates:
                candidate_indices = judgement_candidates

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
            # Record which window of the winning chunk the evidence came from.
            hit = (
                window_hits[sentence_index].get(attach_index)
                if sentence_index < len(window_hits) else None
            )
            if hit:
                chunk_text = section_chunks[attach_index][1]
                record.evidence_span = chunk_text[hit["char_start"]:hit["char_end"]]
                record.evidence_location = {
                    "chunk_index": attach_index,
                    "section": section_chunks[attach_index][0],
                    "window_index": hit["window_index"],
                    "char_start": hit["char_start"],
                    "char_end": hit["char_end"],
                    "score": hit["score"],
                }
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
        pairs_scored=pair_count,
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

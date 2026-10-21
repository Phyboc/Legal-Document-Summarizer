"""Tests for the evidence-aware alignment layer (``src/data/evidence_alignment.py``).

These run fully offline: no Cross-Encoder, no BART tokenizer, no spaCy, no
network. Every alignment test supplies ``precomputed`` scores, which is the same
technique ``tests/test_alignment.py`` uses, so the evidence logic is exercised
deterministically.

The cases map onto the required list:

  1  date forms normalise            10 equal evidence -> CE tie-break
  2  amounts                         11 equal evidence/CE -> coverage tie-break
  3  provisions                      12 final locality tie-break
  4  citations                       13 nearby-no-evidence loses to distant-with-evidence
  5  OCR / footnote normalisation    14 UNSUPPORTED -> primary_chunk None
  6  citation fragment ->            15 AMBIGUOUS multi-event -> primary_chunk None
     NOT_SUMMARY_CONTENT             16 no sentence in multiple chunks
  7  matching date/year -> supported 17 deterministic output
  8  missing != contradiction        18 target-length layer unchanged
  9  the reported Single Judge regression

Run:  python -m pytest tests/test_evidence_alignment.py -q
"""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.data.evidence_alignment import (
    DECISION_AMBIGUOUS,
    DECISION_NOT_SUMMARY_CONTENT,
    DECISION_SUPPORTED,
    DECISION_UNSUPPORTED,
    STATUS_ABSENT,
    STATUS_CONTRADICTED,
    STATUS_MISSING,
    STATUS_SATISFIED,
    Candidate,
    EvidenceSettings,
    coverage_ratio,
    decide,
    extract_amounts,
    extract_citations,
    extract_dates,
    extract_evidence,
    extract_provisions,
    extract_years,
    is_not_summary_content,
    normalize_text,
)
from src.data.section_summary_alignment import (
    align_summary_to_sections,
    build_aligned_target,
    validate_token_lengths,
)
from config import ALIGNMENT_RELATIVE_MARGIN, ALIGNMENT_THRESHOLD

# --------------------------------------------------------------------------- #
# 1. Date forms
# --------------------------------------------------------------------------- #

def test_every_required_date_form_normalises_to_one_canonical_key():
    cases = {
        "September 3, 1959": "1959-09-03",
        "3-9-1959": "1959-09-03",
        "3.9.1959": "1959-09-03",
        "20th August, 1991": "1991-08-20",
        "the 5th day of January, 1981": "1981-01-05",
        "24 July 1985": "1985-07-24",
    }
    for text, expected in cases.items():
        assert extract_dates(normalize_text(text)) == {expected}, text


def test_month_without_a_day_keeps_a_distinct_unknown_day_key():
    assert extract_dates(normalize_text("in January 1981")) == {"1981-01-00"}


def test_a_dayless_form_is_dropped_when_the_same_month_has_a_full_date():
    # "January 5, 1981" must not count as two independent facts.
    assert extract_dates(normalize_text("January 5, 1981")) == {"1981-01-05"}


def test_1981_82_is_a_year_range_not_a_calendar_date():
    assert extract_years(normalize_text("the period 1981-82 was under review")) == {
        "1981", "1981/1982",
    }
    assert extract_dates(normalize_text("the period 1981-82 was under review")) == set()


# --------------------------------------------------------------------------- #
# 2. Amounts
# --------------------------------------------------------------------------- #

def test_amount_forms_normalise_to_currency_integers():
    cases = {
        "Rs. 1,00,000": "inr:100000",
        "\u20b95,00,000": "inr:500000",
        "50 lakhs": "inr:5000000",
        "two crores": "inr:20000000",
        "$2,500": "usd:2500",
    }
    for text, expected in cases.items():
        assert expected in extract_amounts(normalize_text(text)), text


# --------------------------------------------------------------------------- #
# 3. Provisions
# --------------------------------------------------------------------------- #

def test_statutory_provisions_are_extracted():
    assert "section 32" in extract_provisions(normalize_text("Section 32 of the Income Tax Act"))
    assert "article 14" in extract_provisions(normalize_text("Article 14 was invoked"))
    assert "order 41 rule 1" in extract_provisions(normalize_text("Order 41 Rule 1 applies"))


# --------------------------------------------------------------------------- #
# 4. Citations
# --------------------------------------------------------------------------- #

def test_report_citations_are_extracted():
    assert "1982-karnataka-179" in extract_citations(normalize_text("1982 Karnataka 179"))
    assert "1962-suppl" in extract_citations(normalize_text("[1962] Suppl."))
    assert "1966-punjab-509" in extract_citations(normalize_text("A.I.R. 1966 Punjab 509"))


# --------------------------------------------------------------------------- #
# 5. OCR and noise normalisation
# --------------------------------------------------------------------------- #

def test_footnotes_supra_quotes_ampersands_and_vs_are_normalised():
    normalized = normalize_text(
        "Ram & Sons v. State vs Union versus Crown, the appellant's case (supra)"
    )
    assert " and " in normalized
    assert "versus" not in normalized and "vs" in normalized
    assert "appellant case" in normalized, "possessive must be stripped, not fused"
    assert "supra" not in normalized
    evidence = extract_evidence("the order dated January 5, 1981 [489F, H] was challenged.")
    assert "489f" not in evidence.content_words
    assert evidence.dates == {"1981-01-05"}


def test_an_ocr_word_split_still_matches_the_intact_word():
    sentence = extract_evidence("manufacturing company")
    chunk = extract_evidence("Manufac turing company")
    assert coverage_ratio(sentence.content_words, chunk.word_set) == 1.0


# --------------------------------------------------------------------------- #
# 6. Citation-only fragment
# --------------------------------------------------------------------------- #

def test_a_citation_fragment_is_not_summary_content():
    fragment = "A.I.R. 1966 Punjab 509; Siri Chand vs Jot Ram, Punjab Law Reporter Vol."
    assert is_not_summary_content(extract_evidence(fragment)) is True
    decision = decide(fragment, [Candidate(index=0, label="facts", text="any chunk text")])
    assert decision.decision == DECISION_NOT_SUMMARY_CONTENT
    assert decision.primary_index is None


def test_a_short_operative_order_is_still_summary_content():
    # Invariant 6: terse orders must survive, so they must NOT be treated as
    # citation fragments.
    for order in ("Appeal dismissed.", "The appeal is dismissed.", "The petition is allowed."):
        assert is_not_summary_content(extract_evidence(order)) is False, order


# --------------------------------------------------------------------------- #
# 7. Matching date/year is evidence
# --------------------------------------------------------------------------- #

def test_a_matching_date_and_year_are_satisfied_evidence():
    sentence = "The order of January 5, 1981 was challenged."
    chunk = Candidate(
        index=0, label="facts",
        text="The assessee challenged the order dated January 5, 1981 before the High Court.",
        ce_score=0.8,
    )
    decision = decide(sentence, [chunk])
    assert decision.decision == DECISION_SUPPORTED
    assert decision.primary_index == 0
    classes = decision.candidates[0].classes
    assert classes["dates"].status == STATUS_SATISFIED
    assert classes["years"].status == STATUS_SATISFIED


# --------------------------------------------------------------------------- #
# 8. Missing evidence vs contradiction
# --------------------------------------------------------------------------- #

def test_absent_evidence_is_missing_not_contradicted():
    # The chunk mentions no date at all: absence proves nothing, so the candidate
    # stays admissible.
    decision = decide(
        "The order dated January 5, 1981 was challenged.",
        [Candidate(
            index=0, label="facts", ce_score=0.8,
            text="The said order was challenged in appeal by the assessee before the High Court.",
        )],
    )
    classes = decision.candidates[0].classes
    assert classes["dates"].status == STATUS_MISSING
    assert classes["years"].status == STATUS_MISSING
    assert decision.candidates[0].contradicted is False
    assert decision.decision == DECISION_SUPPORTED


def test_a_different_date_is_contradicted_only_under_the_high_confidence_rule():
    sentence = "The order of January 5, 1981 was challenged."
    chunk = Candidate(
        index=0, label="facts", ce_score=0.9,
        text="By order dated September 17, 1982 the efficiency bar order was passed.",
    )
    strict = decide(sentence, [chunk], settings=EvidenceSettings(contradiction_mode="high_confidence"))
    assert strict.candidates[0].classes["dates"].status == STATUS_CONTRADICTED
    assert strict.decision == DECISION_UNSUPPORTED
    assert strict.reason == "all_candidates_contradicted"

    # With the rule switched off, the same pair is merely MISSING - the strong
    # mismatch rule is explicit and therefore opt-outable.
    relaxed = decide(sentence, [chunk], settings=EvidenceSettings(contradiction_mode="off"))
    assert relaxed.candidates[0].classes["dates"].status == STATUS_MISSING
    assert relaxed.candidates[0].contradicted is False


# --------------------------------------------------------------------------- #
# 9. The reported regression
# --------------------------------------------------------------------------- #

SINGLE_JUDGE_SENTENCE = "The Single Judge's decision of January 5, 1981 was upheld."
SINGLE_JUDGE_CHUNK = Candidate(
    index=2, label="facts", ce_score=0.60, position=0.2,
    text="The learned Single Judge by order dated January 5, 1981 held that the assessment was valid.",
)
EFFICIENCY_BAR_CHUNK = Candidate(
    index=9, label="reasoning", ce_score=0.95, position=0.8,
    text=(
        "By order dated September 17, 1982 the efficiency bar order was passed "
        "and the appellant was reverted from the post."
    ),
)


def test_the_single_judge_vs_efficiency_bar_regression():
    candidates = [EFFICIENCY_BAR_CHUNK, SINGLE_JUDGE_CHUNK]

    # The regression is real: Cross-Encoder similarity alone picks the wrong
    # chunk, because both describe the same case and share appellant / High Court
    # / order vocabulary.
    ce_choice = max(candidates, key=lambda c: c.ce_score).index
    assert ce_choice == EFFICIENCY_BAR_CHUNK.index, "fixture must reproduce the failure"

    decision = decide(SINGLE_JUDGE_SENTENCE, candidates)
    assert decision.decision == DECISION_SUPPORTED
    assert decision.primary_index == SINGLE_JUDGE_CHUNK.index, (
        "evidence must pick the chunk that contains the 5 January 1981 decision"
    )
    wrong = next(c for c in decision.candidates if c.index == EFFICIENCY_BAR_CHUNK.index)
    assert wrong.contradicted, "the later efficiency-bar chunk contradicts the 1981 date"


# --------------------------------------------------------------------------- #
# 10-13. Tie-break order
# --------------------------------------------------------------------------- #

IDENTICAL_ORDER_TEXT = "order dated January 5, 1981 challenged"


def test_equal_evidence_falls_back_to_the_cross_encoder_score():
    decision = decide(
        "The order of January 5, 1981 was challenged.",
        [
            Candidate(index=0, label="facts", text=IDENTICAL_ORDER_TEXT, ce_score=0.30),
            Candidate(index=1, label="facts", text=IDENTICAL_ORDER_TEXT, ce_score=0.90),
        ],
    )
    assert decision.decision == DECISION_SUPPORTED
    assert decision.primary_index == 1, "identical evidence -> higher CE wins"


def test_equal_evidence_and_ce_falls_back_to_coverage():
    # max_rare_df=0 switches Tier-2 rarity off, so coverage is the next key.
    settings = EvidenceSettings(max_rare_df=0)
    decision = decide(
        "alpha beta gamma delta epsilon.",
        [
            Candidate(index=0, label="facts", text="alpha beta xxx yyy zzz", ce_score=0.8),
            Candidate(index=1, label="facts", text="alpha beta xxx gamma yyy delta zzz", ce_score=0.8),
        ],
        settings=settings,
    )
    coverages = {c.index: c.coverage for c in decision.candidates}
    assert decision.decision == DECISION_SUPPORTED
    assert decision.primary_index == 1, "higher coverage wins"
    assert coverages[1] > coverages[0]


def test_locality_is_the_final_tie_break():
    shared = "alpha beta xxx gamma yyy delta zzz"
    decision = decide(
        "alpha beta gamma delta epsilon.",
        [
            Candidate(index=0, label="facts", text=shared, ce_score=0.8, position=0.20),
            Candidate(index=1, label="facts", text=shared, ce_score=0.8, position=0.80),
        ],
        settings=EvidenceSettings(max_rare_df=0),
        anchor_position=0.79,
    )
    assert decision.decision == DECISION_SUPPORTED
    assert decision.primary_index == 1, "closer to the anchor wins when everything else ties"


def test_a_nearby_chunk_without_evidence_loses_to_a_distant_one_with_evidence():
    decision = decide(
        "The order of January 5, 1981 was challenged.",
        [
            # Highest CE and closest to the anchor, but no evidence at all.
            Candidate(index=0, label="facts", ce_score=0.99, position=0.50,
                      text="the respondent filed a written statement in the petition"),
            Candidate(index=9, label="facts", ce_score=0.20, position=0.90,
                      text=IDENTICAL_ORDER_TEXT),
        ],
        anchor_position=0.50,
    )
    assert decision.primary_index == 9, "evidence must outrank locality"
    assert decision.reason == "tier1_evidence"


# --------------------------------------------------------------------------- #
# 14-15. Rejection and ambiguity
# --------------------------------------------------------------------------- #

def test_unsupported_sentence_gets_no_primary_chunk():
    decision = decide(
        "The appellant was awarded compensation of Rs. 5,00,000 by the order.",
        [Candidate(
            index=0, label="facts", ce_score=0.7,
            text="The respondent filed a written statement denying all the allegations made.",
        )],
    )
    assert decision.decision == DECISION_UNSUPPORTED
    assert decision.primary_index is None
    assert decision.reason == "soft_similarity_only_coverage_below_bound"


MULTI_EVENT_SENTENCE = (
    "The Single Judge decided the matter on January 5, 1981 and the "
    "efficiency bar order was passed on September 17, 1982."
)


def test_a_multi_event_sentence_spanning_distant_chunks_is_ambiguous():
    decision = decide(
        MULTI_EVENT_SENTENCE,
        [
            Candidate(index=0, label="facts", ce_score=0.90,
                      text="by order dated January 5, 1981 the Single Judge held so"),
            Candidate(index=7, label="judgement", ce_score=0.85,
                      text="by order dated September 17, 1982 the efficiency bar order was passed"),
        ],
    )
    assert decision.decision == DECISION_AMBIGUOUS
    assert decision.primary_index is None
    assert decision.reason == "tier1_evidence_split_across_distant_chunks"
    assert decision.ambiguity is not None
    assert sorted(decision.ambiguity["chunk_indices"]) == [0, 7]


def test_a_genuine_tie_between_candidates_is_ambiguous():
    decision = decide(
        "The order of January 5, 1981 was challenged.",
        [
            Candidate(index=0, label="facts", text=IDENTICAL_ORDER_TEXT, ce_score=0.80),
            Candidate(index=1, label="judgement", text=IDENTICAL_ORDER_TEXT, ce_score=0.80),
        ],
    )
    assert decision.decision == DECISION_AMBIGUOUS
    assert decision.primary_index is None
    assert decision.reason == "candidates_effectively_tied"


# --------------------------------------------------------------------------- #
# Integration with the existing aligner
# --------------------------------------------------------------------------- #

CHUNKS = [
    ("facts", "the appeal was dismissed on january 5, 1981 by the single judge"),
    ("reasoning", "the respondent filed a written statement denying all the allegations"),
    ("judgement", "the efficiency bar order was passed on september 17, 1982"),
]


def _align(sentences, per_label_scores, per_chunk_scores, **kwargs):
    return align_summary_to_sections(
        CHUNKS,
        "unused because precomputed scores are supplied",
        threshold=ALIGNMENT_THRESHOLD,
        allow_multi_label=True,
        relative_margin=ALIGNMENT_RELATIVE_MARGIN,
        precomputed=(sentences, per_label_scores, per_chunk_scores),
        **kwargs,
    )


def test_evidence_mode_is_off_by_default_and_preserves_the_old_behaviour():
    sentences = [MULTI_EVENT_SENTENCE]
    result = _align(sentences, [{"facts": 0.9, "reasoning": 0.1, "judgement": 0.85}], [[0.9, 0.1, 0.85]])
    record = result.per_sentence[0]
    assert result.evidence_mode is False
    assert record.decision is None, "no evidence layer ran"
    # The previous highest-CE-chunk rule still applies unchanged.
    assert record.primary_chunk == 0
    assert result.chunk_sentences == {0: sentences}


def test_no_sentence_is_ever_written_to_more_than_one_chunk_in_evidence_mode():
    sentences = [
        "The appeal was dismissed on January 5, 1981.",
        "The respondent filed a written statement.",
    ]
    per_label = [
        {"facts": 0.9, "reasoning": 0.1, "judgement": 0.1},
        {"facts": 0.1, "reasoning": 0.9, "judgement": 0.1},
    ]
    rows = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]]
    result = _align(sentences, per_label, rows, evidence_mode=True)

    assert result.evidence_mode is True
    assert [r.decision for r in result.per_sentence] == [DECISION_SUPPORTED, DECISION_SUPPORTED]
    written = [s for sents in result.chunk_sentences.values() for s in sents]
    assert len(written) == len(set(written)) == len(sentences)
    assert {r.primary_chunk for r in result.per_sentence} == {0, 1}


def test_ambiguous_sentence_is_attached_nowhere_and_reported():
    result = _align(
        [MULTI_EVENT_SENTENCE],
        [{"facts": 0.9, "reasoning": 0.1, "judgement": 0.85}],
        [[0.9, 0.1, 0.85]],
        evidence_mode=True,
    )
    record = result.per_sentence[0]
    assert record.decision == DECISION_AMBIGUOUS
    assert record.primary_chunk is None
    assert result.chunk_sentences == {}
    assert result.aligned_sentences == {} or MULTI_EVENT_SENTENCE not in {
        s for sents in result.aligned_sentences.values() for s in sents
    }
    assert len(result.ambiguous) == 1
    entry = result.ambiguous[0]
    assert entry["sentence"] == MULTI_EVENT_SENTENCE
    assert {c["chunk_index"] for c in entry["candidates"]} == {0, 2}
    assert entry["ambiguity"] is not None
    assert result.decision_counts.get(DECISION_AMBIGUOUS) == 1


def test_unsupported_sentence_is_not_written_into_any_record():
    result = _align(
        ["The appellant was awarded compensation of Rs. 5,00,000 by the order."],
        [{"facts": 0.9}],
        [[0.9, 0.1, 0.1]],
        evidence_mode=True,
    )
    record = result.per_sentence[0]
    assert record.decision == DECISION_UNSUPPORTED
    assert record.primary_chunk is None
    assert result.chunk_sentences == {}
    assert result.unassigned[0]["decision"] == DECISION_UNSUPPORTED
    assert result.unassigned[0]["reason"] == record.evidence_reason


def test_evidence_decisions_are_recorded_with_the_settings_used():
    result = _align(
        ["The appeal was dismissed on January 5, 1981."],
        [{"facts": 0.9}],
        [[0.9, 0.1, 0.1]],
        evidence_mode=True,
        evidence_settings=EvidenceSettings(coverage_bound=0.55),
    )
    assert result.evidence_settings["coverage_bound"] == 0.55
    assert result.decision_counts == {DECISION_SUPPORTED: 1, "below_threshold": 0}
    assert result.per_sentence[0].evidence is not None
    assert result.per_sentence[0].evidence["candidates"][0]["class_status"]["years"]["status"] == STATUS_SATISFIED


def test_decision_output_is_deterministic():
    kwargs = dict(evidence_mode=True)
    args = (
        [MULTI_EVENT_SENTENCE, "The appeal was dismissed on January 5, 1981."],
        [{"facts": 0.9, "judgement": 0.85}, {"facts": 0.95}],
        [[0.9, 0.1, 0.85], [0.95, 0.1, 0.1]],
    )
    first = _align(*args, **kwargs)
    second = _align(*args, **kwargs)
    assert [r.decision for r in first.per_sentence] == [r.decision for r in second.per_sentence]
    assert first.chunk_sentences == second.chunk_sentences
    assert first.decision_counts == second.decision_counts
    assert first.ambiguous == second.ambiguous

    unit_first = decide(MULTI_EVENT_SENTENCE, [EFFICIENCY_BAR_CHUNK, SINGLE_JUDGE_CHUNK]).as_dict()
    unit_second = decide(MULTI_EVENT_SENTENCE, [EFFICIENCY_BAR_CHUNK, SINGLE_JUDGE_CHUNK]).as_dict()
    assert unit_first == unit_second


# --------------------------------------------------------------------------- #
# 18. The target-length layer is untouched
# --------------------------------------------------------------------------- #

def test_target_budget_still_stops_at_a_sentence_boundary_and_reports_overflow():
    def measure(text):
        return len(text.split())

    target, kept, overflow = build_aligned_target(
        ["one two three.", "four five six."], max_tokens=4, measure=measure
    )
    assert target == "one two three."
    assert kept == ["one two three."]
    assert overflow == ["four five six."]


def test_token_length_validation_still_reports_without_truncating():
    def fake_tokenizer(text):
        return {"input_ids": text.split()}

    records = [{
        "doc_id": "train_00000",
        "chunk_id": 0,
        "chunk_text": "a b c",
        "aligned_target": "one two three four five six seven",
    }]
    report = validate_token_lengths(records, fake_tokenizer, max_input_tokens=5, max_target_tokens=3)
    assert report["target_violations"] == 1
    assert report["max_target_tokens"] == 7, "measured, never truncated"
    assert report["ok"] is False


# --------------------------------------------------------------------------- #
# Builder plumbing (imports offline: `datasets`/`transformers` are loaded lazily)
# --------------------------------------------------------------------------- #

def _build_args(*extra):
    import scripts.build_dataset as build_dataset

    return build_dataset.parse_args(["--output_dir", "unused", "--sample", "5", *extra])


def test_evidence_mode_is_off_unless_the_flag_is_passed_and_knobs_resolve():
    default = _build_args()
    assert default.evidence_mode is False

    enabled = _build_args("--evidence-mode", "--evidence-coverage-bound", "0.65")
    assert enabled.evidence_mode is True
    assert enabled.evidence_coverage_bound == 0.65
    # Unset knobs fall back to the module defaults, so the reported values cannot
    # drift from the values the aligner applies.
    assert enabled.evidence_max_rare_df == EvidenceSettings().max_rare_df
    assert enabled.evidence_contradiction_mode == EvidenceSettings().contradiction_mode


def test_chunk_positions_come_from_para_range_not_chunk_index():
    from src.data.chunker import Chunk
    import scripts.build_dataset as build_dataset

    chunks = [
        Chunk(text="a", section="facts", chunk_id=0, token_count=0, para_range=(0, 10)),
        Chunk(text="b", section="facts", chunk_id=1, token_count=0, para_range=(20, 30)),
        Chunk(text="c", section="reasoning", chunk_id=2, token_count=0, para_range=(40, 40)),
    ]
    positions = build_dataset._chunk_positions(chunks)
    assert positions == [0.125, 0.625, 1.0]

    # Position must follow the paragraph span, not the chunk id: a later chunk id
    # can sit earlier in the document, and chunk_document emits section blocks.
    reordered = [
        Chunk(text="late", section="judgement", chunk_id=0, token_count=0, para_range=(90, 100)),
        Chunk(text="early", section="facts", chunk_id=7, token_count=0, para_range=(0, 10)),
    ]
    span_positions = build_dataset._chunk_positions(reordered)
    assert span_positions[0] > span_positions[1], (
        "chunk id 0 sits later in the document than chunk id 7 and must score later"
    )


def test_build_stats_records_evidence_mode_settings_and_counts():
    import json
    import tempfile
    from pathlib import Path

    import scripts.build_dataset as build_dataset
    from src.data.evidence_alignment import DECISION_AMBIGUOUS, DECISION_SUPPORTED

    args = _build_args("--evidence-mode")
    build = build_dataset.DocumentBuild(
        doc_id="train_00000",
        split="train",
        evidence_decision_counts={
            DECISION_SUPPORTED: 3, DECISION_AMBIGUOUS: 1, "below_threshold": 2,
        },
        ambiguous=[{"doc_id": "train_00000", "sentence": "s", "reason": "tied"}],
    )
    build.records = [{
        "n_target_tokens_est": 10, "n_chunk_tokens_est": 100,
        "alignment_scores": [0.9], "section": "facts", "n_overflow_sentences": 0,
    }]
    stats = build_dataset.build_stats(
        mode="sample5", backend="CrossEncoder",
        alignment_kwargs_settings=EvidenceSettings(coverage_bound=0.6),
        builds=[build], records_by_split={"train": build.records},
        doc_ids_by_split={"train": {"train_00000"}},
        content_hash_by_split={"train": {"train_00000": "abc"}},
        split_isolation={"verified": True, "cross_split_content_duplicates": 0},
        length_validation={
            "ok": True, "max_input_tokens": 10, "max_target_tokens": 5,
            "input_violations": 0, "target_violations": 0,
        },
        args=args,
    )

    evidence = stats["evidence"]
    assert evidence["mode"] is True
    assert evidence["calibrated"] is False
    assert evidence["settings"]["coverage_bound"] == 0.6
    assert evidence["supported"] == 3 and evidence["ambiguous"] == 1
    assert evidence["supported_rate"] == 0.75
    assert evidence["ambiguous_sentences"] == 1
    assert "below_threshold" in evidence["decision_counts"]
    # The stats must be JSON-serialisable: the settings had to be converted from
    # the EvidenceSettings dataclass, or writing build_stats.json would crash.
    json.dumps(stats)

    with tempfile.TemporaryDirectory() as directory:
        build_dataset.write_manifest(Path(directory), args, stats, {"train": build.records}, "CrossEncoder")
        manifest = json.loads((Path(directory) / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["alignment_mode"] == "evidence"
    assert manifest["evidence_mode"] is True
    assert manifest["evidence_settings"]["coverage_bound"] == 0.6


def test_output_safety_gate_still_covers_the_ambiguous_report():
    import scripts.build_dataset as build_dataset

    assert "ambiguous.jsonl" in build_dataset.DATASET_OUTPUT_FILES
    assert "train.jsonl" in build_dataset.DATASET_OUTPUT_FILES

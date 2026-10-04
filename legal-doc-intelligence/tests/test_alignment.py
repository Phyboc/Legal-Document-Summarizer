"""Regression tests for primary-chunk alignment and target construction.

These cover the dataset-quality fix in ``src/data/section_summary_alignment.py``:
each reference-summary sentence now has at most ONE primary training chunk, so it
can never appear in more than one record, while its plausible section labels stay
available as metadata.

The alignment tests never load a model: ``align_summary_to_sections`` accepts
``precomputed`` scores, so the assignment logic is exercised deterministically and
offline. The tokenizer test uses a fake tokenizer for the same reason.

Run:  python -m pytest tests/test_alignment.py -q
"""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config import ALIGNMENT_RELATIVE_MARGIN, ALIGNMENT_THRESHOLD
from src.data.section_summary_alignment import (
    align_summary_to_sections,
    build_aligned_target,
    summary_sentences,
    validate_token_lengths,
)

THRESHOLD = ALIGNMENT_THRESHOLD
MARGIN = ALIGNMENT_RELATIVE_MARGIN

# Three chunks across three sections; index 0 facts, 1 judgement, 2 reasoning.
CHUNKS = [
    ("facts", "facts chunk text"),
    ("judgement", "judgement chunk text"),
    ("reasoning", "reasoning chunk text"),
]


def _align(sentences, per_label_scores, per_chunk_scores):
    """Run the aligner on precomputed scores, so no Cross-Encoder is needed."""
    return align_summary_to_sections(
        CHUNKS,
        "unused because precomputed scores are supplied",
        threshold=THRESHOLD,
        allow_multi_label=True,
        relative_margin=MARGIN,
        precomputed=(sentences, per_label_scores, per_chunk_scores),
    )


# --------------------------------------------------------------------------- #
# Test 1 / 7 / 8: one sentence, multiple plausible sections, one destination
# --------------------------------------------------------------------------- #

def test_multi_label_sentence_gets_exactly_one_primary_chunk():
    sentence = "The court considered the facts and ultimately dismissed the appeal."
    result = _align(
        [sentence],
        [{"facts": 0.80, "judgement": 0.75, "reasoning": 0.10}],
        [[0.70, 0.90, 0.10]],
    )
    record = result.per_sentence[0]

    assert set(record.assigned) == {"facts", "judgement"}, "both plausible labels kept"
    assert record.primary_chunk == 1, "highest-scoring candidate chunk wins"
    holders = [j for j, sents in result.chunk_sentences.items() if sentence in sents]
    assert holders == [1], "the sentence lives in exactly one chunk"


def test_multi_label_sentence_is_emitted_to_one_record_only():
    sentence = "The court considered the facts and ultimately dismissed the appeal."
    result = _align(
        [sentence],
        [{"facts": 0.80, "judgement": 0.75, "reasoning": 0.10}],
        [[0.70, 0.90, 0.10]],
    )

    assert len(result.chunk_sentences) == 1, "only one record is produced"

    # matched_labels is a union over the sentences attached to that one chunk.
    assigned_by_sentence = {r.sentence: r.assigned for r in result.per_sentence}
    matched = result.chunk_sentences[1]
    matched_labels = sorted({
        label for sent in matched for label in assigned_by_sentence[sent]
    })
    assert matched_labels == ["facts", "judgement"], "labels stay as metadata"


def test_no_sentence_is_written_to_more_than_one_chunk():
    sentences = ["A fact sentence.", "A reasoning sentence.", "A judgement sentence."]
    per_label = [
        {"facts": 0.9, "judgement": 0.1, "reasoning": 0.1},
        {"facts": 0.1, "judgement": 0.1, "reasoning": 0.9},
        {"facts": 0.1, "judgement": 0.9, "reasoning": 0.1},
    ]
    rows = [[0.9, 0.1, 0.1], [0.1, 0.1, 0.9], [0.1, 0.9, 0.1]]
    result = _align(sentences, per_label, rows)

    written = [s for sents in result.chunk_sentences.values() for s in sents]
    assert len(written) == len(set(written)) == len(sentences)


# --------------------------------------------------------------------------- #
# Test 2: a single clear section
# --------------------------------------------------------------------------- #

def test_single_label_sentence_selects_that_sections_chunk():
    result = _align(
        ["Pure facts sentence."],
        [{"facts": 0.90, "judgement": 0.10, "reasoning": 0.10}],
        [[0.90, 0.10, 0.10]],
    )
    record = result.per_sentence[0]
    assert record.assigned == ["facts"]
    assert record.primary_chunk == 0


# --------------------------------------------------------------------------- #
# Test 3: below threshold -> unassigned, no primary chunk
# --------------------------------------------------------------------------- #

def test_sentence_below_threshold_is_unassigned_with_no_primary_chunk():
    result = _align(
        ["An unrelated sentence."],
        [{"facts": 0.20, "judgement": 0.10, "reasoning": 0.05}],
        [[0.20, 0.10, 0.05]],
    )
    record = result.per_sentence[0]
    assert record.assigned == []
    assert record.primary_chunk is None
    assert result.chunk_sentences == {}
    assert len(result.unassigned) == 1


# --------------------------------------------------------------------------- #
# Test 4: several sentences land on the same chunk
# --------------------------------------------------------------------------- #

def test_multiple_sentences_can_share_one_chunk():
    sentences = ["First facts sentence.", "Second facts sentence."]
    result = _align(
        sentences,
        [{"facts": 0.90}, {"facts": 0.85}],
        [[0.90, 0.10, 0.10], [0.85, 0.10, 0.10]],
    )
    assert list(result.chunk_sentences) == [0]
    assert result.chunk_sentences[0] == sentences
    target = " ".join(result.chunk_sentences[0])
    assert target == "First facts sentence. Second facts sentence."


# --------------------------------------------------------------------------- #
# Test 5: short operative sentences must survive to scoring
# --------------------------------------------------------------------------- #

def test_short_operative_sentences_are_not_dropped():
    sentences = summary_sentences("The appeal is dismissed. Appeal dismissed.")
    assert "The appeal is dismissed." in sentences
    assert "Appeal dismissed." in sentences, "two-word order must be retained"


def test_cite_verify_short_sentence_drop_is_why_a_local_splitter_exists():
    # Documents the reason section_summary_alignment does not reuse the shared
    # helper: cite_verify.split_sentences drops < 3-word sentences.
    from src.postprocess.cite_verify import split_sentences

    assert "Appeal dismissed." not in split_sentences("Appeal dismissed.")


# --------------------------------------------------------------------------- #
# Test 6 / budget: target construction at whole-sentence boundaries
# --------------------------------------------------------------------------- #

def test_target_is_exactly_the_kept_sentences_joined():
    sentences = ["Alpha sentence one.", "Beta sentence two."]
    target, kept, overflow = build_aligned_target(sentences, max_tokens=None)
    assert target == " ".join(sentences)
    assert kept == sentences
    assert overflow == []


def test_target_budget_stops_at_a_sentence_boundary_and_reports_overflow():
    def measure(text):
        return len(text.split())

    target, kept, overflow = build_aligned_target(
        ["one two three.", "four five six."], max_tokens=4, measure=measure
    )
    assert target == "one two three."
    assert kept == ["one two three."]
    assert overflow == ["four five six."], "overflow is reported, never split"


# --------------------------------------------------------------------------- #
# Test 9: real-tokenizer validation reports, never truncates
# --------------------------------------------------------------------------- #

def test_validate_token_lengths_reports_over_budget_targets():
    def fake_tokenizer(text):
        # One "token" per word - deterministic and dependency-free.
        return {"input_ids": text.split()}

    records = [{
        "doc_id": "train_00000",
        "chunk_id": 0,
        "chunk_text": "a b c",
        "aligned_target": "one two three four five six seven",
    }]
    report = validate_token_lengths(
        records, fake_tokenizer, max_input_tokens=5, max_target_tokens=3
    )

    assert report["records"] == 1
    assert report["target_violations"] == 1
    assert report["input_violations"] == 0
    assert report["max_target_tokens"] == 7
    assert report["ok"] is False
    assert report["violation_examples"][0]["field"] == "aligned_target"


def test_validate_token_lengths_passes_within_limits():
    def fake_tokenizer(text):
        return {"input_ids": text.split()}

    records = [{
        "doc_id": "train_00000",
        "chunk_id": 0,
        "chunk_text": "a b c",
        "aligned_target": "one two",
    }]
    report = validate_token_lengths(
        records, fake_tokenizer, max_input_tokens=5, max_target_tokens=5
    )
    assert report["ok"] is True
    assert report["target_violations"] == 0


# --------------------------------------------------------------------------- #
# Test 10: the chunker is untouched by this change
# --------------------------------------------------------------------------- #

def test_chunker_still_bounds_and_preserves_text():
    from config import CHUNK_SIZE, MAX_INPUT_LENGTH
    from src.data.chunker import approximate_tokens, chunk_section
    from src.data.section_detector import Paragraph

    text = " ".join(f"w{i}." for i in range(4000))  # several thousand estimated tokens
    chunks = chunk_section(
        [Paragraph(index=0, text=text, section="facts", position=0.0, scores={})],
        "facts",
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= CHUNK_SIZE
        assert approximate_tokens(chunk.text) <= MAX_INPUT_LENGTH

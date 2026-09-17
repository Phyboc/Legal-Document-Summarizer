"""Regression tests for section-aware chunking (``src/data/chunker.py``).

The guarantee under test is the one the deleted ``tests/test_model_stages.py``
covered as ``test_chunks_stay_inside_the_bart_window``: no produced chunk exceeds
the chunking budget, so nothing is silently truncated at ``MAX_INPUT_LENGTH``.

These tests use the chunker's own ``approximate_tokens`` estimator, so they run
offline with no model download (the old suite went through the BART tokenizer,
which needs the Hub, and skipped itself when the model was unavailable).

Run:  python -m pytest tests/test_chunker.py -q
"""
import sys
from collections import Counter
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNKS_PER_DOC, MAX_INPUT_LENGTH
from src.data.chunker import (
    approximate_tokens,
    chunk_document,
    chunk_section,
    split_by_tokens,
)
from src.data.section_detector import Paragraph

# A word count small enough that a paragraph stays inside the budget.
SMALL_PARAGRAPH_WORDS = 10


def paragraph(index: int, text: str, section: str = "facts") -> Paragraph:
    """Build a ``Paragraph`` in the shape ``detect_sections`` produces."""
    return Paragraph(index=index, text=text, section=section, position=0.0, scores={})


def prose(n_sentences: int, words_per_sentence: int = 12) -> str:
    """Deterministic prose: ``n_sentences`` sentences of ``words_per_sentence`` words."""
    return " ".join(
        " ".join(f"w{i}x{j}" for j in range(words_per_sentence)) + "."
        for i in range(n_sentences)
    )


def wordless_prose(n_words: int) -> str:
    """A run-on with words but no sentence punctuation, to hit the hard-split path."""
    return " ".join(f"w{i}" for i in range(n_words))


def assert_all_chunks_bounded(chunks) -> None:
    """Every chunk must fit the chunking budget and, with margin, the model window."""
    for chunk in chunks:
        assert chunk.token_count <= CHUNK_SIZE, (
            f"chunk {chunk.chunk_id} carries {chunk.token_count} estimated tokens, "
            f"over the {CHUNK_SIZE}-token budget"
        )
        assert approximate_tokens(chunk.text) <= MAX_INPUT_LENGTH, (
            f"chunk {chunk.chunk_id} would be truncated by the model "
            f"({approximate_tokens(chunk.text)} > {MAX_INPUT_LENGTH})"
        )


# --------------------------------------------------------------------------- #
# The regression this file exists for
# --------------------------------------------------------------------------- #

def test_oversized_paragraph_becomes_multiple_bounded_chunks():
    """A several-thousand-token paragraph must split, not stay one oversized chunk."""
    text = prose(600)  # ~7,200 words -> ~9,500 estimated tokens
    total = approximate_tokens(text)

    assert total > 5 * CHUNK_SIZE, "fixture must be several thousand tokens"
    assert total > MAX_INPUT_LENGTH, "fixture must exceed the model window"

    # Before the fix this returned exactly one chunk carrying all `total` tokens
    # - i.e. ~9x the model window, of which only the first ~1024 ever reached the
    # summarizer.
    chunks = chunk_section([paragraph(0, text)], "facts")

    assert len(chunks) > 1, "an oversized paragraph must be split, not emitted whole"
    assert_all_chunks_bounded(chunks)

    # Every piece comes from the same source paragraph, so the metadata must not
    # invent a paragraph range that spans the whole document.
    assert {chunk.para_range for chunk in chunks} == {(0, 0)}

    # Nothing may be dropped: the union of the chunks covers every source word.
    produced = Counter(word for chunk in chunks for word in chunk.text.split())
    assert produced >= Counter(text.split()), "splitting must not truncate text"


def test_run_on_paragraph_without_sentence_breaks_is_still_bounded():
    """No sentence punctuation exercises ``split_by_tokens``' hard word-window path."""
    text = wordless_prose(4000)  # 4,000 words, no '.' anywhere
    assert approximate_tokens(text) > 5 * CHUNK_SIZE

    chunks = chunk_section([paragraph(0, text)], "facts")

    assert len(chunks) > 1
    assert_all_chunks_bounded(chunks)
    produced = Counter(word for chunk in chunks for word in chunk.text.split())
    assert produced >= Counter(text.split())


def test_oversized_paragraph_next_to_normal_ones_keeps_sections_intact():
    """Splitting must not corrupt chunk ids, sections, or the following paragraph."""
    groups = {
        "facts": [paragraph(0, prose(1, SMALL_PARAGRAPH_WORDS)), paragraph(1, prose(400))],
        "reasoning": [paragraph(2, prose(1, SMALL_PARAGRAPH_WORDS))],
    }
    chunks = chunk_section(groups["facts"], "facts", start_chunk_id=7)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == 7, "start_chunk_id must still be honoured"
    assert [chunk.chunk_id for chunk in chunks] == list(range(7, 7 + len(chunks)))
    assert all(chunk.section == "facts" for chunk in chunks)
    assert chunks[0].text.startswith(groups["facts"][0].text), "first paragraph must lead"
    assert_all_chunks_bounded(chunks)


# --------------------------------------------------------------------------- #
# Existing behaviour that must be preserved
# --------------------------------------------------------------------------- #

def test_paragraph_within_budget_is_not_split():
    """A paragraph already inside the budget is still emitted as a single chunk."""
    text = wordless_prose(601)  # ~799 estimated tokens
    assert approximate_tokens(text) <= CHUNK_SIZE

    chunks = chunk_section([paragraph(0, text)], "facts")

    assert len(chunks) == 1
    assert chunks[0].token_count == approximate_tokens(text)
    assert chunks[0].text == text


def test_in_budget_paragraphs_are_windowed_and_overlapped_as_before():
    """Small paragraphs window exactly as they did: same boundaries, overlap kept."""
    paragraphs = [paragraph(i, prose(1, SMALL_PARAGRAPH_WORDS)) for i in range(100)]
    chunks = chunk_section(paragraphs, "facts")

    assert len(chunks) == 2
    assert chunks[0].para_range == (0, 64)
    assert chunks[1].para_range == (65, 99)
    assert chunks[0].token_count == 845  # 65 paragraphs x 13 estimated tokens
    assert_all_chunks_bounded(chunks)

    # The overlap tail is still carried into the next window.
    assert chunks[1].text.startswith(chunks[0].text[-CHUNK_OVERLAP * 4:])


def test_para_range_tracks_the_paragraphs_actually_included():
    """Non-contiguous section indices must not be reported as a contiguous range.

    ``group_by_section`` produces sections whose paragraph indices are not
    adjacent (labels interleave), so "the paragraph before this one" is not
    ``index - 1``.
    """
    paragraphs = [paragraph(2 * i, prose(1, SMALL_PARAGRAPH_WORDS)) for i in range(100)]
    chunks = chunk_section(paragraphs, "facts")

    assert len(chunks) == 2
    assert chunks[0].para_range == (0, 128), "must end on the last paragraph included"
    assert chunks[1].para_range == (130, 198)


def test_empty_and_tiny_inputs_are_unchanged():
    assert chunk_section([], "facts") == []

    single = chunk_section([paragraph(0, "A short paragraph with a few words.")], "facts")
    assert len(single) == 1
    assert single[0].para_range == (0, 0)
    assert single[0].chunk_id == 0


# --------------------------------------------------------------------------- #
# split_by_tokens, now used by chunk_section
# --------------------------------------------------------------------------- #

def test_split_by_tokens_pieces_are_bounded_and_lossless():
    text = prose(500)
    pieces = split_by_tokens(text, CHUNK_SIZE)

    assert len(pieces) > 1
    assert all(approximate_tokens(piece) <= CHUNK_SIZE for piece in pieces)
    assert Counter(" ".join(pieces).split()) == Counter(text.split())


def test_split_by_tokens_hard_splits_an_unbreakable_sentence():
    text = wordless_prose(3000)
    pieces = split_by_tokens(text, CHUNK_SIZE)

    assert len(pieces) > 1
    assert all(approximate_tokens(piece) <= CHUNK_SIZE for piece in pieces)


# --------------------------------------------------------------------------- #
# Interaction with the 15-chunk document cap
# --------------------------------------------------------------------------- #

def test_exploded_paragraph_still_respects_the_chunk_cap():
    """More chunks from splitting must still be capped, and every section kept."""
    groups = {
        "facts": [paragraph(0, prose(1, SMALL_PARAGRAPH_WORDS))],
        "reasoning": [paragraph(1, prose(1000))],  # ~19 chunks once split
        "judgement": [paragraph(2, prose(1, SMALL_PARAGRAPH_WORDS))],
    }

    uncapped = chunk_section(groups["reasoning"], "reasoning")
    assert len(uncapped) > MAX_CHUNKS_PER_DOC, "fixture must exceed the cap on its own"
    assert_all_chunks_bounded(uncapped)

    capped = chunk_document(groups)

    assert len(capped) <= MAX_CHUNKS_PER_DOC
    assert_all_chunks_bounded(capped)
    assert {chunk.section for chunk in capped} == {"facts", "reasoning", "judgement"}

    # Capping selects a subset of the pre-cap chunks, so ids stay unique and in
    # document order but are not renumbered - a pre-existing chunk_document
    # behaviour, deliberately left as it was.
    ids = [chunk.chunk_id for chunk in capped]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    assert set(ids) <= set(range(len(uncapped) + 2))

"""Tests for encoder-windowed, prefilted evidence scoring.

The Cross-Encoder truncates its input (ms-marco MiniLM: 512 sub-word tokens) while
a chunk is budgeted at ~850 estimated tokens, so scoring a whole chunk hides most
of it and can hand a summary sentence to the wrong chunk. These tests pin the
replacement behaviour:

* a chunk is split into overlapping windows that reconstruct it exactly;
* each sentence is compared only against its top-k lexically ranked chunks, with
  ``top_k = 0`` meaning "every chunk" (no silent recall loss);
* the score a chunk receives is the best of its windows, and the winning window is
  reported so the matched span can be recorded.

Everything runs offline: ``_score_pairs`` is replaced with a deterministic stub, so
no model is downloaded and no GPU is required.

Run:  python -m pytest tests/test_evidence_windows.py -q
"""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.data import section_summary_alignment as sia  # noqa: E402
from src.data.chunker import approximate_tokens  # noqa: E402


def _paragraph(sentence_count: int, marker: str = "appellant") -> str:
    return " ".join(
        f"Sentence {i} states that the {marker} filed the petition in the year 1950."
        for i in range(sentence_count)
    )


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #

def test_windows_are_bounded_by_the_window_budget():
    text = _paragraph(80)
    windows = sia.chunk_windows(text, max_tokens=200, overlap_tokens=20)

    assert len(windows) > 1, "an over-long chunk must be split"
    for _, _, window in windows:
        # One sentence may overflow the budget, so allow a single-sentence slack.
        assert approximate_tokens(window) <= 200 + 80, approximate_tokens(window)


def test_windows_reconstruct_the_chunk_exactly():
    text = _paragraph(50)
    windows = sia.chunk_windows(text, max_tokens=200, overlap_tokens=20)

    for start, end, window in windows:
        assert text[start:end] == window, "offsets must slice the chunk exactly"

    # Every character is visible to the encoder in at least one window.
    covered = set()
    for start, end, _ in windows:
        covered.update(range(start, end))
    assert covered == set(range(len(text))), "no text may be hidden from the encoder"


def test_adjacent_windows_overlap():
    text = _paragraph(40)
    windows = sia.chunk_windows(text, max_tokens=150, overlap_tokens=40)

    assert len(windows) >= 2
    for previous, following in zip(windows, windows[1:]):
        assert following[0] < previous[1], "windows must overlap, not abut"


def test_short_chunk_stays_one_window():
    text = "The appeal is dismissed with costs."
    windows = sia.chunk_windows(text, max_tokens=350, overlap_tokens=50)
    assert len(windows) == 1
    assert windows[0][2] == text


# --------------------------------------------------------------------------- #
# Candidate prefilter
# --------------------------------------------------------------------------- #

CHUNKS = [
    "The appeal is dismissed with costs.",
    "The appellant filed a petition in the High Court in 1950 seeking relief.",
    "The court held that section 4 of the Act is ultra vires and void.",
]


def test_prefilter_ranks_the_lexically_relevant_chunk_first():
    ranked = sia.rank_candidate_chunks(
        "The petition was filed in the High Court in 1950.", CHUNKS, top_k=1
    )
    assert ranked == [1]


def test_prefilter_is_deterministic_and_ordered():
    first = sia.rank_candidate_chunks("petition High Court 1950", CHUNKS, top_k=2)
    second = sia.rank_candidate_chunks("petition High Court 1950", CHUNKS, top_k=2)
    assert first == second
    assert first == sorted(first)


def test_top_k_zero_means_every_chunk_is_a_candidate():
    assert sia.rank_candidate_chunks("anything at all", CHUNKS, top_k=0) == [0, 1, 2]


def test_unrelated_sentence_still_gets_candidates():
    # The prefilter decides what is *scored*, never what is assigned: a sentence
    # sharing no content word must still reach the encoder.
    assert len(sia.rank_candidate_chunks("zqxwv utterly unrelated", CHUNKS, top_k=2)) == 2


def test_every_section_stays_scoreable_under_the_prefilter():
    # Six chunks over three sections, top_k=1: a purely lexical top-1 would drop two
    # whole sections and silently remove their labels from this sentence's
    # possibilities. Section coverage must keep one chunk per section.
    labelled_chunks = [
        ("facts", "The appellant filed a petition in the High Court in 1950."),
        ("facts", "The respondent denies that any petition was ever filed."),
        ("reasoning", "The court considered whether section 4 of the Act is valid."),
        ("reasoning", "We are of the view that the earlier judgment applies here."),
        ("judgement", "The appeal is dismissed with costs."),
        ("judgement", "In the result the petition is allowed in part."),
    ]
    labels = [label for label, _ in labelled_chunks]
    texts = [text for _, text in labelled_chunks]

    ranked = sia.rank_candidate_chunks(
        "The appeal is dismissed with costs.", texts, top_k=1, section_labels=labels
    )
    assert {labels[index] for index in ranked} == set(labels)
    assert len(ranked) < len(texts), "the prefilter must still cut the candidate set"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def test_scoring_uses_windows_and_reports_the_winning_one(monkeypatch):
    chunk = _paragraph(60, marker="respondent")
    marker = "Sentence 55 states that the respondent filed the petition in the year 1950."
    assert marker in chunk, "the marker sentence must exist in the fixture"

    def fake_score(pairs):
        # 1.0 only for the window that contains the marker sentence.
        return [1.0 if marker in window else 0.1 for _, window in pairs]

    monkeypatch.setattr(sia, "_score_pairs", fake_score)

    sentence = "The respondent filed the petition in the year 1950."
    sentences, per_label, per_chunk, window_hits, pairs = sia.score_summary_sentences(
        [("reasoning", chunk)], sentence, window_tokens=200, overlap_tokens=20
    )

    assert sentences == [sentence]
    assert per_chunk[0][0] == 1.0, "the chunk takes its best window's score"
    hit = window_hits[0][0]
    assert hit["char_start"] < len(chunk) and hit["char_end"] <= len(chunk)
    assert marker in chunk[hit["char_start"]:hit["char_end"]]
    assert pairs > 1, "the whole chunk must have been split into more than one window"


def test_prefilter_reduces_the_number_of_scored_pairs(monkeypatch):
    # One section, so section coverage adds nothing and the reduction is purely the
    # effect of top_k.
    chunk_texts = [("reasoning", _paragraph(30, marker=f"party{i}")) for i in range(8)]
    seen = {"pairs": 0}

    def fake_score(pairs):
        seen["pairs"] += len(pairs)
        return [0.5 for _ in pairs]

    monkeypatch.setattr(sia, "_score_pairs", fake_score)
    sentence = "The party3 filed the petition in the year 1950."

    _, _, _, _, prefilted = sia.score_summary_sentences(
        chunk_texts, sentence, window_tokens=200, top_k_candidates=2
    )
    seen["pairs"] = 0
    _, _, _, _, everything = sia.score_summary_sentences(
        chunk_texts, sentence, window_tokens=200, top_k_candidates=0
    )

    assert prefilted < everything, "the prefilter must reduce the encoder's work"
    assert prefilted > 0


def test_alignment_attaches_the_evidence_span():
    chunk = _paragraph(30, marker="appellant")
    sentence = "The appellant filed the petition in the year 1950."

    def fake_score(pairs):
        return [0.9 for _ in pairs]

    original = sia._score_pairs
    sia._score_pairs = fake_score
    try:
        result = sia.align_summary_to_sections(
            [("reasoning", chunk)], sentence, window_tokens=200, top_k_candidates=0
        )
    finally:
        sia._score_pairs = original

    record = result.per_sentence[0]
    assert record.evidence_span, "a matched sentence must carry its evidence span"
    location = record.evidence_location
    assert location["chunk_index"] == record.primary_chunk
    assert chunk[location["char_start"]:location["char_end"]] == record.evidence_span
    assert result.pairs_scored > 0, "the real comparison count must be reported"

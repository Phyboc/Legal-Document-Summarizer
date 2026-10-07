"""Tests for Cite & Verify (``src/postprocess/cite_verify.py``).

Both models are replaced with deterministic fakes, so the tests run offline:
relevance = word overlap, entailment = whether every content word of the
sentence appears in the window. That is enough to pin the wiring - retrieval
picks the citation, NLI decides support, top-k limits the NLI work, and the
fallback is visible - without depending on model weights.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.postprocess.cite_verify as cv
from config import CITATION_THRESHOLD


def _words(text):
    return {w.strip(".,").lower() for w in text.split() if len(w) > 3}


class FakeRelevance:
    def score(self, pairs, normalize=True):
        return [len(_words(q) & _words(p)) / max(1, len(_words(q))) for q, p in pairs]


@pytest.fixture
def fake_models(monkeypatch):
    calls = []

    def fake_entailment(pairs, batch_size=32):
        calls.extend(pairs)
        out = []
        for window, sentence in pairs:
            supported = _words(sentence) <= _words(window)
            negated = " not " in f" {sentence} " and " not " not in f" {window} "
            out.append({
                "entailment": 0.95 if supported and not negated else 0.05,
                "contradiction": 0.9 if negated else 0.05,
                "neutral": 0.05,
            })
        return out

    monkeypatch.setattr(cv, "get_cross_encoder", lambda: FakeRelevance())
    monkeypatch.setattr(cv, "get_nli_model", lambda: object())
    monkeypatch.setattr(cv, "entailment_scores", fake_entailment)
    return calls


CHUNKS = [
    {"chunk_id": 0, "section": "facts", "para_range": (0, 2),
     "source_text": "The appellant purchased the land in 1975. The respondent disputed the sale deed."},
    {"chunk_id": 1, "section": "judgement", "para_range": (9, 10),
     "source_text": "The appeal is dismissed with costs. The order of the High Court is upheld."},
]


def test_supported_sentence_is_verified_and_cited_to_its_chunk(fake_models):
    [c] = cv.cite_and_verify("The appeal is dismissed with costs.", CHUNKS)
    assert c["verified"] and c["support"] == "supported"
    assert c["source_chunk_id"] == 1 and c["source_section"] == "judgement"
    assert c["method"] == "nli"


def test_relevant_but_unsupported_sentence_is_not_verified(fake_models):
    # Topically identical to chunk 1 - exactly what MS-MARCO used to pass.
    [c] = cv.cite_and_verify("The appeal is dismissed and compensation of Rs. 50,000 awarded.", CHUNKS)
    assert c["source_chunk_id"] == 1
    assert not c["verified"]
    assert c["support"] == "unsupported"


def test_negated_claim_is_flagged_as_contradicted(fake_models):
    [c] = cv.cite_and_verify("The appeal is not dismissed with costs.", CHUNKS)
    assert not c["verified"]
    assert c["support"] == "contradicted"


def test_nli_only_runs_on_top_k_chunks(fake_models, monkeypatch):
    monkeypatch.setattr(cv, "CITATION_TOP_K", 1)
    many = CHUNKS + [{"chunk_id": i, "section": "reasoning", "para_range": (i, i),
                      "source_text": f"Unrelated paragraph number {i} about procedure."} for i in range(2, 8)]
    checks = cv.verify_sentences(["The appeal is dismissed with costs."], [c["source_text"] for c in many], top_k=1)
    windows = {w for w, _ in fake_models}
    assert checks[0]["source_index"] == 1
    assert all("Unrelated" not in w for w in windows), "NLI must not score non-retrieved chunks"


def test_falls_back_to_relevance_visibly_without_nli(monkeypatch):
    monkeypatch.setattr(cv, "get_cross_encoder", lambda: FakeRelevance())
    monkeypatch.setattr(cv, "get_nli_model", lambda: None)
    [c] = cv.cite_and_verify("The appeal is dismissed with costs.", CHUNKS)
    assert c["method"] == "relevance"
    assert cv.summarize_verification([c])["method"] == "relevance"


def test_windows_cover_the_whole_source_with_overlap():
    text = " ".join(f"Sentence {i} has exactly six words." for i in range(40))
    windows = cv.source_windows(text, max_words=30)
    assert windows[0].startswith("Sentence 0 ") and "Sentence 39 " in windows[-1]
    assert all(len(w.split()) <= 30 for w in windows)
    # consecutive windows share at least one sentence
    for a, b in zip(windows, windows[1:]):
        assert set(a.split(". ")) & set(b.split(". "))


def test_summary_statistics_count_contradictions(fake_models):
    citations = cv.cite_and_verify(
        "The appeal is dismissed with costs. The appeal is not dismissed with costs.", CHUNKS
    )
    stats = cv.summarize_verification(citations)
    assert stats["total"] == 2 and stats["verified"] == 1 and stats["contradicted"] == 1
    assert 0 < CITATION_THRESHOLD < 1


def test_verbatim_sentence_is_supported_without_nli(fake_models):
    [c] = cv.cite_and_verify("The respondent disputed the sale deed.", CHUNKS)
    assert c["evidence"] == "verbatim" and c["verified"] and c["source_chunk_id"] == 0
    assert not fake_models, "a verbatim match must not spend NLI calls"


def test_one_changed_word_is_not_verbatim():
    sources = [CHUNKS[1]["source_text"]]
    assert cv.verbatim_support("The appeal is dismissed with costs.", sources) == 0
    for altered in ("The appeal is allowed with costs.", "The appeal is not dismissed with costs."):
        assert cv.verbatim_support(altered, sources) is None


def test_compressed_copy_is_supported_but_changes_are_not():
    source = ["as required by r. 2(b) of the Kerala State Higher Judicial Service Rules, 1961 framed under "
              "article 309. The petition was not maintained by the petitioner before the court."]
    # a dropped article is still a copy
    assert cv.verbatim_support(
        "as required by r. 2(b) of Kerala State Higher Judicial Service Rules, 1961.", source) == 0
    for altered in (
        "as required by r. 2(b) of the Kerala State Higher Judicial Service Rules, 1965.",  # changed number
        "as required by r. 2(b) of the Kerala State Lower Judicial Service Rules, 1961.",   # changed word
        "The petition was maintained by the petitioner before the court.",                 # dropped "not"
    ):
        assert cv.verbatim_support(altered, source) is None, altered


def test_split_sentences_keeps_abbreviations_inside_sentences():
    assert cv.split_sentences("The fine was Rs. 50 on each appellant. Appeal is dismissed.") == [
        "The fine was Rs. 50 on each appellant.", "Appeal is dismissed.",
    ]


def test_section_headings_are_not_glued_onto_sentences(fake_models):
    text = "FACTS:\nThe respondent disputed the sale deed.\n\nThe Court's Decision:\nThe appeal is dismissed with costs."
    citations = cv.cite_and_verify(text, CHUNKS)
    assert [c["sentence"] for c in citations] == [
        "The respondent disputed the sale deed.", "The appeal is dismissed with costs.",
    ]
    assert all(c["verified"] for c in citations)

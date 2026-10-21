"""Regression tests for what ``build_document`` writes into a training record.

Two things are pinned here, both of them consequences of a real build failure:

* the auditable evidence span - every record carries the exact window of its chunk
  that the match rests on, with a chunk-relative character range, so a reviewer can
  check the alignment without reading a 600-word chunk;
* the target sanity gate - a target longer than its own chunk, or a target with no
  prose at all, is dropped and counted rather than written out as supervision.

The Cross-Encoder is replaced with a hand-built ``AlignmentResult``, so these tests
need no model, no tokenizer download and no network.

Run:  python -m pytest tests/test_build_records.py -q
"""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _PKG_ROOT / "scripts"
for _path in (str(_PKG_ROOT), str(_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_dataset as bd  # noqa: E402
from src.data.section_summary_alignment import (  # noqa: E402
    AlignmentResult,
    SentenceAlignment,
)

DOCUMENT = (
    "The appellant filed a petition in the High Court on 3 January 1950. "
    "The High Court dismissed it.\n\n"
    "The learned counsel for the appellant submitted that section 4 of the Act "
    "is ultra vires. We have considered the submissions.\n\n"
    "In the result the appeal is dismissed with costs."
)
SUMMARY = "The appeal is dismissed with costs."


def _result(sentences, span_map=None, section="reasoning"):
    """Build an ``AlignmentResult`` as if the aligner had produced it."""
    per_sentence = []
    chunk_sentences: dict[int, list[str]] = {}
    chunk_pair_scores: dict[int, list[float]] = {}
    for index, sentence in enumerate(sentences):
        record = SentenceAlignment(sentence=sentence, scores={section: 0.9})
        record.assigned = [section]
        record.primary_chunk = 0
        record.primary_chunk_score = 0.9
        if span_map:
            record.evidence_span = span_map.get(sentence)
            record.evidence_location = {
                "chunk_index": 0,
                "section": section,
                "window_index": 0,
                "char_start": 0,
                "char_end": len(record.evidence_span or ""),
                "score": 0.9,
            }
        per_sentence.append(record)
        chunk_sentences.setdefault(0, []).append(sentence)
        chunk_pair_scores.setdefault(0, []).append(0.9)
    return AlignmentResult(
        aligned={section: " ".join(sentences)},
        aligned_sentences={section: list(sentences)},
        aligned_by_chunk={0: " ".join(sentences)},
        chunk_sentences=chunk_sentences,
        chunk_pair_scores=chunk_pair_scores,
        per_sentence=per_sentence,
        threshold=0.4,
        multi_label=True,
        pairs_scored=len(sentences) * 2,
    )


def _build(monkeypatch, result):
    """Run ``build_document`` against a hand-built alignment result.

    ``result`` may also be a callable taking the document's ``section_chunks``,
    which lets a test derive a self-consistent evidence span from the real chunk
    text instead of inventing one.
    """
    monkeypatch.setattr(
        bd,
        "align_summary_to_sections",
        (lambda *a, **k: result) if not callable(result) else (lambda s, *a, **k: result(s)),
    )
    return bd.build_document(
        "train_00000",
        "train",
        DOCUMENT,
        SUMMARY,
        token_measure=lambda text: len(text.split()),
        alignment_kwargs={},
    )


def test_record_carries_the_evidence_span_and_its_location(monkeypatch):
    sentence = "In the result the appeal is dismissed with costs."

    def fake_alignment(section_chunks, *args, **kwargs):
        # Derive the span from the real chunk text: the location a build records
        # must slice exactly the span the match rests on.
        span = section_chunks[0][1][:40]
        return _result([sentence], span_map={sentence: span}, section=section_chunks[0][0])

    build = _build(monkeypatch, fake_alignment)

    assert len(build.records) == 1, build.records
    record = build.records[0]
    spans = record["matched_evidence_spans"]
    assert len(spans) == 1
    location = spans[0]["location"]
    assert location["chunk_index"] == record["chunk_id"]
    assert location["section"] == record["section"]
    assert record["chunk_text"][location["char_start"]:location["char_end"]] == spans[0]["span"]
    assert build.pairs_scored == 2


def test_target_longer_than_its_chunk_is_dropped_and_counted(monkeypatch):
    # Seven words of target against a chunk forced to one word: impossible.
    long_sentence = "one two three four five six seven"
    monkeypatch.setattr(bd, "chunk_document", lambda groups: [])
    build = _build(monkeypatch, _result([long_sentence]))

    # No chunks means no records at all; the gate must not invent one.
    assert build.records == []

    # Now the realistic shape: a chunk exists but is shorter than the target.
    from src.data.chunker import Chunk

    chunk = Chunk(text="short chunk", section="reasoning", chunk_id=0,
                  token_count=2, para_range=(0, 0))
    monkeypatch.setattr(bd, "chunk_document", lambda groups: [chunk])
    build = _build(monkeypatch, _result([long_sentence]))
    assert build.records == []
    assert build.degenerate_targets_dropped == 1
    assert build.degenerate_target_details[0]["chunk_id"] == 0


def test_target_with_no_prose_is_dropped(monkeypatch):
    from src.data.chunker import Chunk

    chunk = Chunk(
        text="In the premises we are of the view that the High Court was right and the "
             "appeals must fail and are accordingly dismissed with costs.",
        section="judgement", chunk_id=0, token_count=25, para_range=(0, 0),
    )
    monkeypatch.setattr(bd, "chunk_document", lambda groups: [chunk])
    build = _build(monkeypatch, _result(["M.R. Ry."]))

    assert build.records == []
    assert build.degenerate_targets_dropped == 1


def test_valid_short_operative_order_is_kept(monkeypatch):
    from src.data.chunker import Chunk

    chunk = Chunk(
        text="In the premises the appeals must fail and are accordingly dismissed "
             "with costs.",
        section="judgement", chunk_id=0, token_count=14, para_range=(0, 0),
    )
    monkeypatch.setattr(bd, "chunk_document", lambda groups: [chunk])
    build = _build(monkeypatch, _result(["Appeal dismissed."]))

    assert len(build.records) == 1
    assert build.records[0]["aligned_target"] == "Appeal dismissed."
    assert build.degenerate_targets_dropped == 0

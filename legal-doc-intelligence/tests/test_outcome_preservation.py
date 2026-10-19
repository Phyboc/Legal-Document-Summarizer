"""Regression tests for operative-order (final disposition) preservation.

The dataset audit (scripts/audit_outcome_survival.py) found that a final order
such as "The appeal is allowed." is retained in the training data but is routed by
relevance alone, so most orders landed in facts or reasoning chunks rather than
the judgement section that inference trusts. These tests pin the fixes:

  * the detector recognises the forms it used to miss, and does not mistake a
    lower court's history for this court's order;
  * an operative-order sentence is routed to a judgement chunk when the source
    supports it (judgement chunk above threshold), and only then;
  * training and inference group paragraphs with the same rule (build_section_groups);
  * an order that does not fit the target budget is reported as overflow, never
    dropped silently.

Everything runs offline: alignment is driven by ``precomputed`` scores, so no
Cross-Encoder is loaded.

Run:  python -m pytest tests/test_outcome_preservation.py -q
"""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config import ALIGNMENT_THRESHOLD
from src.data.section_detector import (
    Paragraph,
    build_section_groups,
    is_disposition,
    is_lower_court_history,
    is_operative_order,
)
from src.data.section_summary_alignment import align_summary_to_sections, build_aligned_target

HIGH = 0.95  # comfortably above the alignment threshold
LOW = 0.05   # comfortably below it


def _align(sections, sentence_scores, texts):
    """Align one summary sentence per row of ``sentence_scores``.

    ``sections`` is ``[(label, chunk_text), ...]``. ``sentence_scores[i][j]`` is the
    raw score of sentence ``i`` against chunk ``j``; per-label scores are the best
    over that label's chunks, exactly as ``score_summary_sentences`` computes them.
    ``texts`` are the summary sentences themselves (the routing rule reads them).
    """
    sentences = list(texts)
    per_label_rows = []
    for row in sentence_scores:
        best = {}
        for (label, _), score in zip(sections, row):
            best[label] = max(best.get(label, -1.0), score)
        per_label_rows.append(best)
    return sentences, align_summary_to_sections(
        sections, "",
        precomputed=(sentences, per_label_rows, [list(r) for r in sentence_scores]),
        is_operative=is_operative_order,
    )


# ---------------------------------------------------------------- detector ----

def test_short_final_orders_are_operative():
    assert is_operative_order("The appeal is dismissed.")
    assert is_operative_order("The appeal is allowed.")


def test_forms_the_old_detector_missed_are_now_recognised():
    assert is_operative_order("The matter is remanded.")
    assert is_operative_order("The order is set aside.")
    assert is_operative_order("The impugned judgment is modified.")
    assert is_operative_order("The petition is disposed of.")
    assert is_operative_order("The appeal must accordingly be allowed.")
    assert is_operative_order(
        "The 117 ejectment decree cannot therefore, be allowed to stand and is accordingly set aside."
    )
    assert is_operative_order("The levy of cess is declared invalid.")
    assert is_operative_order("We dismissed the appeal.")


def test_negated_and_qualified_orders_are_operative():
    assert is_operative_order("The appeal is not allowed.")
    assert is_operative_order("The appeal is partly allowed and the decree is modified.")


def test_permission_is_not_an_order():
    assert not is_disposition("The petition is allowed to be withdrawn.")


def test_lower_court_dismissal_is_history_not_final_order():
    sentence = "The High Court dismissed the petition."
    assert is_lower_court_history(sentence)
    assert not is_operative_order(sentence)


def test_lower_court_order_that_this_court_sets_aside_is_final():
    # The judgment named here is the one under appeal; the sentence states this
    # court's own order, so it must not be filed as history.
    assert not is_lower_court_history("The judgment of the High Court is set aside.")
    assert is_operative_order("The judgment of the High Court is set aside.")


def test_factual_history_with_a_dismissal_is_not_operative():
    assert not is_operative_order("The suit was dismissed by the trial court.")
    assert not is_operative_order("The appeal was allowed by the High Court in 1985.")


def test_no_identifiable_disposition_is_not_operative():
    assert not is_operative_order("The court heard the parties at length on the question of limitation.")
    assert not is_operative_order("An election petition cannot be dismissed for default.")
    assert not is_operative_order("The petition was not liable to be dismissed.")


def test_ocr_corrupted_order_is_conservatively_not_operative():
    # A garbled order must not be promoted into a final order on a guess.
    assert not is_operative_order("The appeal is dlsmissed.")
    assert not is_operative_order("The appeal is al- lowed by the Bench in part.")


# ---------------------------------------------------------- alignment ----

def test_short_final_order_routes_to_judgement_chunk_over_higher_scoring_facts():
    sections = [
        ("facts", "On 12 March the appellant was arrested and charged with theft."),
        ("reasoning", "The witnesses were consistent on the timing of the offence."),
        ("judgement", "The appeal is dismissed."),
    ]
    # The order restates facts, so the facts chunk scores highest on relevance.
    _, result = _align(sections, [[0.99, 0.60, 0.90]], ["The appeal is dismissed."])
    record = result.per_sentence[0]
    assert record.operative
    assert "judgement" in record.assigned
    assert record.primary_chunk == 2
    assert result.chunk_sentences == {2: ["The appeal is dismissed."]}


def test_operative_order_after_a_long_explanation_still_reaches_judgement():
    long_reasoning = " ".join(["The Court considered the statute and the evidence in detail."] * 40)
    sections = [("reasoning", long_reasoning), ("judgement", "The appeal is allowed.")]
    _, result = _align(sections, [[0.97, 0.93]], ["The appeal is allowed."])
    assert result.per_sentence[0].primary_chunk == 1


def test_order_routing_respects_threshold_when_judgement_is_unsupported():
    # No judgement chunk clears the threshold: the source does not support the order
    # there, so it must not be attached to the judgement section.
    sections = [("facts", "The appellant was arrested."), ("judgement", "Costs follow the event.")]
    _, result = _align(sections, [[HIGH, LOW]], ["The appeal is allowed."])
    record = result.per_sentence[0]
    assert not record.operative
    assert "judgement" not in record.assigned


def test_lower_court_history_is_not_routed_to_judgement():
    sections = [("facts", "The petition was filed in 1990."), ("judgement", "The appeal is allowed.")]
    sentences = ["The High Court dismissed the petition.", "The appeal is allowed."]
    result = align_summary_to_sections(
        sections, "",
        precomputed=(sentences, [{"facts": HIGH, "judgement": HIGH}, {"facts": LOW, "judgement": HIGH}],
                     [[HIGH, HIGH], [LOW, HIGH]]),
        is_operative=is_operative_order,
    )
    history, final = result.per_sentence
    assert not history.operative
    assert final.operative and final.primary_chunk == 1


def test_multiple_matters_with_different_outcomes_each_reach_judgement():
    sections = [
        ("reasoning", "The Court examined the civil appeal and the writ petition."),
        ("judgement", "The appeal is allowed. The writ petition is dismissed."),
        ("judgement", "No order as to costs."),
    ]
    scores = [[0.97, HIGH, 0.6], [0.98, HIGH, 0.7]]
    _, result = _align(sections, scores, ["The appeal is allowed.", "The writ petition is dismissed."])
    assert all(r.operative for r in result.per_sentence)
    assert {r.primary_chunk for r in result.per_sentence} <= {1, 2}


def test_outcome_near_a_chunk_boundary_is_still_routed():
    sections = [
        ("judgement", "We have heard counsel."),
        ("reasoning", "The first half of the reasoning ends here."),
        ("judgement", "The appeal is dismissed."),
    ]
    _, result = _align(sections, [[0.4, 0.99, 0.5]], ["The appeal is dismissed."])
    assert result.per_sentence[0].primary_chunk == 2


def test_outcome_phrase_only_in_factual_history_is_not_attached_to_judgement():
    sections = [
        ("facts", "The earlier writ was dismissed by the High Court on 3 May 1985."),
        ("judgement", "The appeal is allowed."),
    ]
    sentences = ["The earlier writ was dismissed by the High Court on 3 May 1985."]
    result = align_summary_to_sections(
        sections, "",
        precomputed=(sentences, [{"facts": HIGH, "judgement": 0.5}], [[HIGH, 0.5]]),
        is_operative=is_operative_order,
    )
    record = result.per_sentence[0]
    assert not record.operative
    assert record.primary_chunk == 0


def test_every_sentence_has_at_most_one_primary_chunk_after_routing():
    sections = [("facts", "a"), ("judgement", "b"), ("judgement", "c")]
    _, result = _align(sections, [[0.9, 0.95, 0.9], [0.99, 0.2, 0.3]], ["Sentence a.", "Sentence b."])
    for record in result.per_sentence:
        assert record.primary_chunk is None or record.primary_chunk in range(len(sections))
    assert sum(len(v) for v in result.chunk_sentences.values()) == len(result.per_sentence)


# ------------------------------------------------------------- budget ----

def test_order_that_overflows_the_target_budget_is_reported_not_dropped():
    sentences = ["The appeal is allowed.", "x " * 200 + "end."]
    target, kept, overflow = build_aligned_target(
        sentences, max_tokens=50, measure=lambda text: len(text.split())
    )
    assert "The appeal is allowed." in kept
    assert overflow == [sentences[1]]
    assert target == "The appeal is allowed."


# ------------------------------------------------------- train/inference parity ----

def _para(index, text, section, position):
    return Paragraph(index=index, text=text, section=section, position=position, scores={})


def test_training_and_inference_share_judgement_promotion():
    paragraphs = [
        _para(0, "The appellant was arrested.", "facts", 0.0),
        _para(1, "We have considered the evidence.", "reasoning", 0.5),
        _para(2, "The appeal is allowed and the order is set aside.", "reasoning", 0.9),
    ]
    groups, fallback = build_section_groups(paragraphs)
    assert not fallback
    assert [p.index for p in groups["judgement"]] == [2]
    assert paragraphs[2].section == "judgement"


def test_fallback_keeps_operative_order_in_its_own_judgement_group():
    # Detection gave one dominant section, so the document is grouped in two.
    paragraphs = [_para(i, "The Court considered the matter.", "reasoning", i / 19) for i in range(19)]
    paragraphs.append(_para(19, "The appeal is dismissed.", "reasoning", 1.0))
    groups, fallback = build_section_groups(paragraphs)
    assert fallback
    assert [p.index for p in groups["judgement"]] == [19]
    assert len(groups["reasoning"]) == 19


def test_alignment_threshold_constant_is_the_configured_one():
    # The routing rule relies on the same threshold the alignment uses everywhere.
    assert 0.0 < ALIGNMENT_THRESHOLD < 1.0

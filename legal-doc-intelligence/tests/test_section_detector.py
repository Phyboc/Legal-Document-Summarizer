import csv
from pathlib import Path

import pytest

from src.data import section_detector as sd

ROOT = Path(__file__).resolve().parents[1]


def test_empty_input():
    assert sd.detect_sections([]) == []


def test_rule_fallback_defaults_to_reasoning_mid_document():
    label, _ = sd.classify_paragraph("Nothing distinctive appears in this sentence.", 0.5)
    assert label == "reasoning"


def test_rule_fallback_operative_order_at_end():
    label, _ = sd.classify_paragraph("In the result, the appeal is allowed with costs.", 0.97)
    assert label == "judgement"


def test_learned_model_labels_are_valid():
    if not sd.MODEL_PATH.exists():
        pytest.skip("classifier not trained")
    paras = [
        "The appellant was appointed as a clerk on 3 March 1990.",
        "Learned counsel for the respondent submitted that the appeal is barred by limitation.",
        "In our view the High Court erred in construing Section 5 of the Limitation Act.",
        "The appeal is allowed and the impugned order is set aside.",
    ]
    out = sd.detect_sections(paras)
    assert {p.section for p in out} <= set(sd.SECTION_PATTERNS)
    assert out[-1].section == "judgement"
    assert out[2].section == "reasoning"


def test_learned_model_beats_rules_on_hand_labelled_gold():
    a, c = ROOT / "review/section_labels.a.csv", ROOT / "review/section_labels.claude.csv"
    if not (sd.MODEL_PATH.exists() and a.exists() and c.exists()):
        pytest.skip("model or gold sheets missing")
    A = list(csv.DictReader(open(a, encoding="utf-8")))
    C = list(csv.DictReader(open(c, encoding="utf-8")))
    gold = [(x["text"], float(x["position"]), x["gold_label"]) for x, y in zip(A, C)
            if y["gold_label"] and x["gold_label"] == y["gold_label"] and x["gold_label"] != "issues"]
    norm = lambda l: "arguments" if l.startswith("arguments") else l
    learned = sd._predict_learned([g[0] for g in gold], [g[1] for g in gold])
    acc_l = sum(norm(p) == norm(g[2]) for p, g in zip(learned, gold)) / len(gold)
    acc_r = sum(norm(sd.classify_paragraph(t, p)[0]) == norm(l) for t, p, l in gold) / len(gold)
    assert acc_l >= 0.70 and acc_l > acc_r

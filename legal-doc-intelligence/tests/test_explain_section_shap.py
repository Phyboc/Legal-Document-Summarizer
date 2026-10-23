"""Offline tests for the SHAP section-classifier explanations."""
import json

import pytest

from src.data.section_detector import _load_model
from src.explain import section_shap

pytestmark = pytest.mark.skipif(
    _load_model() is None, reason="section_classifier.joblib not present"
)

shap_missing = pytest.importorskip("shap", reason="shap not installed") is None

_PARAS = [
    "The appellant was employed as an Assistant Engineer and was placed under "
    "suspension on September 3, 1959, pending a departmental inquiry.",
    "Learned counsel for the appellant contended that the suspension was "
    "wholly unjustified and violated the service rules.",
    "The respondent State submitted that the suspension was necessary during "
    "the pendency of the departmental inquiry.",
    "The order of compulsory retirement is set aside and the appeal is allowed "
    "with costs.",
]


def test_empty_input_is_handled():
    out = section_shap.explain_sections([])
    assert out["n_paragraphs"] == 0
    assert out["paragraphs"] == []
    assert "classes" in out


def test_explanation_shape_and_json_serialisable():
    out = section_shap.explain_sections(_PARAS, top_k=5)
    assert out["n_paragraphs"] == len(_PARAS)
    assert out["method"] == "shap.LinearExplainer"
    assert len(out["paragraphs"]) == len(_PARAS)

    for entry, text in zip(out["paragraphs"], _PARAS):
        assert entry["predicted_section"] in out["classes"]
        # probabilities cover every class and sum to roughly 1
        assert set(entry["probabilities"]) == set(out["classes"])
        assert abs(sum(entry["probabilities"].values()) - 1.0) < 1e-3
        assert 0 < len(entry["top_features"]) <= 5
        for feat in entry["top_features"]:
            assert feat["direction"] in ("supports", "opposes")

    # the whole structure must round-trip through JSON for the download
    json.loads(json.dumps(out))


def test_predicted_section_matches_the_classifier():
    """SHAP must explain the label the classifier actually assigns."""
    from src.data.section_detector import detect_sections

    detected = detect_sections(_PARAS)
    out = section_shap.explain_sections(_PARAS)
    for para, entry in zip(detected, out["paragraphs"]):
        assert entry["predicted_section"] == para.section

"""SHAP explanations for the section classifier.

The section classifier (``models/section_classifier.joblib``) is a
``(TfidfVectorizer, LogisticRegression)`` pair that labels each paragraph as
facts / arguments_appellant / arguments_respondent / reasoning / judgement.
Because it is a *linear* model, SHAP values are exact and cheap: this module
runs ``shap.LinearExplainer`` over the same feature matrix the detector uses
(TF-IDF word/bigram features plus the four position features) and reports, for
each paragraph, which terms pushed the model toward the label it chose.

The result is JSON-serialisable so the Gradio UI can offer it as a download.

This is an explanation of an existing model. Nothing is trained here, and the
SHAP values describe the classifier's own weights, not legal ground truth.
"""
from __future__ import annotations

import numpy as np

from src.data.section_detector import _load_model, _posfeat

# Names for the four appended position features (see section_detector._posfeat
# and scripts/train_section_classifier.posfeat - they must stay in this order).
_POSITION_FEATURE_NAMES = ["position", "position^2", "position>0.85", "position<0.2"]


class ShapUnavailable(RuntimeError):
    """Raised when SHAP or the classifier file is not available."""


def _feature_names(vec) -> list[str]:
    return list(vec.get_feature_names_out()) + _POSITION_FEATURE_NAMES


def _build_matrix(vec, texts: list[str], positions: list[float]):
    import scipy.sparse as sp

    tfidf = vec.transform(texts)
    pos = sp.csr_matrix(_posfeat(np.array(positions, dtype=float)))
    return sp.hstack([tfidf, pos]).tocsr()


def explain_sections(
    paragraphs: list[str],
    positions: list[float] | None = None,
    top_k: int = 12,
) -> dict:
    """Explain the classifier's label for each paragraph with SHAP.

    Args:
        paragraphs: the paragraph texts to explain.
        positions: relative position (0..1) of each paragraph; derived evenly
            from the count when omitted, matching ``detect_sections``.
        top_k: how many contributing features to report per paragraph.

    Returns:
        A JSON-serialisable dict: metadata, the class list, and one entry per
        paragraph with its predicted label, the class probabilities, and the
        top features pushing toward the predicted label (signed SHAP values -
        positive pushes toward the label, negative away from it).

    Raises:
        ShapUnavailable: if the ``shap`` package or the classifier file is
            missing. The caller surfaces this to the user.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised via the UI path
        raise ShapUnavailable(
            "SHAP explanations need the `shap` package, which is not installed. "
            "Run `pip install shap` and restart."
        ) from exc

    model = _load_model()
    if model is None:
        raise ShapUnavailable(
            "The section classifier (models/section_classifier.joblib) is not "
            "present, so there is nothing to explain."
        )

    vec, clf = model
    # The classifier was pickled under an older scikit-learn that stored a
    # ``multi_class`` attribute; newer versions dropped it but still read it in
    # predict_proba. Restore the original lbfgs/multinomial behaviour so the
    # probabilities match what the model was trained to produce.
    if not hasattr(clf, "multi_class"):
        clf.multi_class = "multinomial"
    total = len(paragraphs)
    if total == 0:
        return {
            "model": "section_classifier",
            "method": "shap.LinearExplainer",
            "classes": [str(c) for c in clf.classes_],
            "n_paragraphs": 0,
            "paragraphs": [],
        }

    if positions is None:
        positions = [i / max(total - 1, 1) for i in range(total)]

    X = _build_matrix(vec, paragraphs, positions)
    names = _feature_names(vec)
    classes = [str(c) for c in clf.classes_]

    # Linear model -> exact SHAP relative to the mean of the paragraphs being
    # explained. A single-paragraph document falls back to a zero reference
    # (an empty document), which still yields well-defined contributions.
    background = X if total > 1 else X.multiply(0)
    explainer = shap.LinearExplainer(
        clf, background, feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(X)

    # shap returns a list (one array per class) for multiclass models on older
    # versions, or a single (n_samples, n_features, n_classes) array on newer
    # ones. Normalise to shap_by_class[class_index] -> (n_samples, n_features).
    if isinstance(shap_values, list):
        shap_by_class = shap_values
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            shap_by_class = [arr[:, :, k] for k in range(arr.shape[2])]
        else:  # binary case: one array, class 1
            shap_by_class = [-arr, arr]

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    entries = []
    for i in range(total):
        pred_label = str(preds[i])
        class_idx = classes.index(pred_label)
        row = np.asarray(shap_by_class[class_idx][i]).ravel()

        # Rank features by absolute contribution, but only report features that
        # are actually present in this paragraph (non-zero), so the list reads
        # as real words and position cues rather than absent vocabulary.
        present = X[i].toarray().ravel() != 0
        candidate = np.where(present)[0]
        if candidate.size == 0:
            candidate = np.arange(row.size)
        order = candidate[np.argsort(-np.abs(row[candidate]))][:top_k]

        features = [
            {
                "feature": names[j],
                "shap_value": round(float(row[j]), 5),
                "direction": "supports" if row[j] >= 0 else "opposes",
            }
            for j in order
        ]

        entries.append({
            "index": i,
            "position": round(float(positions[i]), 4),
            "predicted_section": pred_label,
            "probabilities": {
                classes[k]: round(float(probs[i][k]), 4) for k in range(len(classes))
            },
            "base_value": round(float(np.ravel(explainer.expected_value)[class_idx]), 5),
            "top_features": features,
            "text_preview": paragraphs[i][:240],
        })

    return {
        "model": "section_classifier",
        "method": "shap.LinearExplainer",
        "feature_perturbation": "interventional",
        "classes": classes,
        "n_paragraphs": total,
        "note": (
            "SHAP values explain the linear section classifier's own weights, "
            "relative to the average paragraph in this document. Positive values "
            "push the paragraph toward its predicted section; negative values push "
            "away. This is model interpretation, not legal ground truth."
        ),
        "paragraphs": entries,
    }


def explain_text(
    text: str,
    redact_names: bool = False,
    strip_preamble: bool = True,
    top_k: int = 12,
) -> dict:
    """Explain a raw document's section labels end to end.

    Runs the same preprocessing the inference pipeline uses (clean -> optional
    PII redaction -> paragraph split) so the explained paragraphs match what
    ``detect_sections`` actually labels, then delegates to ``explain_sections``.
    """
    from src.data.cleaning import preprocess, split_paragraphs
    from src.data.pii_redaction import redact

    clean = preprocess(text, strip_preamble_flag=strip_preamble)
    redacted, _ = redact(clean, redact_names_flag=redact_names)
    paragraphs = split_paragraphs(redacted)
    return explain_sections(paragraphs, top_k=top_k)

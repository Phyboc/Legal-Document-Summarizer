"""Rule-based section detector: identifies Facts, Arguments, Reasoning, Judgement."""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "section_classifier.joblib"
_model = None
_model_missing = False


SECTION_PATTERNS = {
    "facts": [
        r"\b(?:appellant|petitioner|respondent)\s+was\b",
        r"\b(?:on|dated?)\s+\d",
        r"\b(?:high court|trial court|lower court|tribunal)\b",
        r"\bfacts?\s+of\s+the\s+case\b",
        r"\b(?:filed|instituted|preferred)\b",
    ],
    "arguments_appellant": [
        r"\b(?:appellant|petitioner)(?:['']s)?\s+(?:counsel|submitt|contend|argue|urge)",
        r"\blearned\s+counsel\s+for\s+the\s+(?:appellant|petitioner)",
        r"\bon\s+behalf\s+of\s+the\s+(?:appellant|petitioner)",
    ],
    "arguments_respondent": [
        r"\b(?:respondent|state|government)(?:['']s)?\s+(?:counsel|submitt|contend|argue|urge)",
        r"\blearned\s+counsel\s+for\s+the\s+(?:respondent|state)",
        r"\bon\s+behalf\s+of\s+the\s+(?:respondent|state)",
    ],
    "reasoning": [
        r"\b(?:we|court|it)\s+(?:hold|observe|consider|note|find)",
        r"\bin\s+our\s+(?:opinion|view|judgment)",
        r"\bsection\s+\d+\s+of\s+the\s+\w+\s+act",
        r"\bin\s+the\s+case\s+of\s+[A-Z]",
        r"\breliance\s+is\s+placed",
        r"\barticle\s+\d+\s+of\s+the\s+constitution",
    ],
    "judgement": [
        r"\bappeal\s+(?:is\s+)?(?:allowed|dismissed|set aside)",
        r"\b(?:we|court)\s+(?:allow|dismiss|set aside|quash|remit|remand)",
        r"\bin\s+the\s+result\b",
        r"\border(?:s)?\s+accordingly\b",
        r"\bconviction\s+is\s+(?:set aside|upheld|maintained)",
        r"\bthe\s+petition\s+(?:is\s+)?(?:allowed|dismissed)",
    ],
}


@dataclass
class Paragraph:
    index: int
    text: str
    section: str
    position: float  # relative position 0.0-1.0
    scores: dict


def score_paragraph(text: str) -> dict:
    """Score a paragraph against each section's patterns."""
    text_lower = text.lower()
    scores = {}
    for section, patterns in SECTION_PATTERNS.items():
        score = 0
        for pat in patterns:
            score += len(re.findall(pat, text_lower))
        scores[section] = score
    return scores


def classify_paragraph(text: str, position: float) -> tuple[str, dict]:
    """Classify a paragraph into one of the sections."""
    scores = score_paragraph(text)

    # Position-based hints
    if position < 0.2:
        scores["facts"] += 1
    if position > 0.85:
        scores["judgement"] += 1

    best = max(scores.items(), key=lambda x: x[1])
    if best[1] < 2:
        # Weak evidence: most sentences in a judgment are the court's own reasoning.
        if position < 0.2:
            return "facts", scores
        return ("judgement" if position > 0.9 else "reasoning"), scores
    return best[0], scores


def _load_model():
    global _model, _model_missing
    if _model is None and not _model_missing:
        if MODEL_PATH.exists():
            import joblib
            _model = joblib.load(MODEL_PATH)
        else:
            _model_missing = True
    return _model


def _posfeat(pos):
    return np.c_[pos, pos ** 2, pos > 0.85, pos < 0.2]


def _predict_learned(texts: list[str], positions: list[float]):
    """Learned labels for each text, or None if the model file is unavailable."""
    model = _load_model()
    if model is None:
        return None
    import scipy.sparse as sp
    vec, clf = model
    X = sp.hstack([vec.transform(texts), sp.csr_matrix(_posfeat(np.array(positions, float)))]).tocsr()
    labels = clf.predict(X)
    out = []
    for text, label in zip(texts, labels):
        if label == "arguments":
            # The training data does not separate sides; use the cue patterns.
            s = score_paragraph(text)
            label = "arguments_respondent" if s["arguments_respondent"] > s["arguments_appellant"] else "arguments_appellant"
        out.append(str(label))
    return out


def detect_sections(paragraphs: list[str]) -> list[Paragraph]:
    """Assign each paragraph a section label."""
    total = len(paragraphs)
    if total == 0:
        return []

    positions = [i / max(total - 1, 1) for i in range(total)]
    learned = _predict_learned(paragraphs, positions)

    results = []
    for i, para in enumerate(paragraphs):
        rule_section, scores = classify_paragraph(para, positions[i])
        section = learned[i] if learned else rule_section
        results.append(Paragraph(
            index=i, text=para, section=section, position=positions[i], scores=scores
        ))
    return results


def group_by_section(paragraphs: list[Paragraph]) -> dict[str, list[Paragraph]]:
    """Group paragraphs by their section label."""
    groups = {}
    for para in paragraphs:
        groups.setdefault(para.section, []).append(para)
    return groups


def fallback_to_single_section(paragraphs: list[Paragraph]) -> bool:
    """Return True if section detection is too weak; treat as one section."""
    if not paragraphs:
        return True
    groups = group_by_section(paragraphs)
    dominant_ratio = max(len(v) for v in groups.values()) / len(paragraphs)
    return dominant_ratio > 0.85

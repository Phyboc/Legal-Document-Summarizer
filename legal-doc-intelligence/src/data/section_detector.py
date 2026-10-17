"""Rule-based section detector: identifies Facts, Arguments, Reasoning, Judgement."""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "section_classifier.joblib"
_model = None
_model_missing = False


# Disposition / operative-order cues. The original judgement set recognised only
# appeal and criminal dispositions ("appeal is allowed", "conviction is set
# aside"). Civil money decrees - "held the defendant liable", "ordered to refund
# and pay compensation", "the suit is decreed" - matched nothing, so the final
# order was filed under reasoning and then trimmed out of the summary. These
# cover civil decrees and generic operative orders as well, and are shared by the
# rule classifier, the trailing-judgement promotion, and the eval outcome metric.
_DISPOSITION_PATTERNS = [
    # appeals / petitions / revisions / writs / suits being disposed of
    r"\b(?:appeal|petition|revision|application|suit)s?\s+(?:is|are|stands?|be|hereby)?\s*(?:partly\s+)?(?:allowed|dismissed|decreed|set aside|disposed)\b",
    r"\bwrit\s+petitions?\s+(?:is|are)\s+(?:allowed|dismissed)\b",
    r"\b(?:we|court|i)\s+(?:allow|dismiss|decree|set aside|quash|remit|remand|acquit|convict)\b",
    # criminal dispositions
    r"\bconviction\s+(?:and\s+sentence\s+)?(?:is|are)\s+(?:set aside|upheld|maintained|confirmed)\b",
    r"\b(?:is|are|stands?|be)\s+acquitted\b",
    # civil money decrees / liability / refund / compensation
    r"\bheld\s+(?:the\s+\w+\s+)?liable\b",
    r"\bliable\s+(?:to\s+(?:pay|refund|compensate)|for\s+(?:breach|damages))\b",
    r"\b(?:ordered|directed|is\s+directed|are\s+directed)\s+to\s+(?:pay|refund|return|deposit|compensate)\b",
    r"\bthe\s+(?:suit|claim)\s+is\s+(?:decreed|dismissed)\b",
    r"\bdecree(?:d)?\s+(?:for|in\s+favour|with\s+costs|against)\b",
    r"\bentitled\s+to\s+(?:a\s+)?(?:refund|compensation|damages|the\s+amount|interest)\b",
    r"\baward(?:ed|s)?\s+(?:of\s+)?(?:compensation|damages|costs|interest)\b",
    # generic operative markers
    r"\bin\s+the\s+result\b",
    r"\border(?:s|ed)?\s+accordingly\b",
    r"\bit\s+is\s+(?:hereby\s+)?(?:ordered|directed|declared)\b",
]
_DISPOSITION_RE = [re.compile(p) for p in _DISPOSITION_PATTERNS]


def is_disposition(text: str) -> bool:
    """True when the text states the court's operative order (the disposition).

    Covers appeal, criminal and civil-decree forms, so a money decree such as
    "held the defendant liable ... ordered to refund and pay compensation" is
    recognised, not only "the appeal is allowed".
    """
    low = text.lower()
    return any(r.search(low) for r in _DISPOSITION_RE)


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
        # Rebuttal/transition cues: respondent-side paragraphs often don't
        # repeat "respondent" and instead open by countering the appellant.
        r"\bper\s+contra\b",
        r"\bon\s+the\s+other\s+hand\b",
        r"\bin\s+(?:reply|rebuttal|refutation)\b",
        r"\b(?:refut|rebutt|oppos|resist)(?:ed|ing|es)?\s+the\s+(?:submissions?|contentions?|arguments?)",
        r"\b(?:additional\s+)?solicitor\s+general\b",
        r"\bstanding\s+counsel\b",
        r"\bcounsel\s+for\s+the\s+union\b",
    ],
    "reasoning": [
        r"\b(?:we|court|it)\s+(?:hold|observe|consider|note|find)",
        r"\bin\s+our\s+(?:opinion|view|judgment)",
        r"\bsection\s+\d+\s+of\s+the\s+\w+\s+act",
        r"\bin\s+the\s+case\s+of\s+[A-Z]",
        r"\breliance\s+is\s+placed",
        r"\barticle\s+\d+\s+of\s+the\s+constitution",
    ],
    "judgement": _DISPOSITION_PATTERNS,
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
    return [str(label) for label in labels]


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


def promote_trailing_judgement(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """Relabel operative-order paragraphs in the document's tail as ``judgement``.

    Both the learned classifier and the rule patterns tend to file a civil money
    decree under ``reasoning`` because its vocabulary differs from the
    appeal/criminal disposition it was tuned on. Any paragraph in the final
    stretch of the document (position >= 0.6) that states a disposition is moved
    to ``judgement`` so the formatter keeps the final order instead of trimming
    it off the end of the reasoning block. A disposition cue earlier than that -
    e.g. the history of a lower court's decree recited in the facts - is left
    alone. Paragraphs are mutated in place and the list is returned for chaining.
    """
    for para in paragraphs:
        if para.position >= 0.6 and para.section != "judgement" and is_disposition(para.text):
            para.section = "judgement"
    return paragraphs


def fallback_to_single_section(paragraphs: list[Paragraph]) -> bool:
    """Return True if section detection is too weak; treat as one section."""
    if not paragraphs:
        return True
    groups = group_by_section(paragraphs)
    dominant_ratio = max(len(v) for v in groups.values()) / len(paragraphs)
    return dominant_ratio > 0.85

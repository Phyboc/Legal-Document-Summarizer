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
    # appeals / petitions / revisions / writs / suits / decrees / orders being disposed of.
    # The subject may be followed by a modal or an adverb ("must accordingly be
    # allowed", "is accordingly set aside") - the forms the earlier pattern missed.
    r"\b(?:appeal|petition|revision|application|suit|writ|matter|decree|order|judgment|judgement|conviction|sentence|impugned\s+\w+)s?(?:\s+(?:is|are|was|were|must|shall|may|not|stands?|stand|be|been|has|have|hereby|accordingly|therefore|also|duly|partly|partially))*\s+(?:allowed(?!\s+to\b)|dismissed|decreed|set\s+aside|quashed|remanded|remitted|modified|upheld|affirmed|reversed|restored|disposed\s+of)\b",
    # The verb not adjacent to its subject: "is accordingly set aside", "is hereby
    # quashed". "allowed to" is a permission, not an order.
    r"\b(?:is|are|stands?)\s+(?:hereby\s+|accordingly\s+|therefore\s+|also\s+)*(?:set\s+aside|quashed|dismissed|allowed(?!\s+to\b)|modified|remanded|remitted|affirmed|upheld|restored)\b",
    # Past-tense court action in the first person: "We dismissed the appeal".
    r"\bwe\s+(?:hereby\s+|accordingly\s+|therefore\s+)*(?:dismissed|allowed(?!\s+to\b)|set\s+aside|quashed|remanded|modified)\b",
    r"\bwrit\s+petitions?\s+(?:is|are)\s+(?:allowed|dismissed)\b",
    r"\b(?:we|court|i)\s+(?:hereby\s+|accordingly\s+|therefore\s+)*(?:allow|dismiss|decree|set aside|quash|remit|remand|acquit|convict|modify|affirm|uphold)\b",
    r"\bdeclared\s+(?:invalid|void|illegal|unconstitutional|ultra\s+vires)\b",
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


# A lower court's (or the trial court's) decision recited as history: "The High
# Court dismissed the petition", "the suit was decreed by the trial court". Such a
# sentence matches ``is_disposition`` but is not the final order of this judgment.
_LOWER_COURT_RE = re.compile(
    r"\b(?:high\s+court|trial\s+court|lower\s+court|sessions\s+(?:court|judge)|magistrate|"
    r"tribunal|single\s+judge|division\s+bench|first\s+appellate|appellate\s+court|"
    r"district\s+(?:court|judge)|learned\s+(?:judge|magistrate)|commissioner|by\s+this\s+court)\b",
    re.IGNORECASE,
)
# Wording that marks the court's own final order even when a lower court is named:
# "the judgment of the High Court is set aside" is final, "the High Court was
# dismissed" is not.
_FINAL_ORDER_FORM_RE = re.compile(
    r"\b(?:is|are|must|shall|stands?)\s+(?:hereby\s+|accordingly\s+)*(?:set\s+aside|allowed|dismissed|quashed|modified|remanded|affirmed|upheld)\b"
    r"|\b(?:accordingly|hereby)\b",
    re.IGNORECASE,
)
_PAST_MARKER_RE = re.compile(
    r"\b(?:was|were|had|has\s+been|have\s+been|dismissed|allowed|decreed|rejected|upheld|"
    r"confirmed|affirmed|quashed|passed|rendered|delivered)\b",
    re.IGNORECASE,
)


def is_lower_court_history(text: str) -> bool:
    """True when a disposition sentence recites a lower court's order as history.

    A sentence is history when it names a lower court (or "this Court" for an
    earlier special-leave order) in the past tense and does not carry final-order
    wording ("accordingly", "hereby", "is set aside"). Heuristic: it is used to
    keep history out of the final-order routing, and is reported as ambiguous.
    """
    if _FINAL_ORDER_FORM_RE.search(text):
        return False
    return bool(_LOWER_COURT_RE.search(text) and _PAST_MARKER_RE.search(text))


def is_operative_order(text: str) -> bool:
    """True when a sentence states this court's operative order (final disposition).

    ``is_disposition`` minus lower-court history. Used to route operative-order
    sentences in the training data; the inference-side promotion keeps the looser
    ``is_disposition`` because it only moves paragraphs in the document's tail.
    """
    return is_disposition(text) and not is_lower_court_history(text)


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


def build_section_groups(paragraphs: list[Paragraph]) -> tuple[dict[str, list[Paragraph]], bool]:
    """Group detected paragraphs into the sections that chunking consumes.

    This is the one grouping rule shared by inference (``src.pipeline``) and the
    training-data build (``scripts/build_dataset.py``), so both see the same
    section labels. Returns ``(section_groups, structure_fallback)``.

    Operative-order paragraphs in the tail are promoted to ``judgement`` first. When
    detection is too weak to trust (``fallback_to_single_section``), the document is
    kept in two groups - the remainder as ``reasoning`` and the operative order as
    ``judgement`` - rather than one block, so the final order cannot be folded into
    and trimmed out of the reasoning.
    """
    paragraphs = promote_trailing_judgement(paragraphs)
    if not fallback_to_single_section(paragraphs):
        return group_by_section(paragraphs), False
    judgement = [p for p in paragraphs if p.section == "judgement"]
    rest = [p for p in paragraphs if p.section != "judgement"]
    groups: dict[str, list[Paragraph]] = {}
    if rest:
        groups["reasoning"] = rest
    if judgement:
        groups["judgement"] = judgement
    return groups, True

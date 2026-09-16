"""PII redaction: Aadhaar, PAN, phone, email; optional name redaction via NER."""
import re
import sys
sys.path.insert(0, ".")
from config import PII_PATTERNS


def redact_patterns(text: str) -> tuple[str, dict]:
    """Redact Aadhaar/PAN/phone/email patterns. Returns (redacted_text, counts)."""
    counts = {}
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        counts[label] = len(matches)
        text = re.sub(pattern, f"[REDACTED_{label.upper()}]", text)
    return text, counts


def redact_names(text: str) -> tuple[str, int]:
    """Optional: redact PERSON entities via spaCy NER. Off by default."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return text, 0

    doc = nlp(text)
    person_spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PERSON"]
    if not person_spans:
        return text, 0

    person_spans.sort(reverse=True)
    for start, end in person_spans:
        text = text[:start] + "[REDACTED_PERSON]" + text[end:]
    return text, len(person_spans)


def redact(text: str, redact_names_flag: bool = False) -> tuple[str, dict]:
    """Full PII redaction. Returns (redacted_text, stats)."""
    text, pattern_counts = redact_patterns(text)
    stats = dict(pattern_counts)
    if redact_names_flag:
        text, name_count = redact_names(text)
        stats["person"] = name_count
    return text, stats

"""Text cleaning and preamble stripping."""
import re

PREAMBLE_MARKERS = [
    r"The Judgment of the Court was delivered by",
    r"The following Judgment of the Court was delivered",
    r"JUDGMENT\s*:",
    r"ORDER\s*:",
]


def clean_text(text: str) -> str:
    """Normalize whitespace and clean up encoding artifacts."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text.strip()


def strip_preamble(text: str) -> str:
    """Strip case-caption preamble (appeal numbers, counsel lists) before the judgment starts."""
    for marker in PREAMBLE_MARKERS:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            return text[match.end():].strip()
    return text


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def preprocess(text: str, strip_preamble_flag: bool = True) -> str:
    """Full preprocessing: clean then optionally strip preamble."""
    text = clean_text(text)
    if strip_preamble_flag:
        text = strip_preamble(text)
    return text

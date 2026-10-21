"""Deterministic, regex-based extraction for the LexNarrate UI.

Everything here is grounded in the raw document text - no model invention. When
a pattern finds nothing we return an empty list so the UI can show
"Not identified" rather than a hallucinated value. spaCy is intentionally not a
dependency: all extraction is rule-based and reproducible.
"""
import re

# ---- Dates --------------------------------------------------------------
# Numeric dd.mm.yyyy / dd-mm-yyyy / dd/mm/yyyy, and "25th April, 1972" style.
_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})[,.]?\s+\d{{4}}\b", re.I),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?[,.]?\s+\d{{4}}\b", re.I),
]
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

# ---- Money / damages ----------------------------------------------------
_MONEY = re.compile(
    r"(?:Rs\.?|₹|INR|USD|\$|£|EUR)\s?[\d,]+(?:\.\d+)?(?:\s?(?:crore|crores|lakh|lakhs|million|billion))?",
    re.I,
)

# ---- Legal provisions: Act + optional Section ---------------------------
_ACT = re.compile(
    r"\b(?:The\s+)?[A-Z][A-Za-z&'()\-\s]{2,60}?Act,?\s+\d{4}\b"
)
_SECTION = re.compile(
    r"\b(?:Section|Sec\.?|S\.|Article|Art\.?|Order|Rule)\s+\d+[A-Za-z]?(?:\(\w+\))?",
    re.I,
)

# ---- Court --------------------------------------------------------------
_COURT = re.compile(
    r"\b(?:Supreme Court(?: of [A-Z][a-z]+)?|High Court(?: of [A-Z][A-Za-z ]+)?|"
    r"District Court(?: of [A-Z][A-Za-z ]+)?|Tribunal|[A-Z][a-z]+ High Court|"
    r"Court of Appeal|Family Court|Sessions Court)\b"
)

# ---- Case number --------------------------------------------------------
_CASE_NO = re.compile(
    r"\b(?:Civil Appeal|Criminal Appeal|Writ Petition|W\.?P\.?|C\.?A\.?|"
    r"Civil Suit|Suit|Appeal|Petition|SLP|O\.?S\.?)\s*(?:\(C\)|\(Crl\))?\s*"
    r"(?:No\.?s?\.?)?\s*\d+[\s/of\d-]*",
    re.I,
)

# ---- Outcome ------------------------------------------------------------
_OUTCOME_RULES = [
    ("PARTIALLY ALLOWED", re.compile(r"\bpartly allowed|partially allowed\b", re.I)),
    ("ALLOWED", re.compile(r"\bappeal is allowed|petition is allowed|allowed with costs|is hereby allowed\b", re.I)),
    ("DISMISSED", re.compile(r"\bappeal is dismissed|petition is dismissed|is hereby dismissed|dismissed with costs|stands dismissed\b", re.I)),
    ("ALLOWED", re.compile(r"\ballowed\b", re.I)),
    ("DISMISSED", re.compile(r"\bdismissed\b", re.I)),
]

# ---- Parties ------------------------------------------------------------
_VS = re.compile(r"([A-Z][A-Za-z.&'\- ]{2,60}?)\s+(?:v\.?|vs\.?|versus)\s+([A-Z][A-Za-z.&'\- ]{2,60})")
# Role-labelled: "The appellant, M/s Delta Constructions, was..." / "respondent, X,"
_ROLE_PARTY = {
    "appellant": re.compile(r"\b(?:appellant|petitioner|plaintiff)s?,\s+([A-Z/][A-Za-z0-9/.&'\-\s]{2,60}?),", re.I),
    "respondent": re.compile(r"\b(?:respondent|defendant)s?,\s+([A-Z/][A-Za-z0-9/.&'\-\s]{2,60}?),", re.I),
}


def _dedupe_keep_order(items):
    seen, out = set(), []
    for it in items:
        k = it.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out


def find_dates(text: str) -> list[str]:
    out = []
    for pat in _DATE_PATTERNS:
        out += pat.findall(text)
    return _dedupe_keep_order(out)


def find_money(text: str) -> list[str]:
    return _dedupe_keep_order(m.group(0) for m in _MONEY.finditer(text))


def find_provisions(text: str) -> list[dict]:
    """Return [{'law':..., 'section':..., 'context':...}] grounded in the text."""
    provisions = []
    norm = re.sub(r"\s+", " ", text)
    acts = _dedupe_keep_order(re.sub(r"\s+", " ", a).strip() for a in _ACT.findall(text))
    for act in acts:
        # Find a nearby section reference in the same sentence as the act.
        idx = norm.find(act)
        window = norm[max(0, idx - 120): idx + len(act) + 120] if idx >= 0 else ""
        sec = _SECTION.search(window)
        provisions.append({
            "law": act,
            "section": sec.group(0) if sec else "",
            "context": window.strip()[:220],
        })
    # Standalone sections not tied to a named act
    if not provisions:
        for s in _dedupe_keep_order(m.group(0) for m in _SECTION.finditer(text))[:6]:
            provisions.append({"law": "", "section": s, "context": ""})
    return provisions


def find_court(text: str) -> str:
    m = _COURT.search(text)
    return m.group(0).strip() if m else ""


def find_case_number(text: str) -> str:
    m = _CASE_NO.search(text)
    return m.group(0).strip().rstrip(".") if m else ""


def find_parties(text: str) -> dict:
    """Best-effort: the 'X v. Y' caption, else role-labelled mentions."""
    m = _VS.search(text)
    if m:
        return {"appellant": m.group(1).strip(), "respondent": m.group(2).strip()}
    out = {"appellant": "", "respondent": ""}
    for role, pat in _ROLE_PARTY.items():
        rm = pat.search(text)
        if rm:
            out[role] = re.sub(r"\s+", " ", rm.group(1)).strip()
    return out


def find_outcome(text: str) -> str:
    """Scan the tail of the document first (operative order lives at the end)."""
    tail = text[-1200:]
    for label, pat in _OUTCOME_RULES:
        if pat.search(tail):
            return label
    for label, pat in _OUTCOME_RULES:
        if pat.search(text):
            return label
    return "NOT APPLICABLE"


def build_timeline(text: str) -> list[dict]:
    """Associate each sentence that carries a year with that year, sorted."""
    from src.data.section_summary_alignment import summary_sentences
    events = {}
    for sent in summary_sentences(text):
        ym = _YEAR.search(sent)
        if not ym:
            continue
        year = ym.group(0)
        clean = re.sub(r"\s+", " ", sent).strip()
        if len(clean) > 160:
            clean = clean[:157] + "..."
        events.setdefault(year, clean)
    return [{"year": y, "event": events[y]} for y in sorted(events)]


def extract_all(text: str) -> dict:
    """One call returning every grounded field for the UI."""
    return {
        "dates": find_dates(text),
        "money": find_money(text),
        "provisions": find_provisions(text),
        "court": find_court(text),
        "case_number": find_case_number(text),
        "parties": find_parties(text),
        "outcome": find_outcome(text),
        "timeline": build_timeline(text),
    }

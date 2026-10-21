"""Evidence-aware alignment: does THIS summary sentence have evidence in THIS chunk?

Why this module exists
----------------------
The Cross-Encoder used for alignment (``cross-encoder/ms-marco-MiniLM-L-6-v2``)
is a general-domain reranker. On Indian Supreme Court judgments it saturates:
measured on an existing 100-document build, every selected (sentence, chunk)
pair scored >= 0.40 (the configured threshold), 83.6% scored >= 0.90 and the
median was 0.984. It therefore answers "is this about the same case / same legal
register?", which is *not* the question alignment needs to answer.

The question alignment needs is factual: **does this sentence have evidence in
THIS specific chunk?** Two chunks of one judgment both mention the appellant, the
government, the High Court and an order, so semantic similarity cannot separate a
chunk describing the Single Judge's 5 January 1981 decision from a chunk
describing a later 17 September 1982 efficiency-bar order.

This module adds that missing check as a *pure-regex / set-based* layer with no
new runtime dependency (no spaCy, no model, no download). It is deliberately
deterministic: the same text always produces the same evidence.

    summary sentence
        |
    existing plausible candidate chunks   (still chosen by section + CE margin)
        |
    Stage A: evidence / admissibility      (rules, no weights)
        |
    Stage B: lexicographic ranking         (no weights, no weighted sum)
        |
    SUPPORTED / AMBIGUOUS / UNSUPPORTED / NOT_SUMMARY_CONTENT
        |
    primary chunk only for SUPPORTED

Design decisions, stated explicitly
-----------------------------------
* **No weighted sum.** The ranking is lexicographic (tuple comparison), so no
  numeric weight is invented for any feature. See ``rank_key``.
* **Missing is not contradiction.** A Tier-1 token that is absent from a chunk is
  *not* automatically treated as proof that the chunk is the wrong source - see
  the long comment on ``_classify_class`` for the three distinct states
  (SATISFIED / MISSING / CONTRADICTED) and why the distinction matters.
* **Locality is a soft, final tie-break only.** It is the last element of the
  rank key, so a nearby chunk with no evidence can never beat a distant chunk
  with strong evidence. Locality is derived from ``Chunk.para_range`` midpoints
  supplied by the caller - never from ``chunk_index``, because ``chunk_document``
  emits chunks in section-block order, not document order.
* **Ambiguity is a real outcome.** A sentence whose evidence is split across
  non-adjacent chunks, or that ties with another candidate, gets no primary chunk
  and is never duplicated across training records.

Nothing here mutates the existing alignment behaviour: ``evidence_mode`` defaults
to OFF in ``section_summary_alignment`` and ``config``, so the pure Cross-Encoder
path remains available and byte-identical for comparison.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, ".")

# --------------------------------------------------------------------------- #
# Decision / status vocabularies
# --------------------------------------------------------------------------- #

DECISION_SUPPORTED = "SUPPORTED"
DECISION_AMBIGUOUS = "AMBIGUOUS"
DECISION_UNSUPPORTED = "UNSUPPORTED"
DECISION_NOT_SUMMARY_CONTENT = "NOT_SUMMARY_CONTENT"
DECISIONS = (
    DECISION_SUPPORTED,
    DECISION_AMBIGUOUS,
    DECISION_UNSUPPORTED,
    DECISION_NOT_SUMMARY_CONTENT,
)

#: The decision attaches the sentence to a chunk only in this state.
DECISIONS_WITH_PRIMARY_CHUNK = (DECISION_SUPPORTED,)

STATUS_SATISFIED = "SATISFIED"
STATUS_PARTIAL = "PARTIAL"
STATUS_MISSING = "MISSING"
STATUS_ABSENT = "ABSENT"
STATUS_CONTRADICTED = "CONTRADICTED"

#: Tier-1 evidence classes.
TIER1_CLASSES = ("years", "dates", "amounts", "provisions", "citations")

#: Only these classes may *contradict* a candidate. Per the audit, the
#: high-confidence mismatch rules are the date / year / amount ones: a chunk that
#: positively talks about a *different* date, year or amount is strong evidence
#: that it is not the source. Provisions and citations are weaker (a judgment may
#: cite a provision or a case it is not describing), so they can support a
#: candidate but can never contradict one.
CONTRADICTION_CLASSES = ("dates", "years", "amounts")

CONTRADICTION_MODE_HIGH_CONFIDENCE = "high_confidence"
CONTRADICTION_MODE_OFF = "off"
CONTRADICTION_MODES = (CONTRADICTION_MODE_HIGH_CONFIDENCE, CONTRADICTION_MODE_OFF)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvidenceSettings:
    """Knobs for the evidence layer.

    ``coverage_bound`` and ``ambiguity_margin`` are the C / M of the audit. Their
    values here are **UNCALIBRATED PLACEHOLDERS** derived from the audit's
    measured distributions (content-word coverage p10 = 0.43, p25 = 0.60), not
    from a sweep. ``scripts/audit_alignment_evidence.py`` is what produces real
    values, and ``build_stats.json`` records whichever values were used so a
    build can never be mistaken for a calibrated one. Because evidence mode
    defaults to OFF, these placeholders change nothing until they are replaced.
    """

    coverage_bound: float = 0.50
    ambiguity_margin: float = 0.02
    min_common_span: int = 4
    max_rare_df: int = 1
    contradiction_mode: str = CONTRADICTION_MODE_HIGH_CONFIDENCE

    def as_dict(self) -> dict:
        return {
            "coverage_bound": self.coverage_bound,
            "ambiguity_margin": self.ambiguity_margin,
            "min_common_span": self.min_common_span,
            "max_rare_df": self.max_rare_df,
            "contradiction_mode": self.contradiction_mode,
        }


# --------------------------------------------------------------------------- #
# Text normalisation (deterministic)
# --------------------------------------------------------------------------- #

# Legal-reporter / citation vocabulary. These words carry no propositional
# content, so they are excluded from content words (otherwise two sentences that
# merely cite something look lexically identical) and from the
# NOT_SUMMARY_CONTENT proper-noun test.
CITATION_VOCAB = frozenset("""
air scr scc scale mlj crilj cri lj lw itr elt supp suppl suppl. vol vols reporter
law report reports ltd co pvt ors ors. ano anr ibid supra vs versus v another
others same supra) per contra cw
""".split())

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can cannot could did do does doing done down
during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other others ought our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which
while who whom why with would you your yours yourself yourselves s t shall will may
must upon said held says observed submitted contended argued urged behalf
""".split())

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_YEAR = r"(?:1[5-9]\d{2}|20\d{2})"

# Footnote markers, e.g. "[489F, H]" or "[12]" or "[(1)]". Deliberately limited to
# 1-3 digits so a real citation bracket such as "[1962] Suppl." is NOT eaten.
_FOOTNOTE_RE = re.compile(r"\[\s*\d{1,3}\s*[A-Za-z]?(?:\s*,\s*[A-Za-z]{1,3})?\s*\]")
_SUPRA_RE = re.compile(r"\(\s*(?:supra|ibid|infra|hereinafter)[^)]*\)|\b(?:supra|ibid|infra)\b")
# Dotted abbreviations: "a.i.r." -> "air", "s.c.r." -> "scr", "m.r." -> "mr".
_DOTTED_ABBREV_RE = re.compile(r"\b(?:[a-z]\.){2,}")
_QUOTES_RE = re.compile(r"[\u2018\u2019\u201c\u201d\"']")

#: Reporters/places used to recognise report-style citations such as
#: "1982 Karnataka 179". Used only to *mask* citation spans and to build the
#: citation evidence class.
_REPORTER_ALT = (
    "karnataka|kerala|madras|bombay|calcutta|punjab|allahabad|delhi|rajasthan|"
    "gujarat|patna|orissa|mysore|hyderabad|nagpur|lahore|sind|air|scr|scc|scale|"
    "mlj|crilj|itr|elt|kb|lr|klj|glr"
)


def normalize_text(text: str) -> str:
    """Deterministic normalisation shared by every extractor.

    Applies, in order: case folding; quote stripping; ``&`` -> ``and``;
    ``v.``/``vs.``/``versus`` -> ``vs``; h/hpossessive removal; dotted-abbreviation
    collapsing; footnote/supra marker removal; whitespace collapsing.
    """
    if not text:
        return ""
    out = text.lower()
    # Curly quotes first, so the possessive rules below see a real apostrophe.
    out = out.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    out = re.sub(r"'s\b", "", out)          # appellant's -> appellant
    out = re.sub(r"\bs'\b", "s", out)       # appellants' -> appellants
    out = _QUOTES_RE.sub("", out)
    out = out.replace("&", " and ")
    out = re.sub(r"\bv\.?\s|\bvs\.?\b|\bversus\b", " vs ", out)
    out = _DOTTED_ABBREV_RE.sub(lambda m: m.group(0).replace(".", ""), out)
    out = _SUPRA_RE.sub(" ", out)
    out = _FOOTNOTE_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def is_ocr_split(word: str, other: str) -> bool:
    """True when two adjacent words concatenate into a longer single word.

    IN-ABS text contains PDF/OCR word splits such as ``Manufac turing``. This is a
    deliberately cheap, deterministic test used only to *expand* word sets for
    matching; it is never applied as a rewrite of the evidence.
    """
    if not word or not other:
        return False
    joined = word + other
    return len(joined) >= 6 and joined.isalpha()


def expanded_word_set(words: list[str] | tuple[str, ...]) -> set[str]:
    """Word set plus OCR-merge candidate joins of adjacent words."""
    out = {w for w in words if w}
    for left, right in zip(words, words[1:]):
        if is_ocr_split(left, right):
            out.add(left + right)
    return out


# --------------------------------------------------------------------------- #
# Evidence extraction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Evidence:
    """Everything the evidence layer can check about one piece of text."""

    raw: str
    normalized: str
    masked: str
    years: frozenset[str] = frozenset()
    dates: frozenset[str] = frozenset()
    amounts: frozenset[str] = frozenset()
    provisions: frozenset[str] = frozenset()
    citations: frozenset[str] = frozenset()
    content_words: tuple[str, ...] = ()
    has_vs: bool = False
    footnote_only: bool = False

    @property
    def word_set(self) -> set[str]:
        """Content words plus OCR-merge joins."""
        return expanded_word_set(self.content_words)

    def tier1(self, class_name: str) -> frozenset[str]:
        return getattr(self, class_name)

    @property
    def tier1_total(self) -> int:
        return sum(len(getattr(self, name)) for name in TIER1_CLASSES)

    def as_dict(self) -> dict:
        """JSON-serialisable view, for record metadata and the ambiguity report."""
        return {
            "years": sorted(self.years),
            "dates": sorted(self.dates),
            "amounts": sorted(self.amounts),
            "provisions": sorted(self.provisions),
            "citations": sorted(self.citations),
            "content_word_count": len(self.content_words),
            "has_vs": self.has_vs,
        }


def _canonical_date(year: int | None, month: int, day: int) -> str:
    """Canonical date key. Unknown year becomes ``----`` so it still matches."""
    y = f"{year:04d}" if year else "----"
    return f"{y}-{month:02d}-{day:02d}"


def _month_number(token: str) -> int | None:
    return _MONTHS.get(token.lower().strip("."))


def extract_dates(normalized: str) -> set[str]:
    """Extract calendar dates in every form the audit identified.

    Handles ``September 3, 1959``; ``3-9-1959`` / ``3.9.1959`` / ``3/9/1959``;
    ``20th August, 1991``; ``the 5th day of January, 1981``; and month/year pairs
    (``January 1981``) as ``----`` day. Returns canonical ``YYYY-MM-DD`` keys.
    """
    found: set[str] = set()

    # "the 5th day of January, 1981" / "5 January 1981" / "January 5, 1981"
    pattern = re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?({_MONTH_ALT})\b\.?,?\s*({_YEAR})?\b"
    )
    for day, month, year in pattern.findall(normalized):
        m = _month_number(month)
        if m and 1 <= int(day) <= 31:
            found.add(_canonical_date(int(year) if year else None, m, int(day)))

    pattern = re.compile(rf"\b({_MONTH_ALT})\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*({_YEAR})?\b")
    for month, day, year in pattern.findall(normalized):
        m = _month_number(month)
        if m and 1 <= int(day) <= 31:
            found.add(_canonical_date(int(year) if year else None, m, int(day)))

    # Numeric forms: 3-9-1959, 3.9.1959, 3/9/1959 (either day-first or 4-digit-year-first).
    for left, middle, right in re.findall(r"\b(\d{1,4})[.\-/](\d{1,2})[.\-/](\d{2,4})\b", normalized):
        month = int(middle)
        if len(left) == 4:                       # 1959-09-03
            year, day = int(left), int(right)
        else:
            day, year = int(left), int(right)
            year = year + 1900 if year < 100 else year
        if 1 <= month <= 12 and 1 <= day <= 31:
            found.add(_canonical_date(year, month, day))

    # Month + year with no day, e.g. "in January 1981".
    for month, year in re.findall(rf"\b({_MONTH_ALT})\b\.?,?\s+({_YEAR})\b", normalized):
        m = _month_number(month)
        if m:
            found.add(f"{int(year):04d}-{m:02d}-00")

    # Drop the day-less form for any year+month that also produced a full date in
    # this same text ("January 5, 1981" yields both "1981-01-05" and, from the
    # month+year sweep, "1981-01-00"). Keeping the redundancy would let one date
    # mention look like two independent facts and could raise a false ambiguity.
    full_year_months = {token[:7] for token in found if not token.endswith("-00")}
    return {token for token in found if not (token.endswith("-00") and token[:7] in full_year_months)}


def extract_years(normalized: str) -> set[str]:
    """Bare years and year ranges, e.g. ``1981`` and ``1981-82`` -> ``1981/1982``."""
    found: set[str] = set()
    for start, end in re.findall(rf"\b({_YEAR})\s*[-\u2013/]\s*(\d{{2}}|{_YEAR})\b", normalized):
        start_i = int(start)
        end_i = int(end)
        if end_i < 100:
            end_i = (start_i // 100) * 100 + end_i
        found.add(f"{start_i}/{end_i}")
    for year in re.findall(rf"\b({_YEAR})\b", normalized):
        found.add(year)
    return found


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_SCALES = {
    "thousand": 10 ** 3, "lakh": 10 ** 5, "lakhs": 10 ** 5, "lac": 10 ** 5,
    "lacs": 10 ** 5, "crore": 10 ** 7, "crores": 10 ** 7,
    "million": 10 ** 6, "millions": 10 ** 6, "billion": 10 ** 9,
}


def extract_amounts(normalized: str) -> set[str]:
    """Monetary amounts in symbol, digit and word form.

    ``Rs. 1,00,000`` -> ``inr:100000``; ``\u20b95,00,000`` -> ``inr:500000``;
    ``50 lakhs`` -> ``inr:5000000``; ``two crores`` -> ``inr:20000000``.
    Returns canonical ``<currency>:<integer>`` keys.
    """
    found: set[str] = set()

    def _int(raw: str) -> int | None:
        cleaned = raw.replace(",", "").strip()
        try:
            return int(float(cleaned))
        except ValueError:
            return None

    for amount in re.findall(r"(?:rs\.?|inr|rupees?|\u20b9)\s*([\d][\d,]*(?:\.\d+)?)", normalized):
        value = _int(amount)
        if value is not None:
            found.add(f"inr:{value}")

    for amount, scale in re.findall(
        r"\b([\d][\d,]*(?:\.\d+)?)\s*(thousand|lakhs?|lacs?|crores?|millions?|billions?)\b", normalized
    ):
        value = _int(amount)
        if value is not None:
            found.add(f"inr:{value * _SCALES[scale]}")

    for word, scale in re.findall(
        r"\b(" + "|".join(_WORD_NUMBERS) + r")\s+(thousand|lakhs?|lacs?|crores?|millions?|billions?)\b",
        normalized,
    ):
        found.add(f"inr:{_WORD_NUMBERS[word] * _SCALES[scale]}")

    for amount in re.findall(r"\$\s*([\d][\d,]*(?:\.\d+)?)", normalized):
        value = _int(amount)
        if value is not None:
            found.add(f"usd:{value}")

    return found


def extract_provisions(normalized: str) -> set[str]:
    """Statutory provisions: ``Section 32``, ``Article 14``, ``Order 41 Rule 1``."""
    found: set[str] = set()
    for number in re.findall(r"\bsection\s+(\d+[a-z]?)\b", normalized):
        found.add(f"section {number}")
    for number in re.findall(r"\barticle\s+(\d+[a-z]?)\b", normalized):
        found.add(f"article {number}")
    for order, rule in re.findall(r"\border\s+(\d+[a-z]?)\s+rule\s+(\d+[a-z]?)\b", normalized):
        found.add(f"order {order} rule {rule}")
    for number in re.findall(r"\brule\s+(\d+[a-z]?)\b", normalized):
        found.add(f"rule {number}")
    for kind in ("schedule", "clause", "regulation", "sub-section", "subsection"):
        for number in re.findall(rf"\b{re.escape(kind)}\s+(\d+[a-z]?)\b", normalized):
            found.add(f"{kind} {number}")
    return found


_CITATION_PATTERNS = (
    # AIR 1966 Punjab 509
    re.compile(rf"\bair\s*({_YEAR})\s*([a-z]+)\s*(\d{{1,4}})\b"),
    # [1962] Suppl. / (1962) 2 SCR
    re.compile(rf"[\[(]\s*({_YEAR})\s*[\])]\s*([a-z]{{2,10}})"),
    # 1982 Karnataka 179
    re.compile(rf"\b({_YEAR})\s+({_REPORTER_ALT})\s+(\d{{1,4}})\b"),
    # 179 Karnataka 1982
    re.compile(rf"\b(\d{{1,4}})\s+({_REPORTER_ALT})\s+({_YEAR})\b"),
)


def extract_citations(normalized: str) -> set[str]:
    """Legal/report citations, canonicalised to ``<year>-<reporter>-<number>``.

    Covers ``1982 Karnataka 179``, ``[1962] Suppl.``, ``A.I.R. 1966 Punjab 509``
    and the number-first ``179 Karnataka 1982`` form.
    """
    found: set[str] = set()
    for index, pattern in enumerate(_CITATION_PATTERNS):
        for groups in pattern.findall(normalized):
            parts = [str(g) for g in groups]
            if index == 3:                      # number, reporter, year
                number, reporter, year = parts
                found.add(f"{year}-{reporter}-{number}")
            elif index == 1:                    # year bracket, reporter only
                year, reporter = parts
                found.add(f"{year}-{reporter}")
            else:                               # year, reporter, number
                year, reporter, number = parts
                found.add(f"{year}-{reporter}-{number}")
    return found


def mask_citations(normalized: str) -> str:
    """Replace citation spans with a space.

    Used before extracting years/dates/amounts and before computing content
    words, so that a *citation* year (``[1962] Suppl.``) is never mistaken for a
    fact-year and citation tokens never inflate lexical overlap - both were
    confirmed misalignment sources in the audit.
    """
    masked = normalized
    for pattern in _CITATION_PATTERNS:
        masked = pattern.sub(" ", masked)
    masked = re.sub(r"\[\s*\d{4}\s*\]", " ", masked)
    return re.sub(r"\s+", " ", masked).strip()


def content_words(masked: str) -> tuple[str, ...]:
    """Ordered content words: alphabetic, >= 3 chars, not a stopword/citation word."""
    words = []
    for token in re.findall(r"[a-z]{3,}", masked):
        if token in STOPWORDS or token in CITATION_VOCAB:
            continue
        words.append(token)
    return tuple(words)


def _proper_tokens(raw: str) -> set[str]:
    """Normalised forms of tokens that look like proper nouns in the raw text.

    A capitalised token counts as a proper noun unless it is the sentence's first
    token *and* the following token is lower-case (which would just be ordinary
    sentence capitalisation, e.g. ``Time begins``). This distinction is what keeps
    terse operative orders such as ``Appeal dismissed.`` out of the
    NOT_SUMMARY_CONTENT bucket.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z.]*", raw)
    proper: set[str] = set()
    for index, token in enumerate(tokens):
        if not token[:1].isupper():
            continue
        following_capitalised = index + 1 < len(tokens) and tokens[index + 1][:1].isupper()
        if index == 0 and not following_capitalised:
            continue
        cleaned = normalize_text(token)
        cleaned = re.sub(r"[^a-z]", "", cleaned)
        if len(cleaned) >= 2:
            proper.add(cleaned)
    return proper


def extract_evidence(text: str) -> Evidence:
    """Full deterministic extraction for one piece of text."""
    normalized = normalize_text(text)
    if not normalized:
        return Evidence(raw=text or "", normalized="", masked="")

    citations = extract_citations(normalized)
    masked = mask_citations(normalized)

    return Evidence(
        raw=text or "",
        normalized=normalized,
        masked=masked,
        years=frozenset(extract_years(masked)),
        dates=frozenset(extract_dates(masked)),
        amounts=frozenset(extract_amounts(masked)),
        provisions=frozenset(extract_provisions(masked)),
        citations=frozenset(citations),
        content_words=content_words(masked),
        has_vs=bool(re.search(r"\bvs\b", normalized)),
        footnote_only=not content_words(masked) and not re.search(r"[a-z]{2,}", masked),
    )


def document_frequency(texts) -> Counter:
    """Document-local ``df``: in how many *chunks* each content word occurs."""
    df: Counter = Counter()
    for text in texts:
        evidence = text if isinstance(text, Evidence) else extract_evidence(text)
        df.update(set(evidence.content_words))
    return df


# --------------------------------------------------------------------------- #
# Tier 2 / Tier 3 comparisons
# --------------------------------------------------------------------------- #

def longest_common_span(left: list[str] | tuple[str, ...], right: list[str] | tuple[str, ...]) -> int:
    """Length of the longest common *contiguous* content-word run.

    Tolerates OCR word splits on either side by allowing a token to match the
    concatenation of two adjacent tokens.
    """
    if not left or not right:
        return 0
    right_joined = expanded_word_set(list(right))
    previous = [0] * (len(right) + 1)
    best = 0
    for i, lword in enumerate(left, 1):
        current = [0] * (len(right) + 1)
        for j, rword in enumerate(right, 1):
            matched = lword == rword or (lword + rword) in right_joined or lword in right_joined
            if matched:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def coverage_ratio(sentence_words: tuple[str, ...], chunk_word_set: set[str]) -> float:
    """Tier-3 content-word coverage: share of the sentence's content words present."""
    unique = set(sentence_words)
    if not unique:
        return 0.0
    return len(unique & chunk_word_set) / len(unique)


def is_not_summary_content(evidence: Evidence) -> bool:
    """True for citation/footnote fragments rather than actual summary content.

    Requires a citation marker (a recognised citation or an explicit ``vs``) *and*
    a body whose content words are entirely proper nouns or citation vocabulary.
    That is what separates ``A.I.R. 1966 Punjab 509; Siri Chand vs Jot Ram...``
    from a genuine short order such as ``Appeal dismissed.``, which contains
    ordinary content words and therefore stays eligible.
    """
    if evidence.footnote_only:
        return True
    if not (evidence.citations or evidence.has_vs):
        return False
    outside = [word for word in evidence.content_words if word not in CITATION_VOCAB]
    if not outside:
        return True
    if len(outside) > 8:
        return False
    proper = _proper_tokens(evidence.raw)
    return all(word in proper for word in outside)


# --------------------------------------------------------------------------- #
# Stage A: per-class status (SUPPORTED / MISSING / CONTRADICTED)
# --------------------------------------------------------------------------- #

@dataclass
class ClassResult:
    """Outcome of one Tier-1 class for one (sentence, chunk) pair.

    ``SATISFIED``  - the chunk contains every token the sentence carries.
    ``PARTIAL``    - the chunk contains some of them (never a contradiction).
    ``MISSING``    - none are present and absence proves nothing (see below).
    ``ABSENT``     - the sentence carries no token of this class.
    ``CONTRADICTED`` - the chunk positively talks about a *different* value.
    """

    name: str
    status: str
    matched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def supports(self) -> bool:
        return self.status in (STATUS_SATISFIED, STATUS_PARTIAL)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "matched": list(self.matched),
            "missing": list(self.missing),
        }


def _classify_class(
    name: str,
    sentence_tokens: frozenset[str],
    chunk_tokens: frozenset[str],
    contradiction_enabled: bool,
) -> ClassResult:
    """Classify one evidence class for one candidate chunk.

    THE MISSING-vs-CONTRADICTION DISTINCTION (the important part)
    ------------------------------------------------------------
    A Tier-1 token being absent from a chunk is **not** by itself evidence that
    the chunk is the wrong source:

    * The summary is abstractive, so a paraphrased sentence may legitimately
      carry a date in a form the extraction did not recover.
    * IN-ABS judgments are chunked with overlap, so a sentence can straddle two
      adjacent chunks and the token may sit in the neighbour.
    * A sentence often carries several facts (a case and a holding), so no single
      chunk is expected to contain all of them.

    So absence is recorded as ``MISSING`` - "not found, proves nothing" - and the
    candidate stays admissible. ``CONTRADICTED`` is reserved for a much stronger
    situation: the sentence carries tokens of this class, the chunk carries *none*
    of them, **and** the chunk positively contains other values of the same class
    (a different date / year / amount). Then the chunk is talking about a
    different time, or a different sum, and it is genuinely not the source.

    Provisions and citations never contradict: a judgment may cite a provision or
    a case it is not describing, so those classes can only support or be missing.
    """
    if not sentence_tokens:
        return ClassResult(name=name, status=STATUS_ABSENT)
    if not chunk_tokens:
        return ClassResult(name=name, status=STATUS_MISSING, missing=tuple(sorted(sentence_tokens)))

    matched = sentence_tokens & chunk_tokens
    missing = sentence_tokens - chunk_tokens

    if not missing:
        return ClassResult(name=name, status=STATUS_SATISFIED, matched=tuple(sorted(matched)))
    if matched:
        # Some of this class is here, some is not: partial support. This is the
        # multi-event / overlapping-chunk case, and it must never be read as a
        # contradiction - it is exactly what the AMBIGUOUS split-evidence check
        # is for.
        return ClassResult(
            name=name,
            status=STATUS_PARTIAL,
            matched=tuple(sorted(matched)),
            missing=tuple(sorted(missing)),
        )

    if contradiction_enabled and name in CONTRADICTION_CLASSES:
        # Nothing matched, and the chunk holds different values of this class.
        return ClassResult(
            name=name,
            status=STATUS_CONTRADICTED,
            missing=tuple(sorted(sentence_tokens)),
        )
    return ClassResult(name=name, status=STATUS_MISSING, missing=tuple(sorted(sentence_tokens)))


# --------------------------------------------------------------------------- #
# Candidates and ranking
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    """One plausible candidate chunk, with the Cross-Encoder score it received."""

    index: int
    label: str
    text: str
    ce_score: float = 0.0
    position: float | None = None


@dataclass
class CandidateEvidence:
    """Evidence evaluation of one candidate. Also the Stage-B ranking key holder."""

    index: int
    label: str
    ce_score: float
    classes: dict[str, ClassResult] = field(default_factory=dict)
    matched_count: int = 0
    longest_span: int = 0
    rare_shared: int = 0
    coverage: float = 0.0
    locality_score: float = 0.0
    evidence: Evidence | None = None

    # --- Stage A derived ----------------------------------------------------- #
    @property
    def contradicted(self) -> bool:
        return any(result.status == STATUS_CONTRADICTED for result in self.classes.values())

    @property
    def supported_classes(self) -> int:
        return sum(1 for r in self.classes.values() if r.status == STATUS_SATISFIED)

    @property
    def partial_classes(self) -> int:
        return sum(1 for r in self.classes.values() if r.status == STATUS_PARTIAL)

    @property
    def tier1_classes(self) -> int:
        """Rank key 1: how many Tier-1 evidence classes this chunk supports."""
        return self.supported_classes + self.partial_classes

    @property
    def contradiction_classes(self) -> list[str]:
        return sorted(n for n, r in self.classes.items() if r.status == STATUS_CONTRADICTED)

    def as_dict(self) -> dict:
        return {
            "chunk_index": self.index,
            "section": self.label,
            "ce_score": round(float(self.ce_score), 4),
            "tier1_classes": self.tier1_classes,
            "matched_count": self.matched_count,
            "longest_span": self.longest_span,
            "rare_shared": self.rare_shared,
            "coverage": round(float(self.coverage), 4),
            "contradicted": self.contradicted,
            "class_status": {name: r.as_dict() for name, r in self.classes.items()},
        }


#: Stage-B lexicographic keys, in priority order. ``True`` = discrete, so a tie
#: requires exact equality; ``False`` = continuous, so a tie allows the
#: calibrated ambiguity margin. The last key makes locality a soft, final
#: tie-break that can never outrank any evidence.
_RANK_KEYS = (
    ("tier1_classes", True),
    ("matched_count", True),
    ("longest_span", True),
    ("rare_shared", True),
    ("ce_score", False),
    ("coverage", False),
    ("locality_score", False),
)


def rank_key(candidate: CandidateEvidence) -> tuple:
    """Stage-B key. Lexicographic, so no weight is ever invented."""
    return (
        candidate.tier1_classes,
        candidate.matched_count,
        candidate.longest_span,
        candidate.rare_shared,
        candidate.ce_score,
        candidate.coverage,
        candidate.locality_score,
    )


def _effectively_tied(left: CandidateEvidence, right: CandidateEvidence, margin: float) -> bool:
    """True when two candidates tie on every rank key, allowing ``margin`` on the
    continuous keys."""
    for (_, discrete), a, b in zip(_RANK_KEYS, rank_key(left), rank_key(right)):
        if discrete:
            if a != b:
                return False
        elif abs(a - b) > margin:
            return False
    return True


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #

@dataclass
class EvidenceDecision:
    """Final outcome for one summary sentence."""

    decision: str
    primary_index: int | None = None
    reason: str = ""
    candidates: list[CandidateEvidence] = field(default_factory=list)
    ambiguity: dict | None = None
    sentence_evidence: Evidence | None = None

    @property
    def attached(self) -> bool:
        return self.decision in DECISIONS_WITH_PRIMARY_CHUNK and self.primary_index is not None

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "primary_chunk": self.primary_index,
            "reason": self.reason,
            "candidates": [c.as_dict() for c in self.candidates],
            "ambiguity": self.ambiguity,
        }


def _adjacent(indices: list[int]) -> bool:
    """Chunks are adjacent when their global indices differ by at most 1.

    Within a section the chunker emits consecutive indices, and consecutive chunks
    share the ``CHUNK_OVERLAP`` tail, so a sentence straddling indices ``j`` and
    ``j+1`` is a chunking artefact rather than a multi-event sentence. Indices
    further apart - including across section blocks - are genuinely distant.
    """
    return bool(indices) and (max(indices) - min(indices)) <= 1


def _split_evidence_ambiguity(
    sentence_evidence: Evidence, admissible: list[CandidateEvidence]
) -> dict | None:
    """Detect Tier-1 evidence split across non-adjacent chunks.

    Ambiguous when the sentence carries >= 2 tokens of one class, **no single
    candidate contains all of them**, and at least two non-adjacent candidates
    each contain part of it. Such a sentence describes more than one event and
    must not be attached to either chunk (it would otherwise be duplicated across
    training records, or given a target it cannot be produced from).
    """
    for name in TIER1_CLASSES:
        tokens = sentence_evidence.tier1(name)
        if len(tokens) < 2:
            continue
        holders = [
            c for c in admissible
            if c.classes[name].matched
        ]
        if len(holders) < 2:
            continue
        if any(len(c.classes[name].matched) == len(tokens) for c in holders):
            continue
        indices = [c.index for c in holders]
        if _adjacent(indices):
            continue
        return {
            "reason": "tier1_evidence_split_across_distant_chunks",
            "evidence_class": name,
            "tokens": sorted(tokens),
            "chunk_indices": sorted(indices),
            "per_chunk_matched": {c.index: list(c.classes[name].matched) for c in holders},
        }
    return None


def evaluate_candidate(
    sentence_evidence: Evidence,
    candidate: Candidate,
    *,
    settings: EvidenceSettings | None = None,
    document_frequency: Counter | None = None,
    anchor_position: float | None = None,
) -> CandidateEvidence:
    """Stage A + Tier-2/Tier-3 evaluation of one (sentence, candidate) pair.

    Public so the calibration audit can score pairs directly, without going
    through the decision logic. Purely a function of its inputs - no state.
    """
    settings = settings or EvidenceSettings()
    contradiction_enabled = settings.contradiction_mode == CONTRADICTION_MODE_HIGH_CONFIDENCE
    chunk_evidence = extract_evidence(candidate.text)

    classes = {
        name: _classify_class(
            name,
            sentence_evidence.tier1(name),
            chunk_evidence.tier1(name),
            contradiction_enabled,
        )
        for name in TIER1_CLASSES
    }
    word_set = chunk_evidence.word_set
    document_frequency = document_frequency or Counter()
    rare_shared = sum(
        1 for word in set(sentence_evidence.content_words)
        if word in word_set and document_frequency.get(word, 0) <= settings.max_rare_df
    )
    locality = 0.0
    if candidate.position is not None and anchor_position is not None:
        locality = -abs(float(candidate.position) - float(anchor_position))

    return CandidateEvidence(
        index=candidate.index,
        label=candidate.label,
        ce_score=float(candidate.ce_score),
        classes=classes,
        matched_count=sum(len(r.matched) for r in classes.values()),
        longest_span=longest_common_span(
            sentence_evidence.content_words, chunk_evidence.content_words
        ),
        rare_shared=rare_shared,
        coverage=coverage_ratio(sentence_evidence.content_words, word_set),
        locality_score=locality,
        evidence=chunk_evidence,
    )


def decide(
    sentence: str,
    candidates: list[Candidate],
    *,
    settings: EvidenceSettings | None = None,
    document_frequency: Counter | None = None,
    sentence_evidence: Evidence | None = None,
    anchor_position: float | None = None,
) -> EvidenceDecision:
    """Run Stage A (admissibility) then Stage B (lexicographic ranking) on one sentence.

    Returns one of SUPPORTED / AMBIGUOUS / UNSUPPORTED / NOT_SUMMARY_CONTENT. Only
    SUPPORTED carries a ``primary_index``.
    """
    settings = settings or EvidenceSettings()
    sentence_evidence = sentence_evidence or extract_evidence(sentence)

    if is_not_summary_content(sentence_evidence):
        return EvidenceDecision(
            decision=DECISION_NOT_SUMMARY_CONTENT,
            reason="citation_or_footnote_fragment",
            sentence_evidence=sentence_evidence,
        )

    if not candidates:
        return EvidenceDecision(
            decision=DECISION_UNSUPPORTED,
            reason="no_candidate_chunks",
            sentence_evidence=sentence_evidence,
        )

    if document_frequency is None:
        document_frequency = document_frequency_of([c.text for c in candidates])

    if settings.contradiction_mode not in CONTRADICTION_MODES:
        raise ValueError(
            f"unknown contradiction_mode {settings.contradiction_mode!r}; "
            f"expected one of {CONTRADICTION_MODES}"
        )

    # Locality anchor. By default it is the candidate carrying the highest
    # Cross-Encoder score: that is computed WITHOUT any evidence, so locality
    # stays a genuine prior ("stay near where the semantic model pointed") and
    # never becomes circular. A caller may pass an explicit anchor instead.
    if anchor_position is None:
        anchor = None
        for candidate in candidates:
            if candidate.position is None:
                anchor = None
                break
            if anchor is None or candidate.ce_score > anchor[0]:
                anchor = (candidate.ce_score, candidate.position)
        anchor_position = anchor[1] if anchor else None

    evaluated: list[CandidateEvidence] = [
        evaluate_candidate(
            sentence_evidence,
            candidate,
            settings=settings,
            document_frequency=document_frequency,
            anchor_position=anchor_position,
        )
        for candidate in candidates
    ]

    admissible = [c for c in evaluated if not c.contradicted]
    if not admissible:
        return EvidenceDecision(
            decision=DECISION_UNSUPPORTED,
            reason="all_candidates_contradicted",
            candidates=evaluated,
            sentence_evidence=sentence_evidence,
        )

    # Split evidence beats a shallow win: a sentence spanning two distant chunks
    # is ambiguous even though one chunk may score higher on the CE.
    ambiguity = _split_evidence_ambiguity(sentence_evidence, admissible)
    if ambiguity is not None:
        return EvidenceDecision(
            decision=DECISION_AMBIGUOUS,
            reason="tier1_evidence_split_across_distant_chunks",
            candidates=evaluated,
            ambiguity=ambiguity,
            sentence_evidence=sentence_evidence,
        )

    ranked = sorted(admissible, key=rank_key, reverse=True)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    if runner is not None and _effectively_tied(best, runner, settings.ambiguity_margin):
        holder_indices = [best.index] + [c.index for c in ranked if _effectively_tied(best, c, settings.ambiguity_margin)]
        return EvidenceDecision(
            decision=DECISION_AMBIGUOUS,
            reason="candidates_effectively_tied",
            candidates=evaluated,
            ambiguity={
                "reason": "candidates_effectively_tied",
                "chunk_indices": sorted(holder_indices),
                "rank_keys": {str(c.index): [float(v) for v in rank_key(c)] for c in ranked[:3]},
            },
            sentence_evidence=sentence_evidence,
        )

    if best.tier1_classes > 0:
        return EvidenceDecision(
            decision=DECISION_SUPPORTED,
            primary_index=best.index,
            reason="tier1_evidence",
            candidates=evaluated,
            sentence_evidence=sentence_evidence,
        )

    # No Tier-1 evidence anywhere: fall back to Tier-2/Tier-3 and refuse to
    # attach when only soft semantic similarity is on offer.
    soft_enough = (
        best.coverage >= settings.coverage_bound
        or best.longest_span >= settings.min_common_span
        or best.rare_shared > 0
    )
    if not soft_enough:
        return EvidenceDecision(
            decision=DECISION_UNSUPPORTED,
            reason="soft_similarity_only_coverage_below_bound",
            candidates=evaluated,
            sentence_evidence=sentence_evidence,
        )

    return EvidenceDecision(
        decision=DECISION_SUPPORTED,
        primary_index=best.index,
        reason="soft_evidence_only",
        candidates=evaluated,
        sentence_evidence=sentence_evidence,
    )


def document_frequency_of(texts) -> Counter:
    """Alias kept for callers that prefer an explicit name."""
    return document_frequency(texts)

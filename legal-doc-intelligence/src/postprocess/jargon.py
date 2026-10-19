"""Legal jargon simplification dictionary. Inline-safe vs. glossary-only terms."""
import re


# 30 core inline-safe substitutions (in production would be 170+)
INLINE_SUBSTITUTIONS = {
    "appellant": "person appealing",
    "respondent": "other party",
    "petitioner": "person filing the case",
    "plaintiff": "person filing the case",
    "defendant": "accused party",
    "prima facie": "at first glance",
    "ex parte": "from one side only",
    "sub judice": "under judicial consideration",
    "mala fide": "in bad faith",
    "bona fide": "in good faith",
    "quash": "cancel",
    "set aside": "cancel",
    "dismissed": "rejected",
    "allowed": "accepted",
    "remand": "send back",
    "acquit": "declare not guilty",
    "convict": "declare guilty",
    "adjudicate": "decide",
    "affidavit": "sworn statement",
    "counsel": "lawyer",
    "hereinafter": "later in this document",
    "aforesaid": "mentioned earlier",
    "notwithstanding": "despite",
    "whereas": "since",
    "learned counsel": "lawyer",
    "moved this Court": "asked this court",
    "in the result": "in conclusion",
    "impugned order": "order being challenged",
    "operative order": "actual decision",
    "ratio decidendi": "legal reasoning behind the decision",
    # Inflected / plural forms. The base-form \b...\b patterns above miss these
    # ("appellant" does not match "appellants", "convict" not "convicted"), which
    # left Citizen Mode identical to Lawyer Mode on judgements written in the
    # plural. Replacements carry no leading article so "the appellants" reads as
    # "the people appealing", not "the the person appealing".
    "appellants": "people appealing",
    "respondents": "other parties",
    "petitioners": "people filing the case",
    "plaintiffs": "people filing the case",
    "defendants": "accused parties",
    "convicted": "found guilty",
    "acquitted": "found not guilty",
    "quashed": "cancelled",
    "remanded": "sent back",
    "adjudicated": "decided",
    "affidavits": "sworn statements",
}


# Glossary-only terms (not substituted inline — meaning changes too much)
GLOSSARY_ONLY = {
    "consideration": "something of value exchanged in a contract",
    "void": "having no legal effect",
    "voidable": "can be cancelled by one party",
    "estoppel": "prevented from claiming otherwise",
    "res judicata": "already decided by a court",
    "locus standi": "right to bring a case",
    "habeas corpus": "a writ to produce a detained person",
    "mandamus": "a court order to a public official",
    "certiorari": "a court order to review a lower court's decision",
    "ultra vires": "beyond legal authority",
}


def simplify_inline(text: str) -> str:
    """Replace inline-safe jargon with plain-language equivalents."""
    result = text
    # Sort by length (longest first) to avoid partial matches
    for term in sorted(INLINE_SUBSTITUTIONS.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(term) + r"\b"
        replacement = INLINE_SUBSTITUTIONS[term]
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def extract_glossary(text: str) -> dict[str, str]:
    """Find glossary-only terms present in the text; return term -> definition."""
    found = {}
    for term, definition in GLOSSARY_ONLY.items():
        if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
            found[term] = definition
    return found

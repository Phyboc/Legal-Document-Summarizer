"""Legal jargon simplification dictionary. Inline-safe vs. glossary-only terms."""
import re


# 30 core inline-safe substitutions (in production would be 170+)
INLINE_SUBSTITUTIONS = {
    "appellant": "the person appealing",
    "respondent": "the other party",
    "petitioner": "the person filing the case",
    "plaintiff": "the person filing the case",
    "defendant": "the accused party",
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
    "learned counsel": "the lawyer",
    "moved this Court": "asked this court",
    "in the result": "in conclusion",
    "impugned order": "the order being challenged",
    "operative order": "the actual decision",
    "ratio decidendi": "the legal reasoning behind the decision",
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

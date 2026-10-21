"""Format aggregated section summaries into Lawyer Mode and Citizen Mode."""
import re
import sys
sys.path.insert(0, ".")
from config import (
    LAWYER_MODE_MIN_WORDS, LAWYER_MODE_MAX_WORDS,
    CITIZEN_MODE_MIN_WORDS, CITIZEN_MODE_MAX_WORDS,
)
from src.postprocess.jargon import simplify_inline, extract_glossary


LAWYER_SECTION_TITLES = {
    "facts": "FACTS",
    "arguments_appellant": "APPELLANT'S ARGUMENTS",
    "arguments_respondent": "RESPONDENT'S ARGUMENTS",
    "reasoning": "REASONING",
    "judgement": "JUDGEMENT",
    "full_document": "FULL JUDGMENT",
}

CITIZEN_SECTION_TITLES = {
    "facts": "What Happened",
    "arguments_appellant": "What The Appealing Side Said",
    "arguments_respondent": "What The Other Side Said",
    "reasoning": "Why The Court Decided This Way",
    "judgement": "The Court's Decision",
    "full_document": "The Whole Judgment",
}

SECTION_ORDER = [
    "facts", "arguments_appellant", "arguments_respondent", "reasoning",
    "judgement", "full_document",
]


def word_count(text: str) -> int:
    return len(text.split())


def trim_to_budget(text: str, max_words: int) -> str:
    """Trim to a max word budget without breaking sentences.

    Uses the abbreviation-aware splitter: the naive split treated "Nos." and
    "rr." as sentence ends, so trimming left fragments such as "...commanding
    the State Government to appoint respondents Nos." in the output.
    """
    from src.data.section_summary_alignment import summary_sentences

    sents = summary_sentences(text)
    out = []
    count = 0
    for s in sents:
        w = len(s.split())
        if count + w > max_words and out:
            break
        out.append(s)
        count += w
    return " ".join(out)


def allocate_budget(section_summaries: dict[str, str], budget: int,
                    protect: tuple[str, ...] = ("judgement",)) -> dict[str, str]:
    """Fit sections into ``budget`` words, reserving the protected sections first.

    The old scheme scaled every section by the same factor and trimmed each from
    its front, so the final order - at the tail of its block, or folded into the
    much larger reasoning block - was the first thing cut when a document ran over
    budget. Here the protected sections (the judgement by default) are given their
    words up front, and only the remaining budget is shared proportionally among
    the rest. If the protected sections alone exceed the budget they are trimmed
    to fit and the others are dropped: the operative order matters more than the
    recital of facts.
    """
    present = {s: t for s, t in section_summaries.items() if t and t.strip()}
    if sum(word_count(t) for t in present.values()) <= budget:
        return dict(section_summaries)

    protected = {s: present[s] for s in protect if s in present}
    others = {s: t for s, t in present.items() if s not in protected}
    protected_words = sum(word_count(t) for t in protected.values())
    others_words = sum(word_count(t) for t in others.values())

    remaining = budget - protected_words
    adjusted: dict[str, str] = {}
    if remaining <= 0 or others_words == 0:
        # Judgement alone fills (or overflows) the budget: keep it, trim to fit.
        for s, t in protected.items():
            adjusted[s] = trim_to_budget(t, budget) if word_count(t) > budget else t
        return adjusted

    scale = remaining / others_words
    for s, t in others.items():
        adjusted[s] = trim_to_budget(t, int(word_count(t) * scale))
    adjusted.update(protected)
    return adjusted


def format_lawyer_mode(section_summaries: dict[str, str]) -> dict:
    """Build Lawyer Mode: structured, technical, 500-700 words."""
    # main's SECTION_ORDER (which includes full_document) with the branch's
    # allocate_budget, the shared helper citizen mode also uses - it apportions
    # the word budget per section rather than trimming uniformly, protecting the
    # judgement section from being scaled away.
    section_order = SECTION_ORDER
    adjusted = allocate_budget(section_summaries, LAWYER_MODE_MAX_WORDS)

    sections_output = []
    for section in section_order:
        if section in adjusted and adjusted[section].strip():
            title = LAWYER_SECTION_TITLES[section]
            sections_output.append({
                "title": title,
                "section_key": section,
                "text": adjusted[section].strip(),
            })

    full_text = "\n\n".join(f"{s['title']}:\n{s['text']}" for s in sections_output)

    return {
        "mode": "lawyer",
        "sections": sections_output,
        "full_text": full_text,
        "word_count": word_count(full_text),
        "target_range": (LAWYER_MODE_MIN_WORDS, LAWYER_MODE_MAX_WORDS),
    }


def format_citizen_mode(section_summaries: dict[str, str]) -> dict:
    """Build Citizen Mode: simplified, plain language, 150-250 words."""
    section_order = SECTION_ORDER

    # Merge arguments into one for citizen mode
    combined = {}
    for section in section_order:
        if section in section_summaries and section_summaries[section].strip():
            simplified = simplify_inline(section_summaries[section])
            combined[section] = simplified

    # Fit word budget, reserving the court's decision first.
    combined = allocate_budget(combined, CITIZEN_MODE_MAX_WORDS)

    sections_output = []
    for section in section_order:
        if section in combined and combined[section].strip():
            sections_output.append({
                "title": CITIZEN_SECTION_TITLES[section],
                "section_key": section,
                "text": combined[section].strip(),
            })

    full_text = "\n\n".join(f"{s['title']}:\n{s['text']}" for s in sections_output)
    glossary = extract_glossary(" ".join(section_summaries.values()))

    return {
        "mode": "citizen",
        "sections": sections_output,
        "full_text": full_text,
        "word_count": word_count(full_text),
        "target_range": (CITIZEN_MODE_MIN_WORDS, CITIZEN_MODE_MAX_WORDS),
        "glossary": glossary,
    }


def format_both(section_summaries: dict[str, str]) -> dict:
    return {
        "lawyer": format_lawyer_mode(section_summaries),
        "citizen": format_citizen_mode(section_summaries),
    }

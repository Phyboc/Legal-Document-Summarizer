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
}

CITIZEN_SECTION_TITLES = {
    "facts": "What Happened",
    "arguments_appellant": "What The Appealing Side Said",
    "arguments_respondent": "What The Other Side Said",
    "reasoning": "Why The Court Decided This Way",
    "judgement": "The Court's Decision",
}


def word_count(text: str) -> int:
    return len(text.split())


def trim_to_budget(text: str, max_words: int) -> str:
    """Trim to a max word budget without breaking sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = []
    count = 0
    for s in sents:
        w = len(s.split())
        if count + w > max_words and out:
            break
        out.append(s)
        count += w
    return " ".join(out)


def format_lawyer_mode(section_summaries: dict[str, str]) -> dict:
    """Build Lawyer Mode: structured, technical, 500-700 words."""
    section_order = ["facts", "arguments_appellant", "arguments_respondent", "reasoning", "judgement"]
    total_words = sum(word_count(s) for s in section_summaries.values())
    budget = LAWYER_MODE_MAX_WORDS

    # Trim proportionally if we're over budget
    if total_words > budget:
        scale = budget / total_words
        adjusted = {sec: trim_to_budget(text, int(word_count(text) * scale))
                    for sec, text in section_summaries.items()}
    else:
        adjusted = dict(section_summaries)

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
    section_order = ["facts", "arguments_appellant", "arguments_respondent", "reasoning", "judgement"]

    # Merge arguments into one for citizen mode
    combined = {}
    for section in section_order:
        if section in section_summaries and section_summaries[section].strip():
            simplified = simplify_inline(section_summaries[section])
            combined[section] = simplified

    # Fit word budget
    total = sum(word_count(t) for t in combined.values())
    budget = CITIZEN_MODE_MAX_WORDS

    if total > budget:
        scale = budget / total
        combined = {sec: trim_to_budget(text, int(word_count(text) * scale))
                    for sec, text in combined.items()}

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

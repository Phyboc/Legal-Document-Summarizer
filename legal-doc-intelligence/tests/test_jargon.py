"""Tests for jargon simplification and Lawyer/Citizen differentiation
(``src/postprocess/jargon.py`` + ``src/postprocess/mode_formatter.py``).

Pins the fix for the "both modes are identical except headings" problem: the
base-form substitution patterns missed inflected/plural forms, so judgements
written in the plural got no Citizen-mode simplification at all. These run
offline (no model is involved in formatting).

Run:  python -m pytest tests/test_jargon.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.postprocess.jargon import extract_glossary, simplify_inline
from src.postprocess.mode_formatter import format_both


def test_plural_party_terms_are_simplified():
    assert simplify_inline("the appellants") == "the people appealing"
    assert simplify_inline("the respondents") == "the other parties"
    assert simplify_inline("the defendants") == "the accused parties"


def test_inflected_verbs_are_simplified():
    assert simplify_inline("The appellants were convicted.") == \
        "The people appealing were found guilty."
    assert "cancelled" in simplify_inline("The order was quashed.")
    assert "sent back" in simplify_inline("The case was remanded.")


def test_no_double_article_on_plural_party_terms():
    # "the appellants" must not become "the the people appealing".
    assert "the the" not in simplify_inline("judgment against the appellants")


def test_base_singular_forms_still_work():
    assert simplify_inline("the appellant filed") == "the person appealing filed"
    assert simplify_inline("counsel argued") == "lawyer argued"


def test_simplify_is_case_insensitive():
    assert simplify_inline("APPELLANTS").lower() == "people appealing"


def test_glossary_still_detects_terms():
    g = extract_glossary("The plea of res judicata was raised.")
    assert "res judicata" in g


def test_citizen_mode_now_differs_from_lawyer_when_jargon_present():
    # A judgement full of plural party terms: before the fix the two modes'
    # body text was byte-identical; now Citizen simplifies the vocabulary.
    sections = {
        "facts": "Two complaints were filed against the appellants.",
        "reasoning": "The appellants were convicted of an offence. The respondents did not object.",
        "judgement": "The appeal is dismissed.",
    }
    both = format_both(sections)
    lawyer = both["lawyer"]["full_text"]
    citizen = both["citizen"]["full_text"]

    # Lawyer keeps the legal vocabulary; Citizen replaces it.
    assert "appellants" in lawyer
    assert "appellants" not in citizen
    assert "people appealing" in citizen
    assert "found guilty" in citizen

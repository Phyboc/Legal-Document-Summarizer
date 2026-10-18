"""Tests for summarizer model selection, generation config, and output handling
(``src/model/summarizer.py`` + ``config.py``).

These pin the three immediate inference fixes from the forensic audit:

* I2 - the default model is the fine-tuned checkpoint, not ``facebook/bart-base``,
  and ``is_finetuned`` is reported against the resolved name.
* I1 / I3 - one explicit generation config in terms of ``*_new_tokens``, so the
  model's stale saved ``max_length: 20`` can never govern, and mid-sentence tails
  are trimmed.

They run fully offline: the model is never loaded (``transformers`` is imported
lazily inside ``Summarizer._load``), so nothing here downloads weights.

Run:  python -m pytest tests/test_summarizer_config.py -q
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.model import summarizer
from src.model.summarizer import (
    GENERATION_PROFILES,
    Summarizer,
    _trim_to_last_sentence,
    get_summarizer,
)


# --------------------------------------------------------------------------- #
# Model selection (I2)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    os.getenv("LEGAL_BART_MODEL") == config.BASE_BART_MODEL,
    reason="LEGAL_BART_MODEL explicitly pins the base model for this run",
)
def test_default_model_is_the_finetuned_checkpoint():
    assert config.BART_MODEL != config.BASE_BART_MODEL
    s = Summarizer()  # no argument: the CLI/API/Gradio path
    assert s.model_name == config.BART_MODEL
    assert s.is_finetuned is True


def test_base_model_is_reported_not_finetuned():
    assert Summarizer(config.BASE_BART_MODEL).is_finetuned is False


def test_explicit_finetuned_names_are_reported_finetuned():
    for name in ("fakechamp24/legallens-bart-base",
                 "models/legallens-bart-base-e3",
                 "/some/local/e3"):
        assert Summarizer(name).is_finetuned is True


def test_get_summarizer_none_resolves_to_default():
    assert get_summarizer().model_name == config.BART_MODEL


def test_get_summarizer_caches_per_resolved_model():
    # Same name -> same instance; different name -> different instance, so two
    # models can be compared in one process (used by the eval harness).
    a1 = get_summarizer("facebook/bart-base")
    a2 = get_summarizer("facebook/bart-base")
    b = get_summarizer("fakechamp24/legallens-bart-base")
    assert a1 is a2
    assert a1 is not b


# --------------------------------------------------------------------------- #
# Generation configuration (I1 / I3)
# --------------------------------------------------------------------------- #

def test_every_section_has_a_profile():
    for section in config.SECTIONS + ["default"]:
        assert section in GENERATION_PROFILES


def test_profiles_use_new_token_budgets_only():
    # The whole point of the fix: no profile may carry the old max_length/
    # min_length keys that collided with the saved generation_config.
    for section, profile in GENERATION_PROFILES.items():
        assert "max_new_tokens" in profile, section
        assert "min_new_tokens" in profile, section
        assert "max_length" not in profile, section
        assert "min_length" not in profile, section


def test_profile_budgets_are_sane():
    for section, profile in GENERATION_PROFILES.items():
        lo, hi = profile["min_new_tokens"], profile["max_new_tokens"]
        assert 0 < lo < hi, section
        # stay within what the model was actually supervised to produce
        assert hi <= config.MAX_TARGET_LENGTH, section


def test_shared_generation_constants_are_explicit():
    assert config.NUM_BEAMS == 4
    assert config.NO_REPEAT_NGRAM_SIZE == 3
    assert isinstance(config.GENERATION_LENGTH_PENALTY, float)
    assert config.GENERATION_EARLY_STOPPING is True


# --------------------------------------------------------------------------- #
# Output handling - mid-sentence trimming (I1)
# --------------------------------------------------------------------------- #

def test_trim_drops_an_incomplete_trailing_clause():
    text = "The appeal is allowed. The court held that the"
    assert _trim_to_last_sentence(text) == "The appeal is allowed."


def test_trim_keeps_already_complete_text():
    text = "The appeal is dismissed. No order as to costs."
    assert _trim_to_last_sentence(text) == text


def test_trim_accepts_terminal_punctuation_before_a_closing_quote():
    text = 'The court observed "the order is set aside."'
    assert _trim_to_last_sentence(text) == text


def test_trim_leaves_a_boundaryless_fragment_intact():
    # Better a whole fragment than an empty string.
    text = "a single clause with no sentence end"
    assert _trim_to_last_sentence(text) == text


def test_trim_handles_empty_and_whitespace():
    assert _trim_to_last_sentence("") == ""
    assert _trim_to_last_sentence("   ") == ""


def test_trim_strips_surrounding_whitespace():
    assert _trim_to_last_sentence("  Done.  ") == "Done."


def test_fallback_uses_the_new_token_budget_key():
    # The extractive fallback keys off the profile too; make sure the rename
    # didn't leave it reading a now-missing key. First-sentence budget applied.
    out = summarizer._extractive_fallback("One. Two. Three.", 1)
    assert out == "One."

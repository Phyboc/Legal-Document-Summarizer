"""BART summarizer wrapper: per-chunk generation with section-aware profiles."""
import re
import sys
sys.path.insert(0, ".")
from config import (
    BART_MODEL,
    BASE_BART_MODEL,
    MAX_INPUT_LENGTH,
    NUM_BEAMS,
    NO_REPEAT_NGRAM_SIZE,
    GENERATION_LENGTH_PENALTY,
    GENERATION_EARLY_STOPPING,
)


# Per-section generation budgets, expressed as *new* tokens so the model's saved
# max_length (20) can never clamp us. Budgets are deliberately section-specific,
# not uniformly inflated: the model was supervised on aligned targets up to 448
# tokens that always end on a sentence boundary, so the old 100-250 caps sat
# below the lengths it learned to produce and forced the EOS mid-sentence. These
# caps give room to reach a natural end while staying well under 448 and matching
# each section's typical length (judgement = short operative order; reasoning =
# longest; arguments = thin).
GENERATION_PROFILES = {
    "facts": {"min_new_tokens": 50, "max_new_tokens": 320},
    "arguments_appellant": {"min_new_tokens": 40, "max_new_tokens": 200},
    "arguments_respondent": {"min_new_tokens": 40, "max_new_tokens": 200},
    "reasoning": {"min_new_tokens": 60, "max_new_tokens": 400},
    "judgement": {"min_new_tokens": 20, "max_new_tokens": 160},
    "default": {"min_new_tokens": 40, "max_new_tokens": 320},
}


class Summarizer:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or BART_MODEL
        self.tokenizer = None
        self.model = None
        self._loaded = False
        # Judge the resolved name, not the argument: the API and CLI never pass
        # one, so a fine-tuned LEGAL_BART_MODEL used to report False.
        self.is_finetuned = self.model_name != BASE_BART_MODEL

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            import torch
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            self.model.eval()
            self._loaded = True
        except (ImportError, OSError) as e:
            print(f"[Summarizer] Warning: couldn't load {self.model_name}: {e}")
            print("[Summarizer] Using extractive fallback.")
            self._loaded = False

    def summarize_chunk(self, text: str, section: str = "default") -> str:
        """Summarize a single chunk with section-appropriate profile."""
        self._load()
        profile = GENERATION_PROFILES.get(section, GENERATION_PROFILES["default"])

        if not self._loaded:
            return _extractive_fallback(text, profile["max_new_tokens"])

        inputs = self.tokenizer(
            text, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        # One explicit generation config. max_new_tokens (not max_length) means the
        # saved generation_config.json max_length:20 can't clamp the output.
        output_ids = self.model.generate(
            **inputs,
            min_new_tokens=profile["min_new_tokens"],
            max_new_tokens=profile["max_new_tokens"],
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            length_penalty=GENERATION_LENGTH_PENALTY,
            early_stopping=GENERATION_EARLY_STOPPING,
        )
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return _trim_to_last_sentence(summary)

    def summarize_chunks(self, chunks: list) -> list[dict]:
        """Summarize each chunk; returns list of {chunk_id, section, summary, source_text}."""
        results = []
        for chunk in chunks:
            summary = self.summarize_chunk(chunk.text, chunk.section)
            results.append({
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "summary": summary,
                "source_text": chunk.text,
                "para_range": chunk.para_range,
            })
        return results


# Terminal punctuation at the very end of a (stripped) string, allowing a
# trailing closing quote/bracket, e.g. ... order."  or  ... (allowed).
_ENDS_SENTENCE = re.compile(r'[.!?][\'")\]]?$')
# Any sentence-ending punctuation, used to find the last complete boundary.
_SENTENCE_BOUNDARY = re.compile(r'[.!?][\'")\]]?')


def _trim_to_last_sentence(text: str) -> str:
    """Drop a trailing clause left incomplete when generation stops at the cap.

    The summarizer is extractive, so a chunk whose generation reaches
    max_new_tokens can end mid-sentence. We trim back to the last complete
    sentence rather than emit a fragment. This is a formatting safeguard, not a
    rewrite - it never makes the output more abstractive, it only removes a
    dangling tail. If the text has no sentence boundary at all it is returned
    unchanged (a whole fragment beats an empty string).
    """
    s = text.strip()
    if not s or _ENDS_SENTENCE.search(s):
        return s
    matches = list(_SENTENCE_BOUNDARY.finditer(s))
    if not matches:
        return s
    return s[: matches[-1].end()].strip()


def _extractive_fallback(text: str, max_words: int) -> str:
    """Take the first N sentences up to word budget."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = []
    word_count = 0
    for s in sents:
        w = len(s.split())
        if word_count + w > max_words and out:
            break
        out.append(s)
        word_count += w
    return " ".join(out)


_summarizer_instances: dict[str, Summarizer] = {}


def get_summarizer(model_name: str = None) -> Summarizer:
    """Return a cached Summarizer per resolved model, so two models can be compared in one process."""
    key = model_name or BART_MODEL
    if key not in _summarizer_instances:
        _summarizer_instances[key] = Summarizer(key)
    return _summarizer_instances[key]

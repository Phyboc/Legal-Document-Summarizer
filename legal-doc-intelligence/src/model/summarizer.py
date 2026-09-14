"""BART summarizer wrapper: per-chunk generation with section-aware profiles."""
import sys
sys.path.insert(0, ".")
from config import BART_MODEL, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, NUM_BEAMS


GENERATION_PROFILES = {
    "facts": {"min_length": 50, "max_length": 200, "num_beams": NUM_BEAMS},
    "arguments_appellant": {"min_length": 40, "max_length": 150, "num_beams": NUM_BEAMS},
    "arguments_respondent": {"min_length": 40, "max_length": 150, "num_beams": NUM_BEAMS},
    "reasoning": {"min_length": 60, "max_length": 250, "num_beams": NUM_BEAMS},
    "judgement": {"min_length": 20, "max_length": 100, "num_beams": NUM_BEAMS},
    "default": {"min_length": 40, "max_length": 200, "num_beams": NUM_BEAMS},
}


class Summarizer:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or BART_MODEL
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self.is_finetuned = model_name is not None and model_name != "facebook/bart-base"

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
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
            return _extractive_fallback(text, profile["max_length"])

        inputs = self.tokenizer(
            text, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt"
        )
        output_ids = self.model.generate(
            **inputs,
            min_length=profile["min_length"],
            max_length=profile["max_length"],
            num_beams=profile["num_beams"],
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary

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


def _extractive_fallback(text: str, max_words: int) -> str:
    """Take the first N sentences up to word budget."""
    import re
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


_summarizer_instance = None


def get_summarizer(model_name: str = None) -> Summarizer:
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = Summarizer(model_name)
    return _summarizer_instance

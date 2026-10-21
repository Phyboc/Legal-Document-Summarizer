"""Shared NLI (entailment) cross-encoder for Cite & Verify.

The MS-MARCO cross-encoder in ``cross_encoder.py`` scores *topical relevance*:
every sentence of a summary is relevant to its own judgement, so it rated ~98.5%
of citations "verified" at every threshold (``artifacts/citation_distribution.json``)
and could not tell a supported sentence from a hallucinated one. An NLI model
answers the question Cite & Verify actually asks - does this source passage
*entail* the sentence - so it is used for the support decision, while MS-MARCO
is kept for what it is good at: picking which chunk a sentence came from.
"""
import sys

sys.path.insert(0, ".")
from config import CITATION_NLI_MODEL

_model = None
_unavailable = False


def get_nli_model():
    """Load the NLI cross-encoder once; None if it cannot be loaded."""
    global _model, _unavailable
    if _model is None and not _unavailable:
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(CITATION_NLI_MODEL)
        except Exception as exc:  # sentence-transformers missing or model unreachable
            print(f"[nli] Warning: couldn't load {CITATION_NLI_MODEL}: {exc}")
            _unavailable = True
    return _model


def label_index(model, name: str) -> int:
    """Index of an NLI label ('entailment', 'contradiction', 'neutral') in the model output."""
    for idx, label in model.config.id2label.items():
        if label.lower() == name:
            return int(idx)
    raise KeyError(f"{name!r} not in {model.config.id2label}")


def entailment_scores(pairs: list[tuple[str, str]], batch_size: int = 32) -> list[dict]:
    """Score (premise, hypothesis) pairs; returns [{'entailment': p, 'contradiction': p, 'neutral': p}].

    Returns an empty list when the model is unavailable so callers can fall back.
    """
    model = get_nli_model()
    if model is None or not pairs:
        return []
    probs = model.predict(pairs, apply_softmax=True, batch_size=batch_size, show_progress_bar=False)
    names = {name: label_index(model, name) for name in ("entailment", "contradiction", "neutral")}
    return [{name: float(row[i]) for name, i in names.items()} for row in probs]

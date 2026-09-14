"""Shared Cross-Encoder singleton for alignment and citation verification."""
import math
import sys
sys.path.insert(0, ".")
from config import CROSS_ENCODER_MODEL


class CrossEncoderSingleton:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(CROSS_ENCODER_MODEL)
            except ImportError:
                self._model = _MockCrossEncoder()
        return self._model

    def score(self, pairs: list[tuple[str, str]], normalize: bool = True) -> list[float]:
        """Score (query, passage) pairs. Returns floats in [0,1] if normalize=True."""
        model = self.get_model()
        scores = model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = list(scores) if hasattr(scores, "__iter__") else [scores]

        if normalize:
            scores = [_sigmoid(s) for s in scores]
        return scores


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class _MockCrossEncoder:
    """Fallback that returns bag-of-words overlap when transformers isn't available."""
    def predict(self, pairs):
        results = []
        for query, passage in pairs:
            q_tokens = set(query.lower().split())
            p_tokens = set(passage.lower().split())
            if not q_tokens:
                results.append(0.0)
                continue
            overlap = len(q_tokens & p_tokens) / len(q_tokens)
            # Map [0,1] overlap to logit range
            results.append((overlap - 0.5) * 6)
        return results


def get_cross_encoder():
    return CrossEncoderSingleton()

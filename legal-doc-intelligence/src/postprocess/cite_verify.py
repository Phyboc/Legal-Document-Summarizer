"""Citation and verification: match each summary sentence to source text with confidence."""
import re
import sys
sys.path.insert(0, ".")
from config import CITATION_THRESHOLD, CITATION_HIGH_CONFIDENCE
from src.shared.cross_encoder import get_cross_encoder


def split_sentences(text: str) -> list[str]:
    """Split into sentences; keep sentences that are meaningful (>=3 words)."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.strip().split()) >= 3]


def cite_and_verify(summary_text: str, chunk_summaries: list[dict]) -> list[dict]:
    """Match each summary sentence to a source chunk and score confidence."""
    ce = get_cross_encoder()
    sentences = split_sentences(summary_text)

    if not sentences or not chunk_summaries:
        return []

    results = []
    for sent in sentences:
        pairs = [(sent, cs["source_text"]) for cs in chunk_summaries]
        scores = ce.score(pairs, normalize=True)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        results.append({
            "sentence": sent,
            "confidence": round(best_score, 3),
            "verified": best_score >= CITATION_THRESHOLD,
            "high_confidence": best_score >= CITATION_HIGH_CONFIDENCE,
            "source_chunk_id": chunk_summaries[best_idx]["chunk_id"],
            "source_section": chunk_summaries[best_idx]["section"],
            "source_para_range": chunk_summaries[best_idx].get("para_range"),
        })

    return results


def summarize_verification(citations: list[dict]) -> dict:
    """Overall verification statistics."""
    if not citations:
        return {"total": 0, "verified": 0, "verified_rate": 0.0, "avg_confidence": 0.0}

    verified = sum(1 for c in citations if c["verified"])
    high = sum(1 for c in citations if c["high_confidence"])
    avg = sum(c["confidence"] for c in citations) / len(citations)

    return {
        "total": len(citations),
        "verified": verified,
        "high_confidence": high,
        "verified_rate": round(verified / len(citations), 3),
        "avg_confidence": round(avg, 3),
    }

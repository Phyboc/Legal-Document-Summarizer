"""Aggregate chunk summaries into section-level and document-level summaries."""
import re


def dedupe_sentences(text: str) -> str:
    """Remove exact duplicate sentences that arise from overlapping chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    kept = []
    for s in sentences:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(s)
    return " ".join(kept)


def aggregate_by_section(chunk_summaries: list[dict]) -> dict[str, str]:
    """Merge chunk summaries within each section into one coherent block."""
    by_section = {}
    for cs in chunk_summaries:
        by_section.setdefault(cs["section"], []).append(cs["summary"])

    aggregated = {}
    for section, summaries in by_section.items():
        combined = " ".join(summaries)
        aggregated[section] = dedupe_sentences(combined)
    return aggregated


def aggregate_document(chunk_summaries: list[dict]) -> str:
    """Combine all chunk summaries into a single document-level summary."""
    if not chunk_summaries:
        return ""
    all_text = " ".join(cs["summary"] for cs in chunk_summaries)
    return dedupe_sentences(all_text)

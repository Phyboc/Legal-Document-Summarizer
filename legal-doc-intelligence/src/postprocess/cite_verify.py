"""Citation and verification: match each summary sentence to its source and check it is supported.

Two stages, each using the model suited to it:

1. **Cite** - the shared MS-MARCO cross-encoder ranks the document's chunks by
   relevance to the sentence; the best one is the citation.
2. **Verify** - an NLI cross-encoder (``src.shared.nli``) checks whether the top
   ``CITATION_TOP_K`` source chunks *entail* the sentence. Chunks are longer than
   the NLI window, so each is scored as overlapping ~180-word windows and the
   best window counts. ``confidence`` is that entailment probability.

The MS-MARCO score alone could not verify anything: it measures topical
relevance, and every sentence of a summary is relevant to its own judgement
(98.5% "verified" at every threshold). Thresholds are calibrated by
``scripts/calibrate_citations.py`` (``artifacts/citation_calibration.json``).

If the NLI model cannot be loaded, verification falls back to the relevance
score and every citation is marked ``method: "relevance"`` so the weaker
guarantee is visible rather than silent.
"""
import re
import sys

sys.path.insert(0, ".")
from config import (
    CITATION_CONTRADICTION_THRESHOLD,
    CITATION_HIGH_CONFIDENCE,
    CITATION_THRESHOLD,
    CITATION_TOP_K,
    CITATION_WINDOW_WORDS,
)
from src.data.section_summary_alignment import summary_sentences
from src.postprocess.mode_formatter import CITIZEN_SECTION_TITLES, LAWYER_SECTION_TITLES
from src.shared.cross_encoder import get_cross_encoder
from src.shared.nli import entailment_scores, get_nli_model


_SECTION_HEADINGS = {f"{t}:" for t in (*LAWYER_SECTION_TITLES.values(), *CITIZEN_SECTION_TITLES.values())}


def strip_section_headings(text: str) -> str:
    """Drop the "FACTS:" / "What Happened:" lines the mode formatter puts in ``full_text``.

    Left in, they were glued onto the first sentence of each section, which then
    failed verification ("FACTS: The State Government issued...").
    """
    return "\n".join(line for line in text.splitlines() if line.strip() not in _SECTION_HEADINGS)


def split_sentences(text: str) -> list[str]:
    """Split into sentences; keep sentences that are meaningful (>=3 words).

    Uses the abbreviation-aware splitter: the naive punctuation split cut
    generated sentences at "Rs." / "No." / "v." and then failed to verify the
    fragments ("...was a fine of Rs.").
    """
    blocks = re.split(r"\n\s*\n", text)  # a blank line always ends a sentence
    return [s for block in blocks for s in summary_sentences(block) if len(s.split()) >= 3]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(text: str, n: int = 3) -> set:
    tokens = _tokens(text)
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


# Words whose omission changes what a sentence asserts, so a "copied with words
# dropped" match may never skip them.
_NEVER_SKIP = frozenset({"not", "no", "never", "nor", "neither", "without", "cannot", "nothing", "none"})
_MAX_GAP = 3


def _is_compressed_copy(sentence: list[str], source: list[str], max_gap: int = _MAX_GAP) -> bool:
    """True if ``sentence`` appears in ``source`` in order, with at most ``max_gap``
    source words skipped between consecutive matches and no negation skipped.

    This accepts what the summarizer does when it copies - dropping an article
    or a parenthetical ("of the Kerala State" -> "of Kerala State") - while any
    *inserted* or *changed* word (a different number, an added "not", "allowed"
    for "dismissed") has nothing to match and fails.
    """
    first = sentence[0]
    for start, tok in enumerate(source):
        if tok != first:
            continue
        pos, ok = start, True
        for word in sentence[1:]:
            limit = min(len(source), pos + 2 + max_gap)
            nxt = next((k for k in range(pos + 1, limit) if source[k] == word), None)
            if nxt is None or any(t in _NEVER_SKIP for t in source[pos + 1:nxt]):
                ok = False
                break
            pos = nxt
        if ok:
            return True
    return False


def verbatim_support(sentence: str, sources: list[str], source_ngrams: list[set] | None = None) -> int | None:
    """Index of a source that contains ``sentence`` verbatim (or as a compressed copy), else None.

    The fine-tuned model copies many sentences verbatim, and NLI models score a
    verbatim sentence against a long passage unreliably (0.03 entailment for a
    sentence present word-for-word). Exact containment of every word-trigram is
    tried first; then a compressed copy (words dropped, none changed or added,
    no negation dropped). Either way one changed number, an inserted "not" or a
    flipped outcome fails, so altered sentences still go to the NLI check.
    """
    tokens = _tokens(sentence)
    if len(tokens) < 4:
        return None
    grams = {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}
    source_ngrams = source_ngrams or [_ngrams(src) for src in sources]
    for j, src in enumerate(source_ngrams):
        if grams <= src:
            return j
    for j, src in enumerate(sources):
        if _is_compressed_copy(tokens, _tokens(src)):
            return j
    return None


def source_windows(text: str, max_words: int = CITATION_WINDOW_WORDS) -> list[str]:
    """Overlapping sentence windows of at most ~``max_words`` words (half-window stride).

    Uses the abbreviation-aware splitter so a window never starts mid-citation.
    """
    sentences = summary_sentences(text)
    if not sentences:
        return [text] if text.strip() else []
    windows, start = [], 0
    while start < len(sentences):
        words, end = 0, start
        while end < len(sentences) and (end == start or words + len(sentences[end].split()) <= max_words):
            words += len(sentences[end].split())
            end += 1
        windows.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        # advance by about half the window, always at least one sentence
        half, acc, nxt = words / 2, 0, start
        while nxt < end - 1 and acc < half:
            acc += len(sentences[nxt].split())
            nxt += 1
        start = max(nxt, start + 1)
    return windows


def verify_sentences(sentences: list[str], sources: list[str], top_k: int = CITATION_TOP_K) -> list[dict]:
    """Cite and verify ``sentences`` against ``sources`` (one string per chunk).

    Returns, per sentence: ``source_index`` (best-relevance chunk),
    ``retrieval_score``, ``entailment``, ``contradiction`` and ``method``.
    Shared by the pipeline and by the calibration script, so the calibrated
    thresholds apply to exactly this computation.
    """
    if not sentences or not sources:
        return []

    ce = get_cross_encoder()
    relevance = ce.score([(s, src) for s in sentences for src in sources], normalize=True)
    n = len(sources)
    ranked = []
    for i in range(len(sentences)):
        row = relevance[i * n:(i + 1) * n]
        order = sorted(range(n), key=lambda j: row[j], reverse=True)
        ranked.append((order, row))

    source_ngrams = [_ngrams(src) for src in sources]
    verbatim = [verbatim_support(s, sources, source_ngrams) for s in sentences]

    windows_by_source = {}
    pairs, owners = [], []
    use_nli = get_nli_model() is not None
    if use_nli:
        for i, (order, _) in enumerate(ranked):
            if verbatim[i] is not None:
                continue  # already proven by exact containment; skip the NLI cost
            for j in order[:top_k]:
                if j not in windows_by_source:
                    windows_by_source[j] = source_windows(sources[j])
                for window in windows_by_source[j]:
                    pairs.append((window, sentences[i]))
                    owners.append(i)
    scores = entailment_scores(pairs) if use_nli else []

    best = [{"entailment": 0.0, "contradiction": 0.0} for _ in sentences]
    for owner, s in zip(owners, scores):
        if s["entailment"] > best[owner]["entailment"]:
            best[owner]["entailment"] = s["entailment"]
        if s["contradiction"] > best[owner]["contradiction"]:
            best[owner]["contradiction"] = s["contradiction"]

    results = []
    for i, (order, row) in enumerate(ranked):
        top = order[0]
        if verbatim[i] is not None:
            # Cite the chunk that actually contains the sentence.
            top, entail, contra, evidence = verbatim[i], 1.0, 0.0, "verbatim"
        elif use_nli:
            entail, contra, evidence = best[i]["entailment"], best[i]["contradiction"], "entailment"
        else:
            entail, contra, evidence = row[top], 0.0, "relevance"
        results.append({
            "source_index": top,
            "retrieval_score": float(row[top]),
            "entailment": float(entail),
            "contradiction": float(contra),
            "evidence": evidence,
            "method": "nli" if use_nli else "relevance",
        })
    return results


def support_label(entailment: float, contradiction: float) -> str:
    if entailment >= CITATION_THRESHOLD:
        return "supported"
    if contradiction >= CITATION_CONTRADICTION_THRESHOLD:
        return "contradicted"
    return "unsupported"


def cite_and_verify(summary_text: str, chunk_summaries: list[dict]) -> list[dict]:
    """Match each summary sentence to a source chunk and check the chunk supports it."""
    sentences = split_sentences(strip_section_headings(summary_text))
    if not sentences or not chunk_summaries:
        return []

    checks = verify_sentences(sentences, [cs["source_text"] for cs in chunk_summaries])
    results = []
    for sent, check in zip(sentences, checks):
        source = chunk_summaries[check["source_index"]]
        confidence = check["entailment"]
        results.append({
            "sentence": sent,
            "confidence": round(confidence, 3),
            "verified": confidence >= CITATION_THRESHOLD,
            "high_confidence": confidence >= CITATION_HIGH_CONFIDENCE,
            "support": support_label(confidence, check["contradiction"]),
            "contradiction": round(check["contradiction"], 3),
            "retrieval_score": round(check["retrieval_score"], 3),
            "method": check["method"],
            "evidence": check["evidence"],
            "source_chunk_id": source["chunk_id"],
            "source_section": source["section"],
            "source_para_range": source.get("para_range"),
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
        "contradicted": sum(1 for c in citations if c.get("support") == "contradicted"),
        "verified_rate": round(verified / len(citations), 3),
        "avg_confidence": round(avg, 3),
        "method": citations[0].get("method", "relevance"),
    }

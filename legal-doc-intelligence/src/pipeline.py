"""Main pipeline: orchestrates all stages from raw judgment to Lawyer/Citizen output."""
import sys
sys.path.insert(0, ".")
from src.data.cleaning import preprocess, split_paragraphs
from src.data.pii_redaction import redact
from src.data.section_detector import detect_sections, build_section_groups
from src.data.chunker import chunk_document
from src.model.summarizer import get_summarizer
from src.postprocess.aggregation import aggregate_by_section
from src.postprocess.mode_formatter import format_both
from src.postprocess.cite_verify import cite_and_verify, summarize_verification


def process_document(text: str, redact_names: bool = False,
                     strip_preamble: bool = True, model_name: str = None) -> dict:
    """Run the full pipeline on a raw judgment. Returns a dict with everything."""
    # Preprocess
    clean = preprocess(text, strip_preamble_flag=strip_preamble)

    # PII redaction
    redacted, pii_stats = redact(clean, redact_names_flag=redact_names)

    # Split into paragraphs
    paragraphs = split_paragraphs(redacted)

    # Detect sections, promote the operative order to judgement, and group. The
    # grouping is shared with the training-data build so both see the same labels.
    para_objects = detect_sections(paragraphs)
    section_groups, _ = build_section_groups(para_objects)

    # Chunk
    chunks = chunk_document(section_groups)

    # Summarize
    summarizer = get_summarizer(model_name)
    chunk_summaries = summarizer.summarize_chunks(chunks)

    # Aggregate per section
    section_summaries = aggregate_by_section(chunk_summaries)

    # Format both modes
    modes = format_both(section_summaries)

    # Cite & verify each mode
    lawyer_citations = cite_and_verify(modes["lawyer"]["full_text"], chunk_summaries)
    citizen_citations = cite_and_verify(modes["citizen"]["full_text"], chunk_summaries)

    return {
        "input_length": len(text),
        "cleaned_length": len(clean),
        "n_paragraphs": len(paragraphs),
        "n_chunks": len(chunks),
        "sections_detected": list(section_groups.keys()),
        "pii_stats": pii_stats,
        "is_finetuned": summarizer.is_finetuned,
        "lawyer_mode": modes["lawyer"],
        "citizen_mode": modes["citizen"],
        "lawyer_citations": lawyer_citations,
        "citizen_citations": citizen_citations,
        "lawyer_verification": summarize_verification(lawyer_citations),
        "citizen_verification": summarize_verification(citizen_citations),
        "chunk_summaries": chunk_summaries,
    }

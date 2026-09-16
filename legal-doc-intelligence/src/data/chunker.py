"""Section-aware chunking with overlap."""
import re
from dataclasses import dataclass, field
import sys
sys.path.insert(0, ".")
from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC


@dataclass
class Chunk:
    text: str
    section: str
    chunk_id: int
    token_count: int
    para_range: tuple[int, int]  # (start_para_index, end_para_index)


def approximate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 words per token."""
    words = text.split()
    return int(len(words) * 1.33)


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split text into token-bounded pieces."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = approximate_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    # Hard split anything still over the limit
    result = []
    for chunk in chunks:
        if approximate_tokens(chunk) <= max_tokens:
            result.append(chunk)
        else:
            words = chunk.split()
            step = max(int(max_tokens / 1.33), 100)
            for i in range(0, len(words), step):
                result.append(" ".join(words[i:i+step]))
    return result


def chunk_section(paragraphs: list, section: str, chunk_size: int = CHUNK_SIZE,
                  overlap: int = CHUNK_OVERLAP, start_chunk_id: int = 0) -> list[Chunk]:
    """Chunk a single section's paragraphs into overlapping windows."""
    if not paragraphs:
        return []

    chunks = []
    current_text = []
    current_tokens = 0
    current_start_para = paragraphs[0].index
    chunk_id = start_chunk_id

    for para in paragraphs:
        para_tokens = approximate_tokens(para.text)

        if current_tokens + para_tokens > chunk_size and current_text:
            text = "\n".join(current_text)
            chunks.append(Chunk(
                text=text, section=section, chunk_id=chunk_id,
                token_count=current_tokens,
                para_range=(current_start_para, para.index - 1),
            ))
            chunk_id += 1

            # Overlap: keep last chunk's tail
            tail_text = text[-overlap*4:]  # approx 4 chars/token
            current_text = [tail_text, para.text]
            current_tokens = approximate_tokens(tail_text) + para_tokens
            current_start_para = para.index
        else:
            current_text.append(para.text)
            current_tokens += para_tokens

    if current_text:
        chunks.append(Chunk(
            text="\n".join(current_text), section=section, chunk_id=chunk_id,
            token_count=current_tokens,
            para_range=(current_start_para, paragraphs[-1].index),
        ))

    return chunks


def chunk_document(section_groups: dict, strategy: str = "proportional") -> list[Chunk]:
    """Chunk all sections, respecting the max-chunks cap."""
    all_chunks = []
    chunk_id = 0

    for section, paras in section_groups.items():
        section_chunks = chunk_section(paras, section, start_chunk_id=chunk_id)
        all_chunks.extend(section_chunks)
        chunk_id += len(section_chunks)

    # Apply cap
    if len(all_chunks) <= MAX_CHUNKS_PER_DOC:
        return all_chunks

    if strategy == "first_n":
        return all_chunks[:MAX_CHUNKS_PER_DOC]

    # Proportional allocation with floor of 1 per section
    from collections import Counter
    counts_per_section = Counter(c.section for c in all_chunks)
    total = len(all_chunks)
    n_sections = len(counts_per_section)

    # Guarantee 1 chunk per section
    remaining = MAX_CHUNKS_PER_DOC - n_sections
    allocations = {sec: 1 for sec in counts_per_section}

    if remaining > 0:
        for sec, count in counts_per_section.items():
            proportional = int(round((count / total) * remaining))
            allocations[sec] += proportional

    # Trim to cap
    while sum(allocations.values()) > MAX_CHUNKS_PER_DOC:
        largest = max(allocations, key=allocations.get)
        allocations[largest] -= 1

    selected = []
    for sec in counts_per_section:
        section_chunks = [c for c in all_chunks if c.section == sec]
        selected.extend(section_chunks[:allocations[sec]])

    return selected

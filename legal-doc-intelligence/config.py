"""LegalLens configuration - thresholds, budgets, model IDs."""
import os

# Models
BART_MODEL = os.getenv("LEGAL_BART_MODEL", "facebook/bart-base")
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SPACY_MODEL = "en_core_web_sm"

# Chunking
CHUNK_SIZE = 850
CHUNK_OVERLAP = 100
MAX_CHUNKS_PER_DOC = 15

# Alignment
ALIGNMENT_THRESHOLD = 0.40
ALIGNMENT_RELATIVE_MARGIN = 0.90

# Evidence matching. The Cross-Encoder truncates its input (ms-marco MiniLM:
# 512 sub-word tokens), while a chunk is budgeted at CHUNK_SIZE (850 estimated
# tokens). Scoring a whole chunk therefore makes the chunk's tail invisible and
# can hand a summary sentence to the wrong chunk. Chunks are scored as
# overlapping windows that fit the encoder instead, and each sentence is only
# compared against the few chunks a cheap lexical prefilter ranks highest.
ALIGNMENT_WINDOW_TOKENS = int(os.getenv("LEGAL_ALIGNMENT_WINDOW_TOKENS", "350"))
ALIGNMENT_WINDOW_OVERLAP = int(os.getenv("LEGAL_ALIGNMENT_WINDOW_OVERLAP", "50"))
ALIGNMENT_TOP_K_CANDIDATES = int(os.getenv("LEGAL_ALIGNMENT_TOP_K", "5"))
ALIGNMENT_USE_LEXICAL_PREFILTER = os.getenv("LEGAL_ALIGNMENT_PREFILTER", "1") == "1"

# Cross-Encoder inference batch size. The sentence-transformers default (32) is
# tuned for short web passages; 512-token pairs on a CPU/GPU benefit from larger
# batches. Recorded per build, so a benchmark is reproducible.
CROSS_ENCODER_BATCH_SIZE = int(os.getenv("LEGAL_CE_BATCH_SIZE", "64"))

# Evidence-aware alignment layer (src/data/evidence_alignment.py).
#
# OFF by default, deliberately: the layer is implemented but NOT calibrated yet,
# so the pure Cross-Encoder behaviour stays available for a like-for-like
# comparison. Calibrate with scripts/audit_alignment_evidence.py, then set these.
# The numeric values below are UNCALIBRATED placeholders (see EvidenceSettings).
ALIGNMENT_EVIDENCE_MODE = False
ALIGNMENT_EVIDENCE_COVERAGE_BOUND = 0.50
ALIGNMENT_EVIDENCE_AMBIGUITY_MARGIN = 0.02
ALIGNMENT_EVIDENCE_MIN_SPAN = 4
ALIGNMENT_EVIDENCE_MAX_RARE_DF = 1
ALIGNMENT_EVIDENCE_CONTRADICTION_MODE = "high_confidence"

# Summary lengths
LAWYER_MODE_MIN_WORDS = 500
LAWYER_MODE_MAX_WORDS = 700
CITIZEN_MODE_MIN_WORDS = 150
CITIZEN_MODE_MAX_WORDS = 250

# Generation
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 448
NUM_BEAMS = 4

# Citation
CITATION_THRESHOLD = 0.30
CITATION_HIGH_CONFIDENCE = 0.65

# Sections
SECTIONS = ["facts", "arguments_appellant", "arguments_respondent", "reasoning", "judgement"]

# Label used when structure detection cannot resolve a document. The dataset
# builder and the inference pipeline MUST agree on this string: a model trained
# on `full_document` records is never asked to summarise a section inference
# calls `reasoning`, so the mismatch silently wastes that training signal.
FALLBACK_SECTION_LABEL = "full_document"

# PII patterns
PII_PATTERNS = {
    "aadhaar": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
}

# Paths
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

REDACT_NAMES_DEFAULT = False
WARM_START = os.getenv("LEGAL_WARM_START", "0") == "1"

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

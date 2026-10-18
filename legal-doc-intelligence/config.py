"""LegalLens configuration - thresholds, budgets, model IDs."""
import os

# Models
# The un-fine-tuned base checkpoint. Kept as a named constant so the summarizer
# can tell base from fine-tuned, and so LEGAL_BART_MODEL=facebook/bart-base is an
# explicit, supported way to run the original zero-shot behaviour.
BASE_BART_MODEL = "facebook/bart-base"
_LOCAL_BART = os.path.join(os.path.dirname(__file__), "models", "legallens-bart-base-e3")
# Fine-tuned LegalLens summarizer. Prefer a local checkpoint (models/
# legallens-bart-base-e3) when present, otherwise the published Hub repo; override
# with LEGAL_BART_MODEL. The default is the fine-tuned model, not the base model,
# so CLI/API/Gradio all run the fine-tune unless explicitly told otherwise.
BART_MODEL = os.getenv(
    "LEGAL_BART_MODEL",
    _LOCAL_BART if os.path.exists(os.path.join(_LOCAL_BART, "model.safetensors"))
    else "fakechamp24/legallens-bart-base",
)
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
# One explicit configuration, applied on every generate() call, so the model's
# saved generation_config.json (max_length: 20, a leftover bart-base pretraining
# default) never governs. Per-section token budgets live in
# summarizer.GENERATION_PROFILES; these are the shared knobs.
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 448
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
GENERATION_LENGTH_PENALTY = 1.0
GENERATION_EARLY_STOPPING = True

# Citation (Cite & Verify). Support is judged by an NLI model on the top-k
# retrieved chunks (verbatim / compressed copies short-circuit to supported).
# Thresholds are entailment probabilities calibrated by
# scripts/calibrate_citations.py on 634 labelled sentences
# (artifacts/citation_calibration.json): 0.45 maximises F1 (precision 0.84,
# recall 0.77, AUC 0.891); 0.80 gives precision 0.90.
_LOCAL_NLI = os.path.join(os.path.dirname(__file__), "models", "nli-deberta-v3-xsmall")
# Prefer a local copy (models/nli-deberta-v3-xsmall) when present; otherwise the Hub id.
CITATION_NLI_MODEL = os.getenv(
    "LEGAL_NLI_MODEL",
    _LOCAL_NLI if os.path.exists(os.path.join(_LOCAL_NLI, "model.safetensors")) else "cross-encoder/nli-deberta-v3-xsmall",
)
CITATION_TOP_K = 2
CITATION_WINDOW_WORDS = 180
CITATION_THRESHOLD = 0.45
CITATION_HIGH_CONFIDENCE = 0.80
CITATION_CONTRADICTION_THRESHOLD = 0.50

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

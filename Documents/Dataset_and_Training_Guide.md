# LegalLens — Dataset Build, Alternative Data Sources, and BART Training

Working guide for taking LegalLens from "code ready" to "fine-tuned model".
Covers three things:

1. **Dataset modification** — building, validating and uploading the section-aligned dataset.
2. **Using a different data source** — Kaggle, IEEE DataPort, or any local corpus instead of Hugging Face.
3. **BART training** — what to build, what to check first, and how to evaluate it.

> Status note: everything below is written against the code as it stands at the time of
> writing. Where something is *not* implemented yet it is marked **[TO BUILD]**.

---

## 0. Status at a glance

| Item | State |
| --- | --- |
| `src/data/section_summary_alignment.py` | Committed (`92b07ca`, revised `55efe36`) — per-chunk scoring, one primary chunk per sentence |
| `scripts/build_dataset.py` | Committed (`92b07ca`, revised `55efe36`) |
| `scripts/validate_section_alignment.py` | Ported to the new alignment API — now a readable wrapper around `build_dataset.build_document` |
| `src/data/chunker.py` — oversized-paragraph fix | Committed (`b8c310c`) |
| Section classifier (`models/section_classifier.joblib`) | Merged from `review2` (`a76182d`) — MARRO + OpenNyAI InRhetoricalRoles, real appellant/respondent split |
| Tests | `test_chunker.py`, `test_alignment.py`, `test_section_detector.py` — 27 pass, 1 skipped |
| `training data/train.jsonl` | First sample build: 100 train documents → 1,287 records, 0 structure fallbacks. No `build_stats.json` committed with it |
| `sentence-transformers` | **Install before building** — without it the Cross-Encoder silently falls back to a stub |
| Hugging Face login | Needed only for upload |
| `src/model/train.py` | Written and smoke-tested locally (RTX 5070 8 GB, bf16) — §3.3 |
| `src/eval/evaluate.py` | Written and smoke-tested — document-level ROUGE vs IN-ABS, §3.4 |
| Summary sentence splitter | Abbreviation-aware + strips SCR page markers — val100 unassigned rate 28.6% → 12.9% (§1.3) |
| Colab notebook | Not built — local GPU is sufficient for `bart-base` |
| Full dataset built / uploaded | **Not yet** |

So the code side is done and committed; the work that remains is the validation and full build
stages, then training and evaluation. Keep committing as you go. `__pycache__/` is ignored and no
longer tracked, so `git status` stays free of bytecode churn.

**Retraining the section classifier** (`scripts/train_section_classifier.py`) needs the MARRO data
under `artifacts/marro/`, which is no longer in the repository. The committed model works as-is;
re-download MARRO only if you need to retrain.

---

## 1. How to proceed: dataset modification

### 1.1 Prerequisites

```bash
cd legal-doc-intelligence
python -m pip install -r requirements.txt
# leaner equivalent for the dataset step alone:
python -m pip install datasets sentence-transformers
```

- `datasets` → reading IN-ABS, Parquet output, Hub upload (pulls `pyarrow`, `pandas`).
- `sentence-transformers` → the real Cross-Encoder (pulls `torch`, `transformers`, `scikit-learn`).
- CPU is sufficient for dataset creation. First run downloads the cross-encoder (~90 MB)
  and the IN-ABS parquet files (~104 MB).

Hugging Face credentials (only needed for the upload step):

```bash
hf auth login          # or: export HF_TOKEN=hf_xxx
```

Token cached at `~/.cache/huggingface/token` (Windows: `C:\Users\<you>\.cache\huggingface\token`).
Needs **write** scope. In Colab use a Secret named `HF_TOKEN`. Nothing is stored in the repo.

### 1.2 Build stages

Run them in order and read the gates after each one. Do not jump to `--full`.

```bash
# Stage 1 — small enough to inspect by hand
python scripts/build_dataset.py --sample 10  --output_dir artifacts/training_data/sample10

# Stage 2 — timing + a real statistics check
python scripts/build_dataset.py --sample 100 --output_dir artifacts/training_data/val100
# exactly 100 validation documents:
python scripts/build_dataset.py --sample 100 --split validation --output_dir artifacts/training_data/val100

# Stage 3 — everything (hours on CPU; run in the background)
python scripts/build_dataset.py --full --output_dir artifacts/training_data/full
```

`--sample N` means **N complete source documents per selected split** (first N in the official
split order, so runs are reproducible). It never means N chunks, N sentences, or N records.
The number of records that results is whatever the documents yield.

### 1.3 Gates — do not continue until these pass

Check `build_stats.json` after every stage:

| Field | Expected | Why |
| --- | --- | --- |
| `alignment.backend` | `CrossEncoder` | `stub` means the Cross-Encoder did not load and the targets are meaningless (the builder refuses to build, or refuses to upload, in that case) |
| `split_isolation.verified` | `true` | No document, and no chunk of one, crosses a split boundary |
| `split_isolation.cross_split_content_duplicates` | report it | Leakage inherited from IN-ABS itself, not introduced by you |
| `summary_sentences.unassigned_rate` | ~0.10–0.20 | Outside this band means the threshold needs revisiting |
| `summary_sentences.multi_label_rate` | ~0.13–0.31 | Reference: `artifacts/alignment_validation.json` |
| `quality_flags.records_with_target_over_model_input` | 0 or near 0 | Targets that do not fit the model window are untrainable |
| `chunks.dropped_no_aligned_target` | sanity-check the share | Chunks with no alignment signal are excluded by design |

And open `inspect_sample.txt` (written for `--sample` runs). For each record confirm the
`aligned_target` is genuinely *about* the `chunk_text` above it. **A non-empty target is not
evidence by itself.**

**Measured on `val100` (100 validation documents, 2.5 min on GPU):** backend `CrossEncoder`,
split isolation verified, unassigned rate 12.9%, multi-label 23.5%, 0 length violations, 1
structure fallback, 258 of 871 chunks dropped for lack of an aligned target. The first build gave
28.6% unassigned; about half of those were fragments ("No.", "V.", "Cr.", "[315 G] T.") from a
splitter that broke inside citations, which `summary_sentences` now avoids.

Argument sections are thin: 23 of 613 records are `arguments_appellant`/`arguments_respondent`,
so the model sees few argument targets. Carry this into the report.

### 1.4 What a build produces

```
artifacts/training_data/<name>/
  train.jsonl / validation.jsonl / test.jsonl   # one record per retained chunk
  *.parquet                                     # same data, if pyarrow is installed
  reference_summaries.jsonl                     # doc_id, split, original_reference_summary
  unassigned.jsonl                              # sentences no section claimed
  build_stats.json                              # every statistic + config used
  inspect_sample.txt                            # side-by-side, --sample runs only
  README.md                                     # dataset card
```

Record schema:

| Field | Meaning |
| --- | --- |
| `doc_id` | `<split>_<position>`, e.g. `validation_00000` |
| `chunk_id`, `chunk_index` | `Chunk.chunk_id`, and its position in the document's chunk list |
| `section` | `facts` / `arguments_appellant` / `arguments_respondent` / `reasoning` / `judgement` / `full_document` |
| `chunk_text` | **model input** |
| `aligned_target` | **pseudo-target** (not ground truth) |
| `split`, `para_range` | official split; `[start, end]` paragraph indices |
| `matched_sentences`, `alignment_scores`, `matched_labels` | alignment provenance |
| `structure_fallback`, `n_chunk_tokens_est`, `n_target_tokens_est` | diagnostics (token counts are estimator values) |

Three objects must never be confused:

- `chunk_text` — source input.
- `aligned_target` — Cross-Encoder-derived pseudo-target, for chunk-level training only.
- `original_reference_summary` — the IN-ABS summary as published, the **document-level
  evaluation reference**. Kept once per document in `reference_summaries.jsonl`.

### 1.5 Uploading

```bash
python scripts/build_dataset.py --push-to-hub \
    --repo-id <HF_USERNAME>/in-abs-section-aligned \
    --output_dir artifacts/training_data/full
```

Upload only runs when `--push-to-hub` is passed — never during a normal build. It refuses
stub-encoder builds, refuses a directory without `build_stats.json`, and reads the dataset
back afterwards to confirm what the Hub is serving.

### 1.6 Licensing — read before uploading

`percins/IN-ABS` **declares no license**, and its dataset card contains nothing but
"More Information needed". Its provenance (Indian court judgments) is public-domain-adjacent,
but the *curation* may not be freely redistributable.

Before publishing a derived dataset:

1. Try to contact the IN-ABS authors and ask for terms.
2. State provenance explicitly in the card ("derived from `percins/IN-ABS`") and say the
   licence is **undetermined** rather than claiming one you cannot grant.
3. Keep the Hub repo **private** until this is resolved.
4. Remember a model fine-tuned on this data inherits the same uncertainty.

The generated card already states provenance and points at the source dataset; do not
overwrite that section with an assumed licence.

---

## 2. Using a different data source (Kaggle, IEEE DataPort, local files)

### 2.1 The single hard requirement

The whole alignment methodology works by **decomposing a document-level reference summary**
into sentences and matching each sentence to the chunk it came from. So any source must give you,
per document:

- the full judgement text, and
- **a whole-document summary of that judgement.**

If a corpus has no summaries (raw judgment collections, contract clause corpora, case-metadata
tables), it **cannot** produce `aligned_target` pseudo-targets. Such a corpus is still usable for
*continued BART pretraining / domain adaptation*, but not for this supervised chunk-level pipeline.
Check this before planning around a source.

### 2.2 Recommended architecture: make the source pluggable, not rewritten

Hugging Face, Kaggle and IEEE DataPort all end at the same place — a file on disk. Rather than
adding portal-specific code, add **one local-file source** and normalise whatever you downloaded
into it. Nothing downstream of loading changes: the same cleaning, section detection, chunking,
alignment, statistics, split-isolation checks and upload path all still apply.

That is a small change to `scripts/build_dataset.py`, in the loading section only (currently
lines ~86–190). **`[TO BUILD]`** — it does not exist yet. Design:

```python
# 1. Source selection
SOURCES = ("hub", "local")

# 2. A column-name helper that works for a HF Dataset *or* a plain list of dicts
def _column_names(dataset) -> list[str]:
    if hasattr(dataset, "column_names"):          # huggingface Dataset
        return list(dataset.column_names)
    if dataset:                                    # list[dict] from json/jsonl/csv
        return list(dataset[0].keys())
    return []

# 3. resolve_columns() then uses _column_names(dataset) instead of dataset.column_names

# 4. A local loader, dispatched from load_split()
def load_local(source_dir: Path, split: str):
    for name in (f"{split}.jsonl", f"{split}.json", f"{split}.csv", f"{split}.parquet"):
        path = source_dir / name
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".csv":
            import csv
            with path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        datasets = _import_datasets()
        return datasets.load_dataset("parquet", data_files=str(path), split="train")
    raise SystemExit(
        f"No file for split '{split}' in {source_dir} "
        f"(looked for {split}.jsonl / .json / .csv / .parquet)"
    )

# 5. load_split() becomes a dispatch
def load_split(split, limit=None):
    if args.source == "local":
        rows = load_local(Path(args.input_dir), split)
        return rows[:limit] if limit else rows
    ...  # existing hub path, unchanged
```

New CLI flags: `--source {hub,local}` (default `hub`), `--input-dir`, and explicit
`--text-column` / `--summary-column` overrides for sources whose column names are unusual
(`resolve_columns` already tolerates common aliases — see `TEXT_COLUMN_CANDIDATES` and
`SUMMARY_COLUMN_CANDIDATES`).

Once this exists, switching source is a data-preparation exercise, not a code change:

```
data/<source>/
  train.jsonl        # {"text": "<judgement>", "summary": "<reference summary>", ...}
  validation.jsonl
  test.jsonl

python scripts/build_dataset.py --source local --input-dir data/<source> \
    --full --output_dir artifacts/training_data/<source>
```

Everything in §1.3 still applies — including split isolation and the quality gates.

### 2.3 Kaggle

- **Credentials**: Kaggle → *Account* → *Create New API Token* → `kaggle.json`, placed at
  `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY`; in Colab use Secrets).
- **Two download paths**: the `kagglehub` SDK (`kagglehub.dataset_download("<owner>/<slug>")`) or the
  `kaggle` CLI (`kaggle datasets download -d <owner>/<slug> -p data/raw --unzip`). Check the current
  flags in Kaggle's own docs — this guide deliberately does not pin their exact syntax.
- **Then**: convert to `text` + `summary` JSONL and use the local source above.
- **Watch out**: most legal datasets on Kaggle are *contract clause* corpora, token-classification
  sets, or court metadata — **not** judgement→summary pairs. Verify a real summary column exists
  before building anything around it. Also, Kaggle licences vary per dataset; many are scraped
  court PDFs with no declared licence, so check §1.6 before republishing derivatives.

### 2.4 IEEE DataPort

- Requires an IEEE account, and many datasets require accepting terms or requesting access, so
  expect a **manual download** (usually a zip). There is no general-purpose API for arbitrary
  datasets — treat the result as a local folder and use §2.2.
- Each dataset page states its own **licence** and a required **citation**. Record both in your
  dataset card (DOI, title, authors) — IEEE's terms generally expect the citation to travel with
  the data.
- IEEE DataPort is heavily weighted toward engineering/sensor benchmarks; check that a legal-text
  summarization corpus with reference summaries actually exists before planning around it.

### 2.5 Other sources worth considering

Because this pipeline is tuned for **Indian** judgments, Indian sources transfer best:

- **Indian Kanoon** — public judgments, documented API with a free key.
- **Official court portals** — Supreme Court of India, High Court sites, eCourts services (judgment PDFs).
- Any corpus of paired judgement + headnote/summary.

Whatever you choose, two questions decide feasibility: *is there a reference summary per document?*
and *is the jurisdiction/language English?* (§2.7).

### 2.6 Splits when the source has none

IN-ABS provides official `train` / `validation` / `test` splits and the builder preserves them as-is.
A local corpus usually will not. Do **not** split chunks — split **documents**, before chunking,
and document exactly how. A deterministic, order-independent scheme:

```python
def assign_split(doc_id: str, train: float = 0.75, validation: float = 0.10) -> str:
    """Stable hash of the document id -> split. Same doc always lands in the same split."""
    bucket = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"
```

Hashing the id means the split is reproducible without shuffling state, and it is stable when you
add or remove documents. Record the fractions and the scheme in `build_stats.json` and in the
dataset card. The builder's `verify_split_isolation` will then still prove
`train ∩ validation ∩ test = ∅` and report any duplicate document text across splits.

### 2.7 Domain-transfer caveats — the part that actually bites

Swapping the dataset is not just plumbing. These modules encode **Indian/Commonwealth** legal
conventions, and they degrade quietly on other corpora:

| Component | Hardcoded assumption | Consequence on other jurisdictions |
| --- | --- | --- |
| `src/data/section_detector.py` | "appellant", "petitioner", "learned counsel", "High Court", "Section N of the X Act", "Article N of the Constitution" | Detection collapses → `fallback_to_single_section` fires → everything becomes `full_document` and the section-aware structure — the whole point of the product — is lost |
| `config.PII_PATTERNS` | Aadhaar, PAN, `+91` phone numbers | India-specific identifiers in another jurisdiction go unredacted (add SSN/national-ID patterns for that corpus) |
| `src/postprocess/jargon.py` | Indian/Commonwealth legal English ("appellant", "impugned order", "in the result") | Citizen Mode stops simplifying the terms your corpus actually uses |
| Language | English only (`en_core_web_sm`, `bart-base`) | Non-English judgments need a different tokenizer, model and spaCy pipeline |
| Length profile | IN-Abs averages ~4.7k tokens/document | Much longer documents hit the 15-chunk cap and lose their tails (§4) |

Practical implication: a model fine-tuned on corpus A will **not** transfer cleanly to corpus B —
the `aligned_target` values encode A's section taxonomy and phrasing. Changing source usually means
rebuilding the dataset *and* retraining, plus extending the section patterns and jargon dictionary
for the new jurisdiction.

### 2.8 Rule of thumb

Reusing the pipeline with a new source is reasonable when the new corpus is
**English + Indian/Commonwealth + has reference summaries**. Outside that, you are changing the
product's assumptions, and §2.7 becomes a workstream of its own rather than a config change.

---

## 3. BART training

### 3.1 What exists, what does not

| Piece | State |
| --- | --- |
| Chunk-level training data (`chunk_text` → `aligned_target`) | ✅ produced by the builder |
| Document-level references for evaluation (`reference_summaries.jsonl`) | ✅ produced by the builder |
| Zero-shot baseline for comparison (`artifacts/baselines/zero_shot.json`) | ✅ exists |
| `src/model/train.py` | ✅ |
| Colab notebook | ❌ not needed if a local GPU is available |
| Document-level evaluation script (`src/eval/evaluate.py`) | ✅ |

Training is **not** CPU-bound work you want to do locally; use Colab (T4) for the fine-tune.
Dataset *creation* stays local/CPU.

### 3.2 Pre-flight — verify real token lengths first

`approximate_tokens` is `words × 1.33`, an estimate. The model truncates at the real tokenizer
count (`config.MAX_INPUT_LENGTH = 1024`). Measure before training rather than discovering
truncation later:

```python
from transformers import AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
ds = load_dataset("<HF_USERNAME>/in-abs-section-aligned")     # or Dataset.from_json(...)

for split in ds:
    lengths = [len(tok(x["chunk_text"])["input_ids"]) for x in ds[split]]
    over = sum(1 for n in lengths if n > 1024)
    print(split, "n=", len(lengths), "over 1024:", over, "max:", max(lengths))

tgt = [len(tok(x["aligned_target"])["input_ids"]) for x in ds["train"]]
print("target tokens  mean/p90/max:", sum(tgt)/len(tgt), sorted(tgt)[int(.9*len(tgt))], max(tgt))
```

If a material share of inputs exceed 1024, lower `config.CHUNK_SIZE` and **rebuild** the dataset
rather than letting the tokenizer silently truncate. Use the target distribution to pick
`max_target_length` (512 is a safe default; the pre-fix build measured mean ~194 / max ~491).

### 3.3 Training recipe

Implemented in `src/model/train.py`:

```bash
cd legal-doc-intelligence
python -m src.model.train --data artifacts/training_data/full --output_dir models/legallens-bart-base
# quick plumbing check first:
python -m src.model.train --data artifacts/training_data/full --output_dir models/bart-smoke     --max_train_samples 64 --max_eval_samples 16 --epochs 1
```

It reads a build directory (or a Hub dataset id), logs how many inputs/targets the length limits
truncate, saves the best checkpoint at the top of `--output_dir` and writes `training_meta.json`.
Differences from the plan below, for an 8 GB laptop GPU: per-device batch 4 × accumulation 8
(effective 32), bf16 when supported, gradient checkpointing on, `max_target_length` =
`config.MAX_TARGET_LENGTH` (448, what the builder enforces), and validation capped at 500 chunks
(`--max_eval_samples`) because beam-search generation over every chunk each epoch is slow.

- **Base model**: `facebook/bart-base` (i.e. `config.BART_MODEL`; `LEGAL_BART_MODEL` env var overrides).
- **Task**: chunk-level abstractive summarization, `chunk_text` → `aligned_target`.
- **Trainer**: `Seq2SeqTrainer` / `Seq2SeqTrainingArguments`.
- **Where**: Colab T4 (16 GB). Install `transformers`, `datasets`, `accelerate`, `rouge_score`.

Starting hyperparameters (adjust after watching the first epoch):

| Setting | Value |
| --- | --- |
| `learning_rate` | 3e-5 |
| `per_device_train_batch_size` | 8 |
| `gradient_accumulation_steps` | 4 (effective batch 32) |
| `num_train_epochs` | 3 |
| `warmup_ratio` | 0.06 |
| `weight_decay` | 0.01 |
| `label_smoothing_factor` | 0.1 |
| `fp16` | True (T4) |
| `max_input_length` | 1024 |
| `max_target_length` | 512 |
| `predict_with_generate` | True |
| `generation_num_beams` | 4 (matches inference `config.NUM_BEAMS`) |
| `eval_strategy` / `save_strategy` | epoch |
| `load_best_model_at_end`, `metric_for_best_model` | `rougeL` |
| `early_stopping_patience` | 2 |

Important: validation ROUGE during training is **chunk-level**, which is a *proxy*. The number you
report is document-level (§3.4). Also note ~30% of documents are `full_document` (structure the
detector could not resolve), so roughly a third of the training signal carries no section structure.

Time the `val100` build first and extrapolate instead of trusting any estimate.

### 3.4 Evaluation — document level

Implemented in `src/eval/evaluate.py`:

```bash
python -m src.eval.evaluate --models facebook/bart-base models/legallens-bart-base     --sample 100 --output artifacts/eval/test100.json      # add --bertscore if bert-score is installed
```

It runs `src.pipeline.process_document` with each model on the same test documents and scores two
outputs against the published summary: `lawyer` (Lawyer Mode body, 700-word budget) and `full`
(all chunk summaries, de-duplicated). The zero-shot baseline is simply `facebook/bart-base` in the
same run — the committed `zero_shot.json` covers only 3 documents. Zero-shot `bart-base` already
scores high (2-doc smoke test: lawyer ROUGE-L ≈ 42, full ≈ 54) because it mostly copies its input
and IN-ABS summaries are largely extractive, so the fine-tune has a real bar to clear.

The plan it implements:

1. Load the IN-ABS **test** split (`text`, `summary`).
2. Run the frozen pipeline: preprocessing → section detection → chunking (same modules the builder uses).
3. Generate one summary per chunk with the fine-tuned model (batch, `num_beams=4`).
4. Aggregate per section (`postprocess/aggregation.py`) and format
   (`postprocess/mode_formatter.py`).
5. Score the resulting document summary against `reference_summaries.jsonl` with ROUGE-1/2/L and
   BERTScore.
6. Compare against the zero-shot baseline in `artifacts/baselines/zero_shot.json` — a fine-tune
   that does not beat zero-shot on the test split has not earned its keep.

Report chunk-level and document-level numbers separately, and state the unassigned rate of the
alignment pass so readers know how much of each reference summary was actually used as a target.

### 3.4.1 Results — 1 epoch (`models/legallens-bart-base-e1`)

Full build (`artifacts/training_data/full`): 5,346 / 712 / 1,070 documents → 31,029 / 4,198 / 6,133
records; unassigned 17.9%, multi-label 21.5%, no length violations, 71 structure fallbacks (1%),
10 cross-split duplicate documents inherited from IN-ABS. The build took ~2 h on an RTX 5070 laptop GPU.

Training: 1 epoch, 970 steps, 27 min (bf16, batch 4 × accumulation 8). Chunk-level validation
(500 chunks vs pseudo-targets): ROUGE-1/2/L 37.3 / 20.3 / 33.0, eval loss 3.09. The logged *train*
loss (~25) is the sum over the 8 accumulation steps in this `transformers` version - divide by 8.

Document-level, first 100 IN-ABS **test** documents, same documents for both models
(`artifacts/eval/test100_e1.json`):

| Model | Output | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Mean words |
| --- | --- | --- | --- | --- | --- | --- |
| `facebook/bart-base` (zero-shot) | lawyer | 50.93 | 24.96 | 44.65 | 83.51 | 592 |
| fine-tuned, 1 epoch | lawyer | **53.86** | **29.48** | **48.03** | **84.92** | 543 |
| `facebook/bart-base` (zero-shot) | full | 52.86 | 30.05 | 45.11 | 83.67 | 1,361 |
| fine-tuned, 1 epoch | full | **57.53** | **32.54** | **49.68** | **85.00** | 730 |

Reference summaries average 827 words. The fine-tune wins ROUGE-L on 69/100 documents (lawyer) and
70/100 (full), and generation is ~2.5x faster (864 s vs 2,143 s for 100 documents) because it
writes summaries rather than copying its input.

### 3.5 Wiring the fine-tuned model back into inference

```bash
export LEGAL_BART_MODEL=<HF_USERNAME>/legalLens-bart-base   # or a local checkpoint path
python cli.py summarize artifacts/sample_judgment.txt --mode both
```

`config.BART_MODEL` reads that env var, so no code change is needed to *run* the fine-tuned model.

Two bugs that used to misreport it — **both fixed** (the summarizer also runs on GPU when one is available):

1. `Summarizer.is_finetuned` is `False` whenever `model_name is None` — which is always, since the
   API and CLI never pass one — so the API reports `is_finetuned: false` even when
   `LEGAL_BART_MODEL` points at a fine-tuned checkpoint.
2. `get_summarizer(model_name)` caches a global singleton and ignores `model_name` after the first
   call, so two models cannot be compared inside one process.

---

## 4. Known limitations and open decisions

Carry these into the report; all are measured, not speculative.

1. **Chunk cap drops document tails.** `MAX_CHUNKS_PER_DOC = 15` retains the *earliest* chunks per
   section. On the 12.8k-word sample judgment: 27 bounded chunks are produced, 15 are kept, and the
   capped output covers 60% of the words. Options: leave it, stride-sample across the section, or
   raise the cap.
2. **Structure fallback** was ~30% of documents with the earlier regex detector → `full_document`
   labels, i.e. no section signal for a third of the data. With the trained classifier the 100-document
   sample in `training data/train.jsonl` shows 0 fallbacks; confirm on the full build.
3. **Alignment is a proxy.** The unassigned rate has measured 13–19%, meaning that share of each
   reference summary is never used as a target; multi-label sentences are deliberately attached to
   more than one chunk.
4. **The token estimator is word-based.** A paragraph containing no whitespace at all is
   under-estimated and would still arrive as one chunk. Low risk on real judgments; the clean fix
   is a character-length guard.
5. **`Cite & Verify` does not currently discriminate.** Measured verified rate is ~98.5% at every
   threshold from 0.30 to 0.65 (`artifacts/citation_distribution.json`), so it cannot support a
   claim that citations are checked. This does not affect training, but do not present it as a
   working hallucination guard until it is recalibrated.
6. **Some targets may exceed `MAX_TARGET_LENGTH` (448).** In the sample `train.jsonl`, 73 of 1,287
   records have an *estimated* target length above 448 (max ~1.7k). The builder measures real
   tokenizer lengths and reports violations in `build_stats.json` → `length_validation`; check that
   before training rather than relying on the estimate.

---

## 5. Command reference

```bash
# --- setup ---------------------------------------------------------------
cd legal-doc-intelligence
python -m pip install -r requirements.txt
hf auth login

# --- dataset -------------------------------------------------------------
python scripts/build_dataset.py --sample 10  --output_dir artifacts/training_data/sample10
python scripts/build_dataset.py --sample 100 --split validation --output_dir artifacts/training_data/val100
python scripts/build_dataset.py --full --output_dir artifacts/training_data/full

# --- upload (opt-in only) ------------------------------------------------
python scripts/build_dataset.py --push-to-hub \
    --repo-id <HF_USERNAME>/in-abs-section-aligned \
    --output_dir artifacts/training_data/full

# --- tests ---------------------------------------------------------------
python -m pytest tests -q

# --- alignment spot-check (same code path as the builder) ----------------
python scripts/validate_section_alignment.py --sample-size 15 \
    --output artifacts/alignment_validation/validation_15docs.json
python scripts/validate_section_alignment.py --demo

# --- run inference with a fine-tuned checkpoint --------------------------
export LEGAL_BART_MODEL=<HF_USERNAME>/legalLens-bart-base
python cli.py summarize artifacts/sample_judgment.txt --mode both --output result.json
python cli.py inspect artifacts/sample_judgment.txt
```

Relevant configuration (`config.py`): `CHUNK_SIZE=850`, `CHUNK_OVERLAP=100`,
`MAX_CHUNKS_PER_DOC=15`, `MAX_INPUT_LENGTH=1024`, `NUM_BEAMS=4`, `ALIGNMENT_THRESHOLD=0.40`,
`ALIGNMENT_RELATIVE_MARGIN=0.90`, `CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`,
`CITATION_THRESHOLD=0.30`, `BART_MODEL=facebook/bart-base` (env `LEGAL_BART_MODEL`).

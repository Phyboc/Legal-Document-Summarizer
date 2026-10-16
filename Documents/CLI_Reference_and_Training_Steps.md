# LegalLens — CLI Reference and Training-Production Steps

Single reference for **every command** needed to take LegalLens from a clean checkout to
(a) generated training datasets, (b) those datasets uploaded to Hugging Face, and
(c) the models trained on them.

Paths in this document assume the repository root:

```
D:\Studies\projects\LegalDocumentSummarizer\Legal-Document-Summarizer
```

All dataset commands run from inside the `legal-doc-intelligence/` package directory.
Everything below was verified against the code as it currently stands; anything that is
**not yet implemented** is marked **[TO BUILD]**.

---

## 0. Pipeline at a glance

```
IN-ABS (percins/IN-ABS, Hugging Face)
        │
        ▼
build_dataset.py ── preprocess → PII redact → split paragraphs → detect sections
        │            → chunk → align reference summary to chunks (Cross-Encoder)
        │            → write chunk-level records + stats + dataset card
        ▼
artifacts/training_data/<name>/   (train/validation/test .jsonl + .parquet)
        │
        ├──► --push-to-hub ──► Hugging Face dataset repo
        │
        └──► (future) BART fine-tuning ──► fine-tuned summariser ──► document-level eval
```

Three objects that must never be confused:

| Object | Meaning |
| --- | --- |
| `chunk_text` | model **input** |
| `aligned_target` | Cross-Encoder-derived **pseudo-target** (not ground truth) |
| `original_reference_summary` | the published IN-ABS summary — **document-level evaluation reference** |

---

## 1. One-time environment setup

### 1.1 Activate the project virtual environment

The project venv lives at `D:\Studies\projects\venv`. **PowerShell:**

```powershell
cd D:\Studies\projects\LegalDocumentSummarizer\Legal-Document-Summarizer\legal-doc-intelligence
& "D:\Studies\projects\venv\Scripts\Activate.ps1"

# confirm the right interpreter is active
python -c "import sys; print(sys.executable)"
# expected: D:\Studies\projects\venv\Scripts\python.exe
```

**Git Bash:**

```bash
cd /d/Studies/projects/LegalDocumentSummarizer/Legal-Document-Summarizer/legal-doc-intelligence
source /d/Studies/projects/venv/Scripts/activate
```

If activation does not take effect (Windows can fall through to the Microsoft Store
Python), call the interpreter by full path instead — no activation required:

```powershell
D:\Studies\projects\venv\Scripts\python.exe scripts\build_dataset.py --help
```

### 1.2 Install dependencies

```bash
# CPU-only torch first (avoids the multi-GB CUDA wheel on Windows)
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

# everything the dataset build needs
python -m pip install "transformers>=4.30.0" "sentence-transformers>=2.2.0" \
    "datasets>=2.14.0" "huggingface_hub" "spacy>=3.5.0" "accelerate"

# or the whole declared set (adds API/UI/extra-eval packages)
python -m pip install -r requirements.txt
```

What each is for: `datasets` = read IN-ABS + write parquet + Hub push; `sentence-transformers`
= the real Cross-Encoder (uploads are blocked if it is missing); `transformers` = real BART
tokenizer for length validation; `spacy` = only needed for `--redact-names`.

Versions confirmed working on this machine: torch 2.14.1+cpu, transformers 5.20.1,
sentence-transformers 6.1.0, datasets 5.1.0, huggingface_hub 1.33.0, spacy 3.8.16.

### 1.3 Hugging Face login (only needed to upload)

```bash
hf auth login                 # paste an hf_... token with WRITE scope
# token cached at C:\Users\HP\.cache\huggingface\token
# older installs: huggingface-cli login
```

Or, per shell / CI / Colab:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx      # Git Bash
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"      # PowerShell
```

Verify: `hf auth whoami` → prints your username (currently **`Phyboc`**).

Setup on huggingface.co (once): create an account, then **Settings → Access Tokens →
New token → type Write**. `push_to_hub` **auto-creates** each dataset repo, so you do not
pre-create them. Use `--private` (recommended — `percins/IN-ABS` declares no licence).

### 1.4 Optional: quiet the Windows symlink warning

```bash
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

---

## 2. Producing the datasets — canonical builder

`scripts/build_dataset.py` is the **only** dataset generator. Run one invocation per dataset.

### 2.1 The three stages (run in order, read the gates after each)

```bash
# Stage 1 — small enough to inspect by hand
python scripts/build_dataset.py --sample 10  --output_dir artifacts/training_data/sample10

# Stage 2 — timing + real statistics (100 validation documents only)
python scripts/build_dataset.py --sample 100 --split validation --output_dir artifacts/training_data/val100

# Stage 3 — everything (hours on CPU; run in the background)
python scripts/build_dataset.py --full --output_dir artifacts/training_data/full
```

`--sample N` means **N complete source documents per selected split** (the first N in the
official split order, so runs are reproducible). It never means N chunks, N sentences or
N records.

### 2.2 Full flag list

| Flag | Meaning |
| --- | --- |
| `--sample N` | Process the first N complete documents of each selected split |
| `--full` | Process every document in the selected splits |
| `--split {train,validation,test}` | Restrict to a split (repeatable). Default: all three |
| `--output_dir PATH` | **Required.** Where to write the dataset |
| `--overwrite` | Allow writing into an output dir that already holds a build |
| `--redact-names` | Also redact PERSON entities (spaCy) |
| `--no-strip-preamble` | Keep the case-caption preamble |
| `--threshold FLOAT` | Override the alignment threshold (default 0.40) |
| `--relative-margin FLOAT` | Override the relative margin (default 0.90) |
| `--no-multi-label` | Force a single label per summary sentence |
| `--evidence-mode` | Enable the evidence-aware alignment layer (**UNCALIBRATED**) |
| `--evidence-coverage-bound FLOAT` | Tier-3 coverage bound C |
| `--evidence-ambiguity-margin FLOAT` | Ambiguity margin M |
| `--evidence-min-span INT` | Minimum longest common content-word span |
| `--evidence-max-rare-df INT` | Max document-local chunk frequency still "rare" |
| `--evidence-contradiction-mode {high_confidence,off}` | Contradiction policy |
| `--allow-stub-encoder` | Allow the meaningless mock encoder (plumbing only; blocks upload) |
| `--push-to-hub` | Upload to the Hub (see §4) |
| `--append` | With `--push-to-hub`: merge this build INTO the existing Hub dataset instead of replacing it. Upserts by `doc_id` + `chunk_id`; remote-only rows/splits are kept. Use to accumulate runs into one repo. |
| `--repo-id USER/NAME` | Target Hub repo (required with `--push-to-hub`) |
| `--private` | Create the Hub repo as private |

### 2.3 Worked examples for each variant

```bash
python scripts/build_dataset.py --full --evidence-mode \
    --output_dir artifacts/training_data/evidence
python scripts/build_dataset.py --full --no-multi-label \
    --output_dir artifacts/training_data/singlelabel
python scripts/build_dataset.py --full --threshold 0.30 \
    --output_dir artifacts/training_data/thr030
python scripts/build_dataset.py --full --threshold 0.55 \
    --output_dir artifacts/training_data/thr055
python scripts/build_dataset.py --full --redact-names \
    --output_dir artifacts/training_data/redacted
```

### 2.4 What a build produces

```
artifacts/training_data/<name>/
  train.jsonl / validation.jsonl / test.jsonl   # one record per retained chunk
  *.parquet                                     # same data (pyarrow present)
  reference_summaries.jsonl                     # doc_id, split, original_reference_summary
  unassigned.jsonl                              # sentences no section claimed
  ambiguous.jsonl                               # evidence-mode ambiguous sentences
  build_stats.json                              # every statistic + config used
  build_manifest.json                           # identity + sha256 of each output
  inspect_sample.txt                            # side-by-side (--sample runs only)
  README.md                                     # dataset card
```

Record fields: `doc_id, chunk_id, section, chunk_text, aligned_target, split, para_range,
matched_sentences, alignment_scores, matched_labels, structure_fallback, overflow_sentences,
n_overflow_sentences, chunk_index, n_chunk_tokens_est, n_target_tokens_est`.

### 2.5 Quality gates — check before continuing

Open `build_stats.json` (and `inspect_sample.txt` for `--sample` runs):

| Field | Expected |
| --- | --- |
| `alignment.backend` | `CrossEncoder` (`stub` = targets are meaningless) |
| `split_isolation.verified` | `true` |
| `split_isolation.cross_split_content_duplicates` | report it (inherited from IN-ABS) |
| `summary_sentences.unassigned_rate` | ~0.10–0.20 |
| `summary_sentences.multi_label_rate` | ~0.13–0.31 |
| `quality_flags.records_with_target_over_model_input` | 0 or near 0 |
| `length_validation.ok` | `true` (build fails otherwise) |

For each record in `inspect_sample.txt`, confirm `aligned_target` is genuinely *about* the
`chunk_text` above it. A non-empty target is not evidence by itself.

---

## 3. Producing ten datasets in one run — batch driver

`scripts/build_and_push_datasets.py` orchestrates the builder over ten configurations and
uploads each to its own repo. It reuses the builder's own `parse_args → run_build → report →
push_to_hub`, so all gates still apply.

### 3.1 The ten datasets

Each varies one axis so results are comparable.

| Dataset | Builder flags | Docs (× splits) |
| --- | --- | --- |
| `sample10` | `--sample 10` | 30 |
| `sample100` | `--sample 100` | 300 |
| `full` | `--full` | 7,128 |
| `val100` | `--sample 100 --split validation` | 100 |
| `test-only` | `--full --split test` | 1,070 |
| `evidence` | `--full --evidence-mode` | 7,128 |
| `singlelabel` | `--full --no-multi-label` | 7,128 |
| `thr030` | `--full --threshold 0.30` | 7,128 |
| `thr055` | `--full --threshold 0.55` | 7,128 |
| `redacted` | `--full --redact-names` | 7,128 |

Each targets repo `<namespace>/legalLens-in-abs-<name>`.

### 3.2 Driver flags

| Flag | Meaning |
| --- | --- |
| `--namespace USER` | Hub user/org that owns the datasets (required to push) |
| `--repo-prefix NAME` | Repo prefix (default `legalLens-in-abs`) |
| `--output-root PATH` | Holds one sub-directory per dataset (default `artifacts/training_data`) |
| `--only NAME` | Build only this dataset (repeatable) |
| `--fast` | Replace every `--full` with `--sample <fast-n>` |
| `--fast-n N` | Documents per split used by `--fast` (default 50) |
| `--private` | Create Hub repos as private |
| `--append` | Merge each build INTO the existing Hub dataset (upsert, see §2.2) |
| `--overwrite` | Allow writing into dirs that already hold a build |
| `--skip-push` | Build only; never upload |
| `--dry-run` | Print the builder commands without running them |
| `--summary PATH` | JSON run summary (default `<output-root>/build_and_push_summary.json`) |

### 3.3 Commands

```powershell
# see exactly what will be built/pushed, touching nothing
python scripts\build_and_push_datasets.py --namespace Phyboc --dry-run

# validate one light dataset first (30 docs, build + upload)
python scripts\build_and_push_datasets.py --namespace Phyboc --only sample10 --private

# a few datasets
python scripts\build_and_push_datasets.py --namespace Phyboc --only full --only evidence --private

# everything, in the background (long; see §8)
python scripts\build_and_push_datasets.py --namespace Phyboc --private > build.log 2>&1

# build only, no upload
python scripts\build_and_push_datasets.py --namespace Phyboc --skip-push
```

> `--fast` only swaps `--full`; it does **not** shrink the pre-set `sample100`/`val100`.
> Use `--only` to pick light datasets for a fast validation pass.

---

## 4. Uploading datasets to Hugging Face

Upload is **opt-in only** — it never happens during a normal build.

```bash
# build + upload in one command
python scripts/build_dataset.py --full \
    --output_dir artifacts/training_data/full \
    --push-to-hub --repo-id Phyboc/legalLens-in-abs-full --private

# push an already-built directory (no rebuild)
python scripts/build_dataset.py --push-to-hub \
    --repo-id Phyboc/legalLens-in-abs-full \
    --output_dir artifacts/training_data/full --private
```

Behaviour and guards:

- Refuses **stub-encoder** builds (meaningless targets).
- Refuses a directory without `build_stats.json`.
- Fails early if not authenticated, with login instructions.
- Uploads splits + the dataset card (`README.md`), then **reads the dataset back** from the
  Hub to confirm what is served.
- `--private` only applies when the repo is first created; it cannot change an existing repo.

**Licensing:** `percins/IN-ABS` declares no licence. Keep derived repos **private** until the
terms are resolved, and state provenance in the card.

### 4.1 Appending instead of replacing

HF datasets are snapshot-replace on every push; there is no native `append_rows`. `--append`
implements upsert semantics in that gap: it loads the existing repo, keeps all its rows, drops
from them only the `(doc_id, chunk_id)` pairs the fresh build produces, concatenates the rest,
then pushes the combined DatasetDict. Rows already on the Hub are updated to the new values;
anything else on the Hub is kept. If the repo does not exist yet it behaves like a normal push.

```bash
# accumulate every run into ONE repo (one per configuration)
python scripts/build_and_push_datasets.py --namespace Phyboc --append --private

# or append a single fresh build into an existing repo
python scripts/build_dataset.py --sample 10 --output_dir artifacts/training_data/sample10 \
    --push-to-hub --repo-id Phyboc/legalLens-in-abs-sample10 --append --private
```

Credentials/provenance caveats: the local `artifacts/training_data/<name>/` output is still
per-run (not accumulated) — the accumulated dataset lives only on the Hub. The dataset card is
overwritten each push, so it reflects the most recent run's settings.

---

## 5. Validation and audit commands

```bash
# section-summary alignment on real documents
python scripts/validate_section_alignment.py --sample-size 15
python scripts/validate_section_alignment.py --demo          # no network / no dataset
python scripts/validate_section_alignment.py --sample-size 15 --quiet

# evidence-layer calibration audit (offline; no model download)
python scripts/audit_alignment_evidence.py \
    --from-jsonl "training data/train.jsonl" \
    --report artifacts/evidence_audit.json
python scripts/audit_alignment_evidence.py \
    --from-candidates artifacts/evidence_candidates.jsonl \
    --out artifacts/evidence_sweep.json --report artifacts/evidence_audit.json
```

`audit_alignment_evidence.py` options: `--from-jsonl`, `--from-candidates`, `--out`,
`--report`, `--min-positives`, `--min-negatives`, `--positive-span`, `--positive-classes`,
`--limit`. Exit code `2` = a recommendation was requested but the corpus was below the
minimum size (the report is still written).

### Tests

```bash
python -m pytest tests/ -q
# runnable test modules present: test_alignment, test_chunker,
# test_evidence_alignment, test_section_detector
```

---

## 6. Training step 1 — sentence-level section classifier

`scripts/train_section_classifier.py` trains a TF-IDF + Logistic-Regression classifier on
**MARRO** + **OpenNyAI InRhetoricalRoles** and writes `models/section_classifier.joblib`.

```bash
python scripts/train_section_classifier.py
```

Output: `models/section_classifier.joblib` (consumed by `src/data/section_detector.py`).

**Prerequisites — not present in the repo as committed:**

- `artifacts/marro/IN-train-set/**/*.txt` and `IN-test-set/**/*.txt` (MARRO data).
- `review/section_labels.a.csv` and `review/section_labels.claude.csv` (the gold review sheets).
- `opennyaiorg/InRhetoricalRoles` — **gated**; accept its terms and log in first, or the
  OpenNyAI half of the training signal is unavailable (which removes the only real
  appellant-vs-respondent split).

Without these, the script cannot run as-is. Supply the data or skip this step.

---

## 7. Training step 2 — BART fine-tuning  —  **[TO BUILD]**

There is **no** `src/model/train.py` and no Colab notebook in the repo yet. The intended
recipe (dataset *creation* is local/CPU; training belongs on a GPU, e.g. Colab T4):

- **Base model:** `facebook/bart-base` (`config.BART_MODEL`; `LEGAL_BART_MODEL` overrides).
- **Task:** chunk-level abstractive summarisation, `chunk_text` → `aligned_target`.
- **Trainer:** `Seq2SeqTrainer` / `Seq2SeqTrainingArguments`.

Pre-flight — measure real token lengths before training (the builder already validates them,
but re-check on the uploaded dataset):

```python
from transformers import AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
ds = load_dataset("Phyboc/legalLens-in-abs-full")

for split in ds:
    lengths = [len(tok(x["chunk_text"])["input_ids"]) for x in ds[split]]
    print(split, "n=", len(lengths), "over 1024:", sum(n > 1024 for n in lengths), "max:", max(lengths))
```

Starting hyperparameters:

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
| `max_input_length` / `max_target_length` | 1024 / 512 |
| `predict_with_generate` | True |
| `generation_num_beams` | 4 (matches `config.NUM_BEAMS`) |
| `eval_strategy` / `save_strategy` | epoch |
| `load_best_model_at_end`, `metric_for_best_model` | `rougeL` |
| `early_stopping_patience` | 2 |

Wiring the fine-tuned model back into inference (no code change needed):

```bash
export LEGAL_BART_MODEL=Phyboc/legalLens-bart-base      # or a local checkpoint path
python cli.py summarize artifacts/sample_judgment.txt --mode both
```

---

## 8. Training step 3 — document-level evaluation  —  **[TO BUILD]**

No `src/eval/` script exists yet. Intended procedure:

1. Load the IN-ABS **test** split (`text`, `summary`).
2. Run the frozen pipeline (preprocessing → section detection → chunking).
3. Generate one summary per chunk with the fine-tuned model (batch, `num_beams=4`).
4. Aggregate (`postprocess/aggregation.py`) and format (`postprocess/mode_formatter.py`).
5. Score against `reference_summaries.jsonl` with ROUGE-1/2/L and BERTScore.
6. Compare against the zero-shot baseline in `artifacts/baselines/zero_shot.json`.

A fine-tune that does not beat the zero-shot baseline on the test split has not earned its keep.
Report chunk-level and document-level numbers separately, plus the alignment unassigned rate.

---

## 9. Runtime and limitations

- **CPU cost:** alignment is Cross-Encoder-bound at roughly **1–2 minutes per document**.
  3 documents ≈ several minutes; `--full` (7,128 docs) is **multi-day** on CPU.
  Do `--sample`/`--fast` locally, or run the full batch on a GPU.
- **Chunk cap:** `MAX_CHUNKS_PER_DOC = 15` keeps the earliest chunks per section; long
  judgments lose their tails.
- **Structure fallback:** ~30% of documents become `full_document` (no section signal).
- **Alignment is a proxy:** the unassigned rate (13–19% typical) is the share of each
  reference summary never used as a target.
- **`Cite & Verify` does not discriminate** (~98.5% verified at every threshold) — do not
  present it as a working hallucination guard until recalibrated.
- **Existing detector model:** `models/section_classifier.joblib` was pickled with
  scikit-learn 1.9.0 and warns under 1.9.1; harmless but worth re-training.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `can't open file '...\scripts\build_and_push_datasets.py'` | Ran from the repo root | `cd legal-doc-intelligence` first |
| `python` is `...\WindowsApps\PythonSoftwareFoundation...` | Wrong interpreter — Store Python, not the venv | `& "D:\Studies\projects\venv\Scripts\Activate.ps1"`, or call `D:\Studies\projects\venv\Scripts\python.exe` by full path |
| `The '<' operator is reserved for future use` (PowerShell) | `<HF_USERNAME>` placeholder treated as redirection | Use the literal name, e.g. `Phyboc` |
| `Refusing to build: ... _MockCrossEncoder stub` | `sentence-transformers` not installed | `python -m pip install sentence-transformers` |
| `Refusing to build: output directory already contains ...` | Non-empty `--output_dir` | Use a fresh dir or pass `--overwrite` |
| `Not authenticated with the Hugging Face Hub` | No token | `hf auth login` or `export HF_TOKEN=hf_...` |
| `Refusing to truncate silently` / `length validation` failure | A chunk/target exceeds `MAX_INPUT_LENGTH`/`MAX_TARGET_LENGTH` | Lower `config.CHUNK_SIZE` and rebuild |

---

## 11. Command cheat sheet

```bash
# --- setup ---------------------------------------------------------------
cd legal-doc-intelligence
& "D:\Studies\projects\venv\Scripts\Activate.ps1"      # PowerShell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install "transformers>=4.30.0" "sentence-transformers>=2.2.0" \
    "datasets>=2.14.0" "huggingface_hub" "spacy>=3.5.0" "accelerate"
hf auth login

# --- one dataset ---------------------------------------------------------
python scripts/build_dataset.py --sample 10 --output_dir artifacts/training_data/sample10
python scripts/build_dataset.py --full      --output_dir artifacts/training_data/full

# --- ten datasets + upload ----------------------------------------------
python scripts/build_and_push_datasets.py --namespace Phyboc --dry-run
python scripts/build_and_push_datasets.py --namespace Phyboc --only sample10 --private
python scripts/build_and_push_datasets.py --namespace Phyboc --private

# --- upload an existing build -------------------------------------------
python scripts/build_dataset.py --push-to-hub \
    --repo-id Phyboc/legalLens-in-abs-full \
    --output_dir artifacts/training_data/full --private

# --- append / accumulate (see §4.1) -------------------------------------
# Reuse ONE Hub repo and grow it every run, instead of a new dataset per run.
# Upserts by (doc_id, chunk_id); remote-only rows are kept.
python scripts/build_and_push_datasets.py --namespace Phyboc --append --private
python scripts/build_dataset.py --push-to-hub \
    --repo-id Phyboc/legalLens-in-abs-sample10 \
    --output_dir artifacts/training_data/sample10 --private --append

# --- validate / audit ----------------------------------------------------
python scripts/validate_section_alignment.py --sample-size 15
python scripts/audit_alignment_evidence.py --from-jsonl "training data/train.jsonl" \
    --report artifacts/evidence_audit.json

# --- tests ---------------------------------------------------------------
python -m pytest tests/ -q

# --- train (classifier) --------------------------------------------------
python scripts/train_section_classifier.py        # needs MARRO + review CSVs (see §6)

# --- run inference with a fine-tuned checkpoint --------------------------
export LEGAL_BART_MODEL=Phyboc/legalLens-bart-base
python cli.py summarize artifacts/sample_judgment.txt --mode both --output result.json
python cli.py inspect artifacts/sample_judgment.txt
```

Relevant configuration (`config.py`): `CHUNK_SIZE=850`, `CHUNK_OVERLAP=100`,
`MAX_CHUNKS_PER_DOC=15`, `MAX_INPUT_LENGTH=1024`, `MAX_TARGET_LENGTH=448`, `NUM_BEAMS=4`,
`ALIGNMENT_THRESHOLD=0.40`, `ALIGNMENT_RELATIVE_MARGIN=0.90`,
`CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`, `CITATION_THRESHOLD=0.30`,
`BART_MODEL=facebook/bart-base` (env `LEGAL_BART_MODEL`).

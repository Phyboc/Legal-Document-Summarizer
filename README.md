<p align="center">
  <img src="public/favicon.svg" alt="LegalLens" width="64" />
</p>

<h1 align="center">Legal Lens</h1>

<p align="center">
  <strong>Read the fine print.</strong><br/>
  Hallucination-safe summarization of Indian court judgments with source-grounded Cite&nbsp;&amp;&nbsp;Verify.
</p>

<p align="center">
  <a href="#quick-start--web-console"><img src="https://img.shields.io/badge/React_19-+_Vite-61DAFB?logo=react&logoColor=white" alt="React 19 + Vite"/></a>
  <a href="#quick-start--python-pipeline"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="#rest-api"><img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white" alt="FastAPI"/></a>
  <a href="#gradio-space"><img src="https://img.shields.io/badge/Gradio-Space-FF7C00?logo=huggingface&logoColor=white" alt="Gradio Space"/></a>
  <a href="https://huggingface.co/fakechamp24/legallens-bart-base-e3-outcome"><img src="https://img.shields.io/badge/🤗_Model-legallens--bart--base--e3--outcome-FFD21E" alt="HuggingFace Model"/></a>
</p>

---

## What It Does

Legal Lens reads contracts, agreements, and Indian court judgments and explains them **twice**:

| Mode | For Whom | What It Produces |
|---|---|---|
| **Lawyer Mode** | Legal professionals | Structured summary preserving court sections (facts, arguments, reasoning, operative order) with legal vocabulary — **500–700 words** |
| **Citizen Mode** | The person the document binds | Plain-language rewrite with inline jargon simplification and a glossary — **150–250 words** |

Every summary sentence can be **traced back to the source text** via the Cite & Verify system, which uses NLI entailment (not just relevance) to mark each claim as *supported*, *unsupported*, or *contradicted*.

> [!IMPORTANT]
> Legal Lens is an explainer. It is not legal advice and does not replace a qualified lawyer.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         INPUT SURFACES              │
                        │  React Console · Gradio · CLI · API │
                        └────────────────┬────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          src/pipeline.py                                   │
│                                                                            │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌──────────┐               │
│  │ cleaning │→ │ PII       │→ │ section      │→ │ chunker  │               │
│  │ .py      │  │ redaction │  │ detector     │  │ .py      │               │
│  └──────────┘  └───────────┘  │ rule+joblib  │  │ 850 tok  │               │
│                               └──────────────┘  │ overlap  │               │
│                                                 └────┬─────┘               │
│                                                      ▼                     │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────────────┐    │
│  │ cite_verify.py  │← │ mode_format  │← │ summarizer.py (BART)        │    │
│  │ NLI entailment  │  │ + jargon     │  │ section-aware profiles      │    │
│  │ + verbatim      │  │ + extract    │  │ + extractive fallback       │    │
│  └─────────────────┘  └──────────────┘  └─────────────────────────────┘    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Two readings of one document.** Lawyer Mode keeps the court's structure; Citizen Mode rewrites it as plain sentences.
- **Clause by clause, original in view.** Every plain-language note sits beside the clause it came from.
- **Three ways in.** Upload PDF, Word, or image; paste text directly; or scan a page with the camera.
- **Cite & Verify.** Each summary sentence is checked against the source via NLI entailment — not just topical relevance. Calibrated at **F1-maximizing threshold 0.45** (precision 0.84, recall 0.77, AUC 0.891).
- **Outcome preservation.** Operative-order sentences are routed to a dedicated `judgement` section so the court's decision is never quietly dropped by the formatter's word budget.
- **PII redaction.** Aadhaar, PAN, phone, email pattern detection; optional spaCy NER for person names.
- **Eight Indian languages.** English, Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali in the web console.
- **PDF export.** Browser-native `window.print()` → "Save as PDF" — no PDF library needed, Indic scripts render correctly.
- **Paper and dark themes.** Light editorial default; dark theme remembered in `localStorage`; honours `prefers-reduced-motion`.

---

## Evaluation Results

Measured on 100 held-out IN-ABS test judgments, full pipeline, vs the published IN-ABS reference summary:

| Metric | Zero-shot BART | Fine-tuned e3 | e3-outcome |
|---|---:|---:|---:|
| **ROUGE-1** (Lawyer) | 50.93 | 55.15 | **55.4** |
| **ROUGE-2** (Lawyer) | 24.96 | 30.75 | **30.9** |
| **ROUGE-L** (Lawyer) | 44.65 | 49.64 | **50.3** |
| **BERTScore F1** (Lawyer) | 83.51 | 85.04 | **85.2** |
| **Outcome Presence** (full) | 98% | 48% | **53%** |

> [!NOTE]
> Zero-shot BART scores high on outcome presence because it near-copies the source verbatim (the operative sentence is carried along); it loses significantly on ROUGE/BERTScore. The e3-outcome checkpoint lifts the fine-tune's holding rate while preserving summarization quality.

---

## Four Surfaces

| Surface | Stack | Entry Point |
|---|---|---|
| Marketing site + interactive console | React 19 + Vite | `src/`, `npm run dev` |
| Hugging Face Space UI | Python + Gradio | `legal-doc-intelligence/gradio_app.py` |
| LexNarrate (advanced UI) | Python + Gradio | `legal-doc-intelligence/lexnarrate.py` |
| REST API | FastAPI | `legal-doc-intelligence/api.py` |
| Command line | Python | `legal-doc-intelligence/cli.py` |

---

## Quick Start — Web Console

**Requirements:** Node 20+ (Node 22 recommended), npm.

```bash
npm install
npm run dev        # → http://localhost:5173
npm run build      # production bundle in dist/
npm run preview    # serve the build
npm run lint       # oxlint
```

The console runs on mock sample documents and a browser-side clause heuristic — it works without the model server. Deep-link a theme with `http://localhost:5173/?theme=dark`.

### What Is in the Front End

```
src/
├── App.jsx                     theme, language and section layout
├── mockData.js                 languages, loading steps, sample documents, text samples
├── components/
│   ├── Navbar.jsx              plain bar, language menu, theme toggle, mobile menu
│   ├── Hero.jsx                the sentence with the fine print inline
│   ├── UploadInterface.jsx     upload / paste / scan console, results, tabs, export
│   ├── HowItWorks.jsx          slider-driven three-stage walkthrough
│   ├── Features.jsx            capability rows with interface fragments and photography
│   ├── FAQ.jsx                 question list beside the answer panel
│   ├── Footer.jsx              closing call to action and credits
│   └── FadeInSection.jsx       intersection-observer reveal
├── lib/
│   ├── analyzeText.js          keyword clause detection for pasted text
│   └── exportPdf.js            print-ready HTML and PDF export
└── index.css                   design tokens, light/dark themes, layout, motion
```

**Design notes:** One ground, one ink, one accent (deep chambers green). Type: **Newsreader** for display, **Public Sans** for interface. No gradients, glass panels, or icon cards. Photography is real and credited.

---

## Quick Start — Python Pipeline

**Requirements:** Python 3.10+, a few GB of disk for model weights. The summariser loads a BART checkpoint from the Hugging Face Hub unless a local copy exists.

```bash
cd legal-doc-intelligence
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt

python gradio_app.py          # Gradio UI        → http://localhost:7860
python cli.py sample --out judgment.txt
python cli.py summarize judgment.txt --mode both
python cli.py inspect judgment.txt
```

A single judgment takes roughly **60–90 seconds** on a free CPU Space. The Gradio app queues requests so the call survives the wait.

### REST API

```bash
curl -X POST http://localhost:7860/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "…at least 100 characters…", "mode": "both"}'
```

| Route | Purpose |
|---|---|
| `GET /health` | Liveness probe |
| `POST /summarize` | JSON body: `text`, `redact_names`, `strip_preamble`, `mode` (`lawyer` / `citizen` / `both`) |
| `POST /summarize/upload` | Multipart text file; decoded as UTF-8 |

### Pipeline Stages

`src/pipeline.py` orchestrates all processing in a single `process_document()` call:

| # | Stage | Module | Description |
|---|---|---|---|
| 1 | **Clean** | `src/data/cleaning.py` | Normalize whitespace, strip case-caption preamble |
| 2 | **Redact** | `src/data/pii_redaction.py` | Optional Aadhaar/PAN/phone/email/name masking |
| 3 | **Section Detect** | `src/data/section_detector.py` | Rule patterns + joblib classifier → 5 sections; operative-order promotion |
| 4 | **Chunk** | `src/data/chunker.py` | Section-aware, 850-token windows with 100-token overlap |
| 5 | **Summarize** | `src/model/summarizer.py` | BART generation per chunk with section-specific budgets |
| 6 | **Aggregate** | `src/postprocess/aggregation.py` | Merge and deduplicate per section |
| 7 | **Format** | `src/postprocess/mode_formatter.py` | Lawyer + Citizen mode with budget allocation |
| 8 | **Simplify** | `src/postprocess/jargon.py` | 40+ inline substitutions + glossary extraction |
| 9 | **Extract** | `src/postprocess/legal_extract.py` | Dates, amounts, provisions, parties, timeline |
| 10 | **Cite & Verify** | `src/postprocess/cite_verify.py` | MS-MARCO retrieval + NLI entailment verification |

### Models and Configuration

`config.py` resolves the summariser in this order:

1. `models/legallens-bart-base-e3-outcome/` (outcome-preserving checkpoint)
2. `models/legallens-bart-base-e3/`
3. `fakechamp24/legallens-bart-base-e3-outcome` on the Hub

Override with `LEGAL_BART_MODEL`. Other models:

| Model | Purpose |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Evidence matching / citation retrieval (scored over overlapping windows) |
| `cross-encoder/nli-deberta-v3-xsmall` | NLI entailment verification (Cite & Verify) |
| `models/section_classifier.joblib` | Section classification (TF-IDF + LogReg, trained on MARRO + OpenNyAI) |

### Gradio Space

`gradio_app.py` is the Hugging Face Spaces entry point:

- Paste-text and file-upload tabs (PDF via `pypdf`, DOCX via `python-docx`, TXT)
- Redact-names and strip-preamble switches
- Citizen / Lawyer / Citations result tabs with verification counts
- **Export PDF** button using browser print pipeline
- Paper-and-green design system matching the React site

### LexNarrate (Advanced UI)

`lexnarrate.py` is a richer Gradio front end with additional features:

- **Model selector** — switch between e3-outcome, e3, e1, and Hub checkpoints
- **Grounded Q&A** — chatbot answers only from the uploaded document
- **Full-text search** — keyword highlighting with match count
- **Structured extraction** — dates, money, provisions, timeline, entities, parties
- **JSON / Markdown / text export**
- **Animated evaluation dashboard** — scroll-reveal metrics with model comparison charts

Launch with `python lexnarrate.py`.

---

## Data, Training and Evaluation

### Training Pipeline

Training and dataset tooling lives in `legal-doc-intelligence/`:

```bash
python scripts/train_section_classifier.py   # the joblib section classifier (MARRO + OpenNyAI)
python -m src.model.train \
  --data artifacts/training_data/full \
  --output_dir models/legallens-bart-base-e3  # BART fine-tune (Seq2SeqTrainer)
python -m src.eval.evaluate \
  --models facebook/bart-base models/legallens-bart-base-e3 \
  --split test --sample 100 --bertscore \
  --output artifacts/eval/test100.json        # document-level ROUGE / BERTScore / outcome
```

**Key training details:**

| Parameter | Value |
|---|---|
| Base model | `facebook/bart-base` (139M parameters) |
| Learning rate | 3e-5 with 6% warmup |
| Batch size | 4 × 8 gradient accumulation = effective 32 |
| Epochs | 3 with early stopping (patience=2) |
| Max input / target tokens | 1024 / 448 |
| Best checkpoint selection | ROUGE-L on validation |
| Precision | bf16 (if available) / fp16 / fp32 |
| Label smoothing | 0.1 |

### Alignment & Outcome Preservation

The IN-ABS dataset provides only **document-level** reference summaries. Training chunk-level summarization requires aligning each summary sentence to its source chunk:

- **`section_summary_alignment.py`** — Cross-Encoder windowed scoring with abbreviation-aware sentence splitting, multi-label assignment with relative margin, operative-order routing
- **`evidence_alignment.py`** — Optional evidence layer (OFF by default) using deterministic regex extraction (dates, amounts, provisions, citations) for factual alignment

**Outcome-preservation fix:** Operative-order sentences are promoted to the `judgement` section during both training-data build and inference, so the court's final decision is never folded into `reasoning` and trimmed by the word budget.

### Audit & Calibration Scripts

| Script | Purpose |
|---|---|
| `scripts/audit_alignment_evidence.py` | Evidence layer calibration |
| `scripts/audit_outcome_survival.py` | Outcome survival tracking |
| `scripts/calibrate_citations.py` | Citation threshold calibration (634 labelled sentences) |
| `scripts/rescore_outcomes.py` | Consistent outcome scoring across checkpoints |
| `scripts/diagnose_dataset.py` | Dataset diagnostics |
| `scripts/validate_section_alignment.py` | Section alignment validation |

### Tests

```bash
cd legal-doc-intelligence
python -m pytest tests -q
```

The suite covers chunking, section detection, alignment and evidence windows, citation verification, jargon replacement, outcome preservation, and summariser configuration.

---

## Repository Layout

```
.
├── README.md
├── package.json / vite.config.js / vercel.json
├── public/
│   ├── favicon.svg
│   ├── icons.svg
│   └── images/                     credited photographs
├── src/                            React marketing site + interactive console
│   ├── App.jsx
│   ├── mockData.js
│   ├── components/                 Navbar, Hero, UploadInterface, Features, FAQ, etc.
│   ├── lib/                        analyzeText.js, exportPdf.js
│   └── index.css                   design tokens, themes, layout
│
├── legal-doc-intelligence/         Python package: pipeline, API, Gradio, training, tests
│   ├── config.py                   models, budgets, thresholds — the control panel
│   ├── gradio_app.py               Hugging Face Space UI
│   ├── lexnarrate.py               Advanced Gradio UI (Q&A, search, export, metrics)
│   ├── cli.py                      command-line interface
│   ├── requirements.txt
│   ├── src/
│   │   ├── pipeline.py             main orchestrator
│   │   ├── data/
│   │   │   ├── cleaning.py         text normalization, preamble stripping
│   │   │   ├── pii_redaction.py    Aadhaar/PAN/phone/email/name masking
│   │   │   ├── section_detector.py rule + classifier section labelling
│   │   │   ├── chunker.py          section-aware token-bounded chunking
│   │   │   ├── section_summary_alignment.py   training target alignment
│   │   │   └── evidence_alignment.py          evidence-aware alignment layer
│   │   ├── model/
│   │   │   ├── summarizer.py       BART wrapper with section profiles
│   │   │   └── train.py            Seq2SeqTrainer fine-tuning
│   │   ├── postprocess/
│   │   │   ├── aggregation.py      merge & deduplicate chunk summaries
│   │   │   ├── mode_formatter.py   Lawyer + Citizen mode formatting
│   │   │   ├── jargon.py           40+ inline substitutions + glossary
│   │   │   ├── legal_extract.py    deterministic date/money/provision extraction
│   │   │   └── cite_verify.py      NLI-based Cite & Verify
│   │   ├── shared/
│   │   │   ├── cross_encoder.py    MS-MARCO singleton for alignment & retrieval
│   │   │   └── nli.py              DeBERTa-v3 NLI for entailment verification
│   │   └── eval/
│   │       └── evaluate.py         document-level ROUGE/BERTScore/outcome scoring
│   ├── scripts/                    dataset build, audits, calibration
│   ├── tests/                      pytest suite (9 test modules)
│   ├── models/                     section_classifier.joblib (weights not committed)
│   ├── artifacts/                  training-data builds, eval results
│   ├── reports/                    evaluation reports, outcome audits
│   └── notebooks/                  legallens_colab.ipynb
│
├── lexnarrate-web/                 standalone HTML viewer
├── training data/                  train.jsonl
└── Documents/                      guides, reports, presentations
    ├── LegalLens_Paper.docx
    ├── LegalLens_Literature_Review_2026.docx
    └── Team 4 NLP Case Study Presentation.pptx
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Summarization** | BART-base fine-tuned on IN-ABS Indian judgments (`fakechamp24/legallens-bart-base-e3-outcome`) |
| **Evidence Matching** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (retrieval), `cross-encoder/nli-deberta-v3-xsmall` (entailment) |
| **Section Classification** | TF-IDF + Logistic Regression (MARRO + OpenNyAI data) |
| **Backend** | Python, PyTorch, Transformers, Gradio, FastAPI |
| **Frontend** | React 19, Vite, Newsreader + Public Sans typography |
| **Training** | HuggingFace `Seq2SeqTrainer`, `datasets`, `rouge-score`, `bert-score` |
| **Deployment** | Vercel (React), Hugging Face Spaces (Gradio) |

---

## Known Limits

- Pasted text in the web console is explained by a keyword heuristic in the browser, not by the fine-tuned model; the sample documents carry hand-written model-style output.
- The Python pipeline generates English summaries. The eight-language support in the web console is interface plus translated sample documents.
- Scanned PDFs without a text layer cannot be read by the Gradio upload; paste the text or run OCR first.
- A free CPU Space needs 60–90 seconds per judgment. A GPU tier is faster.
- The interface is a demonstration, not a legal service: no accounts, no stored documents, and every export carries the disclaimer.

---

## Credits

**Photography** (via Wikimedia Commons):
- *Calcutta High Court* — Biswarup Ganguly, CC BY 3.0
- *High Court and Old Lighthouse, Madras* — Higginbothams Ltd. / KITLV, CC BY 4.0
- *The First Part of the Institutes of the Lawes of England* (1628) — Smuconlaw, CC BY-SA 3.0

**Models:**
- Summariser — BART-base fine-tuned on Indian judgment sections ([`fakechamp24/legallens-bart-base-e3-outcome`](https://huggingface.co/fakechamp24/legallens-bart-base-e3-outcome))
- Evidence matching — `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Entailment verification — `cross-encoder/nli-deberta-v3-xsmall`

**Built with:** React, Vite, Gradio, FastAPI, PyTorch, Transformers, scikit-learn.

---

## Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/MANIDEEP2407-SYS"><img src="https://github.com/MANIDEEP2407-SYS.png" width="90" height="90" style="border-radius:50%; object-fit:cover;" alt="Akula Manideep Reddy"/><br/>
      <strong>Akula Manideep Reddy</strong></a><br/>
      <sub>Frontend, Model &amp; Integration</sub>
    </td>
    <td align="center">
      <a href="https://github.com/mahima1006"><img src="https://github.com/mahima1006.png" width="90" height="90" style="border-radius:50%; object-fit:cover;" alt="Mahima R Iyer"/><br/>
      <strong>Mahima R Iyer</strong></a><br/>
      <sub>Backend, Model &amp; Integration</sub>
    </td>
    <td align="center">
      <a href="https://github.com/Phyboc"><img src="https://github.com/Phyboc.png" width="90" height="90" style="border-radius:50%; object-fit:cover;" alt="Sivasubramani K J"/><br/>
      <strong>Sivasubramani K J</strong></a><br/>
      <sub>Backend, Model &amp; Integration</sub>
    </td>
    <td align="center">
      <a href="https://github.com/SamanyuNair"><img src="https://github.com/SamanyuNair.png" width="90" height="90" style="border-radius:50%; object-fit:cover;" alt="Samanyu Nair"/><br/>
      <strong>Samanyu Nair</strong></a><br/>
      <sub>Frontend, Model &amp; Integration</sub>
    </td>
  </tr>
</table>

<p align="center"><strong>Team 4 — LegalLens</strong> · NLP Case Study</p>

---

<p align="center"><em>Justice begins with understanding.</em></p>

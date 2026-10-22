# Legal Lens

**Read the fine print.** Legal Lens reads contracts, agreements and Indian court
judgments and explains them twice: a structured summary for a lawyer and a
plain-language summary for the person the document binds. Every summary sentence
can be traced back to the source text.

The repository holds two surfaces over the same idea:

| Surface | Stack | Entry point |
| --- | --- | --- |
| Marketing site + interactive console | React 19 + Vite | `src/`, `npm run dev` |
| Hugging Face Space UI | Python + Gradio | `legal-doc-intelligence/app.py` |
| REST service | FastAPI | `legal-doc-intelligence/api.py` |
| Command line | Python | `legal-doc-intelligence/cli.py` |

---

## What it does

- **Two readings of one document.** *Lawyer mode* keeps the court's structure
  (facts, arguments, reasoning, operative order) and legal vocabulary.
  *Citizen mode* rewrites the same material as plain sentences with a glossary
  for jargon.
- **Clause by clause, original in view.** Every plain-language note sits beside
  the exact clause it came from. Nothing is paraphrased away.
- **Three ways in.** Upload a PDF, Word file or photo; paste text directly; or
  capture a page with the camera in the web console. The Gradio UI accepts
  PDF/DOCX/TXT uploads or pasted text.
- **PDF export.** Both UIs export the summary, the clause pairs and the
  disclaimer through the browser's print pipeline ("Save as PDF"). This keeps
  Telugu, Devanagari and the other scripts correctly shaped instead of shipping a
  broken PDF font.
- **Eight Indian languages in the interface.** The language picker covers
  English, Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi and Bengali. The
  sample documents in the web console carry hand-written translations in all
  eight; the Python summariser currently produces English output.
- **Paper and dark themes.** The web console defaults to a light editorial
  theme, remembers the dark theme in `localStorage`, and honours
  `prefers-reduced-motion`.
- **Outcome preservation.** Section routing deliberately keeps the court's
  operative order (the holding) in its own section so a summary cannot quietly
  drop the decision.

> Legal Lens is an explainer. It is not legal advice and it does not replace a
> qualified lawyer.

---

## Quick start — web console

Requirements: Node 20+ (Node 22 recommended), npm.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # production bundle in dist/
npm run preview    # serve the build
npm run lint       # oxlint
```

The console runs on mock sample documents and a browser-side clause heuristic, so
it works without the model server. Deep-link a theme with
`http://localhost:5173/?theme=dark`.

### What is in the front end

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

Design notes:

- One ground, one ink, one accent (deep chambers green); surfaces are separated
  by tone rather than borders.
- Type: **Newsreader** for display and document prose, **Public Sans** for
  interface text. Five sizes do all the work.
- No background gradients, glass panels or icon cards. Photography is real and
  credited in the footer.
- PDF export builds an A4 document in a hidden frame and calls `window.print()`;
  no PDF library or extra dependency is required.

---

## Quick start — Python pipeline

Requirements: Python 3.10+, a few GB of disk for model weights. The summariser
loads a BART checkpoint from the Hugging Face Hub unless a local copy exists.

```bash
cd legal-doc-intelligence
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt

python app.py                 # Gradio UI        → http://localhost:7860
python api.py                 # FastAPI service  → http://localhost:7860
python cli.py sample --out judgment.txt
python cli.py summarize judgment.txt --mode both
python cli.py inspect judgment.txt
```

A single judgment takes roughly **60–90 seconds** on a free CPU Space. The Gradio
app queues requests so the call survives the wait.

### REST API

```bash
curl -X POST http://localhost:7860/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "…at least 100 characters…", "mode": "both"}'
```

| Route | Purpose |
| --- | --- |
| `GET /health` | liveness probe |
| `POST /summarize` | JSON body: `text`, `redact_names`, `strip_preamble`, `mode` (`lawyer` / `citizen` / `both`) |
| `POST /summarize/upload` | multipart text file; the body is decoded as UTF-8, so use `/summarize` for PDFs |

### Pipeline stages

`src/pipeline.py` orchestrates:

| Stage | Module |
| --- | --- |
| Clean the text, strip the case caption, split paragraphs | `src/data/cleaning.py` |
| Optional person-name redaction | `src/data/pii_redaction.py` |
| Detect sections, promote the operative order | `src/data/section_detector.py` |
| Chunk sections within a token budget | `src/data/chunker.py` |
| Summarise chunk by chunk | `src/model/summarizer.py` |
| Aggregate per section | `src/postprocess/aggregation.py` |
| Format lawyer and citizen outputs | `src/postprocess/mode_formatter.py` |
| Replace jargon, extract dates, amounts and obligations | `src/postprocess/jargon.py`, `src/postprocess/legal_extract.py` |
| Cite each sentence and verify it against the source | `src/postprocess/cite_verify.py` |

### Models and configuration

`legal-doc-intelligence/config.py` resolves the summariser in this order:

1. `models/legallens-bart-base-e3-outcome/` (outcome-preserving checkpoint),
2. `models/legallens-bart-base-e3/`,
3. `fakechamp24/legallens-bart-base-e3-outcome` on the Hub.

Override with `LEGAL_BART_MODEL`. The cross-encoder used for evidence matching is
`cross-encoder/ms-marco-MiniLM-L-6-v2`, scored over overlapping windows; the
section classifier is the small `models/section_classifier.joblib` committed to
the repository. Only that classifier is committed — model weights are not.

### Gradio Space

`app.py` is the Hugging Face Spaces entry point; the UI lives in
`gradio_app.py`. The interface includes:

- paste-text and file-upload tabs (PDF via `pypdf`, DOCX via `python-docx`, TXT),
- redact-names and strip-preamble switches,
- citizen / lawyer / citations result tabs with verification counts,
- an **Export PDF** button that prints a server-built A4 summary,
- a remembered light/dark theme toggle.

`lexnarrate.py` is an alternative, richer Gradio front end (grounded Q&A, search,
JSON/Markdown export, evaluation panels) that can be launched separately with
`python lexnarrate.py`.

---

## Data, training and evaluation

Training and dataset tooling lives in `legal-doc-intelligence/`:

```bash
python scripts/build_dataset.py            # aligned records from the source judgments
python scripts/train_section_classifier.py # the joblib section classifier
python -m src.model.train                  # BART fine-tune
python -m src.eval.evaluate                # ROUGE / BERTScore
```

`scripts/audit_alignment_evidence.py`, `audit_outcome_survival.py`,
`calibrate_citations.py` and `rescore_outcomes.py` are the checks used to keep
citations honest and outcomes present. See `Documents/Dataset_and_Training_Guide.md`
and `Documents/CLI_Reference_and_Training_Steps.md` for the long form.

### Tests

```bash
cd legal-doc-intelligence
python -m pytest tests -q
```

The suite covers chunking, section detection, alignment and evidence windows,
citation verification, jargon replacement, dataset record building, resume
behaviour and summariser configuration.

---

## Repository layout

```
.
├── index.html                  Vite entry, fonts, pre-paint theme bootstrap
├── src/                        React site and console
├── public/images/              credited photographs
├── legal-doc-intelligence/     Python package: pipeline, API, Gradio, scripts, tests
│   ├── app.py                  Space entry point
│   ├── gradio_app.py           Space UI
│   ├── api.py                  FastAPI service
│   ├── cli.py                  command line
│   ├── config.py               models, budgets, thresholds
│   ├── src/                    data, model, postprocess, eval
│   ├── scripts/                dataset build, audits, calibration
│   └── tests/                  pytest suite
├── Documents/                  guides, reports, presentations
└── legal-doc-intelligence/artifacts/   training-data builds and checkpoints
```

---

## Known limits

- Pasted text in the web console is explained by a keyword heuristic in the
  browser, not by the fine-tuned model; the sample documents carry the
  hand-written model-style output.
- The Python pipeline generates English summaries. The eight-language support in
  the web console is interface plus translated sample documents.
- Scanned PDFs without a text layer cannot be read by the Gradio upload; paste
  the text or run OCR first.
- A free CPU Space needs 60–90 seconds per judgment. A GPU tier is faster.
- The interface is a demonstration, not a legal service: no accounts, no stored
  documents, and every export carries the disclaimer.

## Credits

Photography (via Wikimedia Commons):

- *Calcutta High Court* — Biswarup Ganguly, CC BY 3.0.
- *High Court and Old Lighthouse, Madras* — Higginbothams Ltd. / KITLV, CC BY 4.0.
- *The First Part of the Institutes of the Lawes of England* (1628) — Smuconlaw,
  CC BY-SA 3.0.

Summariser: BART-base fine-tuned on Indian judgment sections
(`fakechamp24/legallens-bart-base-e3-outcome`). Evidence matching:
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Built with React, Vite, Gradio, FastAPI,
PyTorch and Transformers.

## Team

Team 4 — LegalLens: Akula Manideep Reddy, Mahima R Iyer, Sivasubramani K J,
Samanyu Nair.

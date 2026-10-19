# Deploying LegalLens to Hugging Face Spaces (Gradio)

The Space serves the Gradio UI in `app.py` + `gradio_app.py`. The model is pulled
from the Hub at runtime (`fakechamp24/legallens-bart-base`), so nothing large is
committed. Free CPU tier (16 GB RAM) fits the models; inference is ~60–90 s/doc.

**You do the auth** — Claude cannot log into your Hugging Face account or create
the Space. These are the exact steps.

## 1. Create the Space

Go to https://huggingface.co/new-space and set:

- **Owner**: your account (e.g. `fakechamp24`)
- **Space name**: `legallens`
- **SDK**: **Gradio**
- **Hardware**: CPU basic (free) — upgrade to a GPU tier later for speed
- Visibility: Public or Private

Create it. You now have a git repo at
`https://huggingface.co/spaces/<you>/legallens`.

## 2. Push the backend into the Space

The Space's root must be the contents of `legal-doc-intelligence/` (so `app.py`,
`src/`, `config.py`, `models/section_classifier.joblib` sit at the Space root).

```bash
# from the repo, after this branch is checked out
cd legal-doc-intelligence
git init -b main                      # a fresh repo just for the Space
git remote add space https://huggingface.co/spaces/<you>/legallens
git add app.py gradio_app.py api.py cli.py config.py requirements.txt README.md \
        src models/section_classifier.joblib scripts Documents
git commit -m "LegalLens Gradio Space"
git push --force space main
```

`git push` will ask for your HF username and a **write token**
(https://huggingface.co/settings/tokens). The Space builds and goes live
automatically after the push.

> Do **not** add the `models/legallens-bart-*` or `models/nli-*` directories —
> they are large and are fetched from the Hub at runtime. `.gitignore` already
> excludes them.

## 3. (Optional) faster, leaner build

The committed `requirements.txt` includes evaluation-only packages the UI does not
need (`bert-score`, `rouge-score`, `datasets`, `fastapi`, `uvicorn`,
`python-multipart`, `spacy`). For a smaller, faster Space image, replace
`requirements.txt` **in the Space repo only** with:

```
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.10.0
joblib>=1.3.0
gradio>=4.44.0
```

(Name redaction via spaCy is then unavailable — it is off by default and the code
degrades gracefully.)

## 4. Verify

Open the Space URL, paste a judgment (the box is pre-filled with a sample), and
click **Summarize**. Confirm the **Lawyer Mode** output ends with a JUDGEMENT
section stating the court's order, and that **Citizen Mode** differs from it.

## Frontend note

The React site in the repo root (`src/`, deployable to Vercel) is currently
mock-only and does not call this Space. To make it live, point its fetch calls at
`https://<you>-legallens.hf.space/` (the Space also exposes the Gradio API) or run
`api.py` as a separate service and call `/summarize`. That wiring is not done yet.

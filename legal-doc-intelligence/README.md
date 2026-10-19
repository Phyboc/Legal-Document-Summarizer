---
title: LegalLens
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# LegalLens — Indian legal judgment summarizer

Turns an Indian court judgment into two summaries — **Lawyer Mode** (structured,
section-aware) and **Citizen Mode** (plain language) — with a Cite & Verify pass
that flags each claim as supported / unsupported / contradicted by the source.

- **Model:** a BART-base fine-tune published at
  [`fakechamp24/legallens-bart-base`](https://huggingface.co/fakechamp24/legallens-bart-base),
  loaded from the Hub at runtime (no weights are committed to this Space).
- **Verification:** `cross-encoder/nli-deberta-v3-xsmall`.
- **Section routing:** a small rule + classifier that keeps the court's operative
  order (the holding) in its own section so the summary does not drop it.

## Performance note

The free Spaces CPU tier has no GPU, so a single judgment takes roughly **60–90
seconds**. The request queue keeps the call alive while it runs. For faster
inference, upgrade the Space hardware to a GPU tier.

## Running locally

```bash
cd legal-doc-intelligence
python -m pip install -r requirements.txt
python app.py              # Gradio UI on http://localhost:7860
# or the FastAPI service:
python api.py              # POST /summarize
```

This app is the inference UI. Dataset building, training and evaluation live in
`scripts/` and `src/`; see `Documents/Dataset_and_Training_Guide.md`.

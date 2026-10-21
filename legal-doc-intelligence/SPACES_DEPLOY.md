# Hosting LegalLens — current options

The app is a Python + BART service (~900 MB of model weights loaded at runtime),
so it needs a host that runs compute. As of this writing Hugging Face gates the
compute SDKs, so read this before picking a path.

## What's free vs paid (Hugging Face, checked Oct 2026)

| Path | Cost | Works for this app? |
|---|---|---|
| **Static** Space | free for everyone | No — can't run the Python/BART backend (browser-only). Fine for the React frontend. |
| **Gradio / Docker** Space on `cpu-basic` | **requires PRO** (~$9/mo) | Yes, once subscribed. |
| **Gradio + ZeroGPU** Space | free **only** if account is >30 days old and in good standing (verified email); otherwise PRO | Yes, when eligible — GPU, fast. |
| **Google Colab + share link** | free | Yes, now — temporary URL (dies with the session). |

This account (`fakechamp24`) is <30 days old, so ZeroGPU and `cpu-basic` both
returned `402 Payment Required`. Until the account ages past 30 days (or you take
PRO), the free working path is **Colab**.

## Free now — Colab public demo

Open `notebooks/legallens_colab.ipynb` in Google Colab, run **step 1 (clone)** and
**step 2 (install)**, then **step 8 (Live public demo)**. It prints a public
`https://*.gradio.live` link, live while the cell runs. Pick a GPU runtime for
~10s summaries. No checkpoint needed — the model loads from the Hub.

## When eligible — ZeroGPU Space (free, permanent, GPU)

Files are ready in this repo:

- `app_zerogpu.py` — ZeroGPU entrypoint (`import spaces` first; the whole pass runs
  inside `@spaces.GPU`). Rename to `app.py` in the Space.
- `requirements-space.txt` — lean deps (no `gradio`/`spaces`/`huggingface_hub`,
  which the ZeroGPU runtime manages). Rename to `requirements.txt` in the Space.
- `README.md` — already carries the Space frontmatter (`sdk: gradio`).

Deploy (after `hf auth login`, once the account is eligible or on PRO):

```bash
# assemble a clean Space dir (only what the app imports — NOT the big checkpoints)
mkdir -p /tmp/space && cp app_zerogpu.py /tmp/space/app.py \
  && cp requirements-space.txt /tmp/space/requirements.txt \
  && cp README.md config.py /tmp/space/ && cp -r src /tmp/space/ \
  && mkdir -p /tmp/space/models && cp models/section_classifier.joblib /tmp/space/models/

hf repos create fakechamp24/legallens --type space --space-sdk gradio --flavor zero-a10g --public --exist-ok
hf upload fakechamp24/legallens /tmp/space . --repo-type space --exclude "**/__pycache__/**"
hf spaces logs fakechamp24/legallens --tail 200   # confirm clean startup
```

## Paid, permanent, no hassle — PRO CPU Space

Subscribe to PRO, then the same `hf repos create … --flavor cpu-basic` works. Use
the plain `app.py` (not the ZeroGPU one) and `requirements-space.txt`. ~60–90s/doc
on CPU.

## Frontend

The React site (repo root `src/`, deployable free to Vercel / GitHub Pages / a
Static Space) is currently **mock-only** and does not call the backend. To make it
live, point its fetch at the running backend's `/summarize` (from `api.py`) or the
Space's Gradio API. That wiring is not done yet.

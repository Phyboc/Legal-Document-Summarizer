"""Hugging Face Spaces entrypoint for LegalLens (Gradio SDK).

Spaces runs ``python app.py`` from the repository root, which is this
``legal-doc-intelligence/`` directory when the Space is created from it. The UI
lives in ``gradio_app.py``; this module only imports that ``demo`` and launches it
with a request queue, because a single summary takes ~60-90s on the free CPU tier
and the queue keeps the request alive instead of timing out.

No model weights are committed: ``config.BART_MODEL`` resolves to the published
Hub checkpoint ``fakechamp24/legallens-bart-base`` (and the NLI model to
``cross-encoder/nli-deberta-v3-xsmall``) whenever no local ``models/`` directory
is present, which is the case on a fresh Space. Only the small rule-based section
classifier (``models/section_classifier.joblib``, ~11 MB) is in the repo.
"""
from gradio_app import demo

demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch()

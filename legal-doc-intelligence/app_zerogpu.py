"""LegalLens — Hugging Face Space (Gradio + ZeroGPU).

``import spaces`` MUST precede any torch / transformers import so the ZeroGPU
runtime can patch CUDA before it is initialized. The whole summarization pass
(BART generation + NLI verification) runs inside the ``@spaces.GPU`` function, so
a GPU is attached only for the ~10-20s a request needs.

Model weights are not committed: ``config.BART_MODEL`` resolves to the published
Hub checkpoint ``fakechamp24/legallens-bart-base`` and the NLI model to
``cross-encoder/nli-deberta-v3-xsmall`` when no local ``models/`` copy exists
(the case on this Space). Only the ~11 MB rule-based section classifier ships.
"""
import spaces  # noqa: F401  (must come first)
import sys

sys.path.insert(0, ".")
import gradio as gr

from src.pipeline import process_document

SAMPLE_TEXT = """The Judgment of the Court was delivered by SAHAI, J.

The appellant was employed as an Assistant Engineer in the Central Public
Works Department. He was placed under suspension on September 3, 1959,
pending a departmental inquiry.

The appellant filed a writ petition in the High Court challenging the
prolonged suspension. Learned counsel for the appellant contended that the
suspension was unjustified.

The respondent State submitted that the suspension was necessary during the
pendency of the departmental inquiry.

We have considered the arguments. In our opinion, a suspension continuing
for over a decade without a determination violates principles of natural
justice.

The order of compulsory retirement dated 25.4.1972 is set aside. The appeal
is allowed with costs."""


@spaces.GPU(duration=120)
def _run_pipeline(text: str, redact_names: bool, strip_preamble: bool) -> dict:
    """Run the full LegalLens pipeline on one judgment (GPU-attached)."""
    return process_document(text, redact_names=redact_names, strip_preamble=strip_preamble)


def summarize_ui(text: str, redact_names: bool, strip_preamble: bool):
    """Summarize a judgment into Lawyer and Citizen modes with verification."""
    if not text or len(text.strip()) < 100:
        return "⚠️ Please enter at least 100 characters of legal text.", "", "", ""

    result = _run_pipeline(text, redact_names, strip_preamble)

    lawyer_text = result["lawyer_mode"]["full_text"]
    citizen_text = result["citizen_mode"]["full_text"]
    stats = (
        f"**Paragraphs:** {result['n_paragraphs']} | "
        f"**Chunks:** {result['n_chunks']} | "
        f"**Sections:** {', '.join(result['sections_detected'])} | "
        f"**Model:** {'fine-tuned' if result['is_finetuned'] else 'base'}\n\n"
        f"**Lawyer Mode:** {result['lawyer_mode']['word_count']} words | "
        f"Verified: {result['lawyer_verification']['verified']}/{result['lawyer_verification']['total']}\n\n"
        f"**Citizen Mode:** {result['citizen_mode']['word_count']} words | "
        f"Verified: {result['citizen_verification']['verified']}/{result['citizen_verification']['total']}"
    )

    citations_md = "### Citations (top 5 by confidence)\n\n"
    top = sorted(result["lawyer_citations"], key=lambda c: -c["confidence"])[:5]
    for c in top:
        citations_md += (
            f"- **{c['confidence']:.2f}** ({c['source_section']}): "
            f"_{c['sentence'][:120]}..._\n"
        )
    return stats, lawyer_text, citizen_text, citations_md


with gr.Blocks(title="LegalLens") as demo:
    gr.Markdown(
        "# ⚖️ LegalLens\n"
        "Turn an Indian court judgment into two summaries — **Lawyer Mode** "
        "(structured) and **Citizen Mode** (plain language) — with a Cite & Verify "
        "pass. The operative order (the holding) is kept in its own section."
    )
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Legal judgment text", lines=15, value=SAMPLE_TEXT)
            with gr.Row():
                redact_names_cb = gr.Checkbox(label="Redact person names", value=False)
                strip_preamble_cb = gr.Checkbox(label="Strip case-caption preamble", value=True)
            run_btn = gr.Button("Summarize", variant="primary")
        with gr.Column(scale=3):
            stats_md = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("👨‍⚖️ Lawyer Mode"):
                    lawyer_out = gr.Textbox(label="Structured legal summary", lines=15)
                with gr.Tab("👥 Citizen Mode"):
                    citizen_out = gr.Textbox(label="Simplified summary", lines=15)
                with gr.Tab("🔗 Citations"):
                    citations_out = gr.Markdown()

    run_btn.click(
        fn=summarize_ui,
        inputs=[text_input, redact_names_cb, strip_preamble_cb],
        outputs=[stats_md, lawyer_out, citizen_out, citations_out],
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch()

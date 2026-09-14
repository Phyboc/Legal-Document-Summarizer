"""Gradio web UI for LegalLens."""
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


def summarize_ui(text: str, redact_names: bool, strip_preamble: bool):
    if not text or len(text.strip()) < 100:
        return "⚠️ Please enter at least 100 characters of legal text.", "", "", ""

    result = process_document(
        text,
        redact_names=redact_names,
        strip_preamble=strip_preamble,
    )

    lawyer_text = result["lawyer_mode"]["full_text"]
    citizen_text = result["citizen_mode"]["full_text"]

    stats = (
        f"**Paragraphs:** {result['n_paragraphs']} | "
        f"**Chunks:** {result['n_chunks']} | "
        f"**Sections:** {', '.join(result['sections_detected'])}\n\n"
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
    gr.Markdown("# ⚖️ LegalLens\nConvert legal judgments to two summaries: Lawyer Mode + Citizen Mode.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Legal judgment text",
                lines=15,
                value=SAMPLE_TEXT,
            )
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
    demo.launch(server_name="0.0.0.0", server_port=7860)

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


# --- Brand: dark charcoal + gold, matching the LexNarrate web front end ---
LEXNARRATE_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.yellow,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#111113",
    body_background_fill_dark="#111113",
    body_text_color="#f4f0e9",
    background_fill_primary="#19191d",
    background_fill_primary_dark="#19191d",
    background_fill_secondary="#1f1f25",
    border_color_primary="#2a2a2f",
    block_background_fill="#19191d",
    block_border_color="#2a2a2f",
    block_label_background_fill="#1f1f25",
    block_label_background_fill_dark="#1f1f25",
    block_label_text_color="#d9a94e",
    block_label_text_color_dark="#d9a94e",
    block_title_text_color="#f4f0e9",
    input_background_fill="#15151a",
    input_border_color="#2a2a2f",
    button_primary_background_fill="#d9a94e",
    button_primary_background_fill_hover="#e7bf6e",
    button_primary_text_color="#17120a",
    color_accent_soft="#3a2f1c",
    body_text_color_subdued="#9b958b",
)

CUSTOM_CSS = """
:root, .gradio-container { --ln-gold:#d9a94e; }
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
#ln-hero {
  background: linear-gradient(135deg,#1d1a12,#262013);
  border: 1px solid #3a2f1c; border-radius: 20px;
  padding: 26px 30px; margin-bottom: 18px;
}
#ln-hero h1 {
  font-size: 30px; font-weight: 700; letter-spacing:-.4px; margin: 0 0 4px; color:#f4f0e9;
}
#ln-hero h1 .gold { color: var(--ln-gold); }
#ln-hero .eyebrow {
  display:inline-block; font-size:11px; letter-spacing:2px; text-transform:uppercase;
  color: var(--ln-gold); font-weight:700; margin-bottom:10px;
}
#ln-hero p { color:#9b958b; font-size:15px; margin:4px 0 0; max-width:720px; line-height:1.6; }
.tabitem { border-radius: 12px; }
button.primary, #ln-run { font-weight:600 !important; }
#ln-foot { color:#6f6a62; font-size:12.5px; font-style:italic; text-align:center;
  margin-top:14px; line-height:1.6; }
.tab-nav button.selected { color: var(--ln-gold) !important; border-bottom-color: var(--ln-gold) !important; }
"""

with gr.Blocks(title="LexNarrate · LegalLens") as demo:
    gr.HTML(
        """
        <div id="ln-hero">
          <div class="eyebrow">⚖ Source-grounded legal summarization</div>
          <h1>Lex<span class="gold">Narrate</span></h1>
          <p>Turn a dense court judgment into two clear summaries — a section-aware
          <b>Lawyer Mode</b> and a plain-language <b>Citizen Mode</b> — with every
          claim checked against the source document, never invented.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Legal judgment text",
                info="Paste a judgment (≥100 characters). The sample below is pre-loaded.",
                lines=15,
                value=SAMPLE_TEXT,
            )
            with gr.Row():
                redact_names_cb = gr.Checkbox(label="Redact person names", value=False)
                strip_preamble_cb = gr.Checkbox(label="Strip case-caption preamble", value=True)
            run_btn = gr.Button("Summarize document", variant="primary", elem_id="ln-run")

        with gr.Column(scale=3):
            stats_md = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Lawyer Mode"):
                    lawyer_out = gr.Textbox(label="Structured legal summary", lines=15)
                with gr.Tab("Citizen Mode"):
                    citizen_out = gr.Textbox(label="Simplified summary", lines=15)
                with gr.Tab("Cite & Verify"):
                    citations_out = gr.Markdown()

    gr.HTML(
        "<div id='ln-foot'>AI-generated summaries should be reviewed by a qualified "
        "legal professional. · Built on the LegalLens fine-tuned BART models · Team 4</div>"
    )

    run_btn.click(
        fn=summarize_ui,
        inputs=[text_input, redact_names_cb, strip_preamble_cb],
        outputs=[stats_md, lawyer_out, citizen_out, citations_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=LEXNARRATE_THEME,
        css=CUSTOM_CSS,
    )

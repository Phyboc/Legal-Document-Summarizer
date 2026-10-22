"""Gradio web UI for LegalLens.

The same paper-and-green system as the React site: a document goes in (pasted,
uploaded as PDF/DOCX/TXT, or picked from a sample), a structured lawyer summary
and a plain citizen summary come out with citations, and the pair can be
exported as a PDF through the browser's print dialog.

PDF text extraction uses pypdf when it is installed; without it the upload tab
explains how to add the dependency instead of failing silently.
"""
import html
import os
import sys
from datetime import datetime

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


def extract_text(path: str) -> tuple[str, str]:
    """Read text from an uploaded file. Returns (text, error message)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", (
                "PDF reading needs the `pypdf` package, which is not installed in this "
                "environment. Paste the judgment text instead, or run "
                "`pip install pypdf` and restart."
            )
        try:
            reader = PdfReader(path)
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            return "", f"Could not read that PDF: {exc}"
        if len(text.strip()) < 100:
            return "", "That PDF produced no selectable text. It may be a scan; paste the text instead."
        return text, ""

    if ext == ".docx":
        try:
            import docx
        except ImportError:
            return "", (
                "Word documents need the `python-docx` package. Save the file as PDF or "
                "text, or install `python-docx`."
            )
        try:
            document = docx.Document(path)
            text = "\n\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as exc:  # noqa: BLE001
            return "", f"Could not read that Word document: {exc}"
        return text, ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(), ""
    except OSError as exc:
        return "", f"Could not read that file: {exc}"


def _print_document(title: str, result: dict, generated_at: str) -> str:
    """The print-only block the browser turns into a PDF."""
    sections = ", ".join(result["sections_detected"]) or "Not identified"
    citations = sorted(result["lawyer_citations"], key=lambda item: -item["confidence"])[:5]

    citation_rows = "".join(
        f"<li><b>{item['confidence']:.2f}</b> ({html.escape(item['source_section'])}): "
        f"{html.escape(item['sentence'][:220])}</li>"
        for item in citations
    ) or "<li>No citations were extracted.</li>"

    lawyer = html.escape(result["lawyer_mode"]["full_text"]).replace("\n", "<br>")
    citizen = html.escape(result["citizen_mode"]["full_text"]).replace("\n", "<br>")

    return f"""
    <div id="ll-print">
      <div class="p-brand">LegalLens</div>
      <h1>{html.escape(title)}</h1>
      <p class="p-meta">{html.escape(generated_at)} &middot; {result['n_paragraphs']} paragraphs &middot;
        {result['n_chunks']} chunks &middot; Sections: {html.escape(sections)}</p>
      <p class="p-meta">Verification &mdash; lawyer mode {result['lawyer_verification']['verified']}/{result['lawyer_verification']['total']},
        citizen mode {result['citizen_verification']['verified']}/{result['citizen_verification']['total']}</p>

      <h2>Plain-language summary</h2>
      <div class="p-summary">{citizen}</div>

      <h2>Structured legal summary</h2>
      <div class="p-body">{lawyer}</div>

      <h2>Citations</h2>
      <ol class="p-cites">{citation_rows}</ol>

      <p class="p-note">LegalLens explains documents for general understanding. It is not legal advice
      and does not replace a qualified lawyer. Generated {html.escape(generated_at)}.</p>
    </div>
    """


def summarize_ui(file_obj, pasted_text: str, redact_names: bool, strip_preamble: bool):
    text = (pasted_text or "").strip()
    title = "Pasted legal text"

    if file_obj:
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", "")
        extracted, error = extract_text(path)
        if error:
            raise gr.Error(error)
        if extracted.strip():
            text = extracted.strip()
            title = os.path.basename(path) or title

    if len(text) < 100:
        raise gr.Error("Enter at least 100 characters, or upload a document with selectable text.")

    result = process_document(
        text,
        redact_names=redact_names,
        strip_preamble=strip_preamble,
    )

    generated_at = datetime.now().strftime("%d %b %Y, %H:%M")
    sections = ", ".join(result["sections_detected"]) or "Not identified"
    stats = f"""
    <div class="ll-stats">
      <span>{result['n_paragraphs']} paragraphs</span>
      <span>{result['n_chunks']} chunks</span>
      <span>Sections: {html.escape(sections)}</span>
      <span>Lawyer verified {result['lawyer_verification']['verified']}/{result['lawyer_verification']['total']}</span>
      <span>Citizen verified {result['citizen_verification']['verified']}/{result['citizen_verification']['total']}</span>
      <span>{'Fine-tuned model' if result['is_finetuned'] else 'Base model'}</span>
    </div>
    """

    citations_md = "### Citations (top five by confidence)\n\n"
    top = sorted(result["lawyer_citations"], key=lambda item: -item["confidence"])[:5]
    if top:
        for item in top:
            citations_md += (
                f"- **{item['confidence']:.2f}** ({item['source_section']}): "
                f"_{item['sentence'][:160]}…_\n"
            )
    else:
        citations_md += "_No citations were extracted for this document._\n"

    return (
        stats,
        result["lawyer_mode"]["full_text"],
        result["citizen_mode"]["full_text"],
        citations_md,
        _print_document(title, result, generated_at),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300..600;1,6..72,300..500&family=Public+Sans:wght@400;500;600&display=swap');

.gradio-container {
  --ll-paper: #ececeb;
  --ll-page: #f7f7f7;
  --ll-panel: #e3e3e1;
  --ll-quality: #d6d6d4;
  --ll-ink: #14150f;
  --ll-muted: #4b4e42;
  --ll-accent: #1e4436;
  --ll-on-accent: #f4f4f4;
  --ll-field: #1e4436;
  --ll-on-field: #f4f4f4;
  background: var(--ll-paper) !important;
  color: var(--ll-ink) !important;
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 0 28px 64px !important;
  font-family: 'Public Sans', ui-sans-serif, system-ui, sans-serif !important;
}

html.dark .gradio-container,
.dark .gradio-container {
  --ll-paper: #12140f;
  --ll-page: #181b14;
  --ll-panel: #1f231a;
  --ll-quality: #2a2f23;
  --ll-ink: #efebdf;
  --ll-muted: #a9ac9c;
  --ll-accent: #c7d9b5;
  --ll-on-accent: #12140f;
  --ll-field: #24392c;
  --ll-on-field: #efebdf;
}

footer { display: none !important; }

.ll-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 0 90px;
}
.ll-wordmark {
  font-family: 'Newsreader', Georgia, serif;
  font-size: 21px;
  letter-spacing: -0.02em;
  color: var(--ll-ink);
}
.ll-head-note { font-size: 13px; color: var(--ll-muted); }

.ll-hero h1 {
  font-family: 'Newsreader', Georgia, serif;
  font-weight: 400;
  font-size: clamp(46px, 8vw, 108px);
  line-height: 0.95;
  letter-spacing: -0.04em;
  margin: 0 0 24px;
  color: var(--ll-ink);
}
.ll-hero h1 em {
  font-style: normal;
  background: var(--ll-panel);
  padding: 0 10px;
  border-radius: 2px;
}
.ll-hero p { font-size: 19px; line-height: 1.5; max-width: 60ch; margin: 0 0 40px; color: var(--ll-ink); }

.ll-strip {
  background: var(--ll-panel);
  color: var(--ll-ink);
  padding: 16px 20px;
  border-radius: 2px;
  margin-bottom: 36px;
  font-family: 'Newsreader', Georgia, serif;
  font-size: 13px;
  line-height: 1.5;
  columns: 2;
  column-gap: 36px;
}
.ll-strip p { margin: 0; max-width: none; }

.gradio-container .prose { color: var(--ll-ink) !important; max-width: none !important; }
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 {
  font-family: 'Newsreader', Georgia, serif !important;
  font-weight: 500 !important;
  color: var(--ll-ink) !important;
  letter-spacing: -0.01em;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.gradio-container .block { padding: 0 !important; }

.gradio-container label span,
.gradio-container .label-wrap > span {
  color: var(--ll-muted) !important;
  font-size: 13px !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
}

.gradio-container textarea,
.gradio-container input[type='text'],
.gradio-container .wrap,
.gradio-container .wrap-inner {
  background: var(--ll-page) !important;
  color: var(--ll-ink) !important;
  border: none !important;
  border-radius: 2px !important;
  box-shadow: none !important;
  font-family: 'Newsreader', Georgia, serif !important;
  font-size: 17px !important;
}
.gradio-container textarea { padding: 14px !important; }

.gradio-container .tab-nav {
  border: none !important;
  gap: 4px !important;
  margin-bottom: 18px !important;
}
.gradio-container .tab-nav button {
  background: transparent !important;
  border: none !important;
  border-radius: 2px !important;
  color: var(--ll-muted) !important;
  font-size: 13px !important;
  padding: 8px 12px !important;
}
.gradio-container .tab-nav button.selected {
  background: var(--ll-panel) !important;
  color: var(--ll-ink) !important;
}

.gradio-container button.primary,
.gradio-container .primary {
  background: var(--ll-accent) !important;
  color: var(--ll-on-accent) !important;
  border: none !important;
  border-radius: 2px !important;
  box-shadow: none !important;
  font-weight: 500 !important;
}
.gradio-container button.primary:hover { filter: none !important; opacity: 0.92; }
.gradio-container button.secondary,
.gradio-container button.sm {
  background: var(--ll-panel) !important;
  color: var(--ll-ink) !important;
  border: none !important;
  border-radius: 2px !important;
  box-shadow: none !important;
}

.ll-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 20px;
  background: var(--ll-ink);
  color: var(--ll-paper);
  padding: 16px 18px;
  border-radius: 2px;
  font-size: 13px;
  margin-bottom: 18px;
}

.ll-note { font-size: 13px; color: var(--ll-muted); margin-top: 10px; }
.ll-export-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  background: var(--ll-panel);
  border-radius: 2px;
  padding: 14px 18px;
  margin-top: 16px;
}
.ll-export-row p { margin: 0; font-size: 14px; color: var(--ll-ink); }

#ll-print { display: none; }

@media print {
  @page { margin: 16mm; }
  body * { visibility: hidden !important; }
  #ll-print, #ll-print * { visibility: visible !important; }
  #ll-print {
    display: block !important;
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    background: #fff;
    color: #15160f;
    font-family: 'Newsreader', Georgia, serif;
    font-size: 11pt;
    line-height: 1.5;
    padding: 0;
  }
  #ll-print h1 { font-size: 24pt; font-weight: 500; margin: 0 0 6pt; }
  #ll-print h2 {
    font-size: 13pt;
    font-weight: 500;
    margin: 18pt 0 6pt;
    border: 0;
  }
  #ll-print .p-brand { font-size: 10pt; color: #4b4e42; margin-bottom: 12pt; }
  #ll-print .p-meta { font-size: 9pt; color: #4b4e42; margin: 0 0 4pt; }
  #ll-print .p-summary,
  #ll-print .p-body { padding: 10pt 12pt; background: #f1f1ee; }
  #ll-print .p-summary { background: #e5e9dd; }
  #ll-print .p-cites { font-size: 9pt; padding-left: 16pt; }
  #ll-print .p-note { margin-top: 18pt; font-size: 8.5pt; color: #4b4e42; }
}

@media (max-width: 720px) {
  .gradio-container { padding: 0 18px 48px !important; }
  .ll-head { padding-bottom: 48px; }
  .ll-hero p { font-size: 17px; }
  .ll-strip { columns: 1; }
}

@media (prefers-reduced-motion: reduce) {
  * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
}
"""

THEME = gr.themes.Base(
    primary_hue="green",
    neutral_hue="stone",
    font=["Public Sans", "ui-sans-serif", "sans-serif"],
    font_mono=["ui-monospace", "monospace"],
)

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300..600&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
"""

INITIAL_RESULTS = """
<div class="ll-note">Nothing has been read yet. Choose a sample, paste a judgment, or upload a
PDF and press <b>Explain this document</b>. The summary, the citations and the PDF export all
appear here.</div>
"""

with gr.Blocks(title="LegalLens — plain-language legal summaries", theme=THEME, css=CSS, head=HEAD) as demo:
    theme_btn = gr.Button("Toggle dark theme", size="sm", elem_id="ll-theme-toggle")
    gr.HTML(
        """
        <div class="ll-head">
          <div class="ll-wordmark">LegalLens</div>
          <div class="ll-head-note">Structured and plain-language summaries for Indian legal documents</div>
        </div>
        <div class="ll-hero">
          <h1>Read the <em>fine print</em>.</h1>
          <p>Paste a judgment or upload a document. LegalLens returns two readings of it: a structured
          legal summary for the lawyer, and a plain-language summary for the person it binds.</p>
        </div>
        <div class="ll-strip">
          <p>The appellant was placed under suspension on September 3, 1959, pending a departmental
          inquiry. Learned counsel for the appellant contended that the suspension was unjustified.
          The respondent State submitted that the suspension was necessary during the pendency of the
          departmental inquiry. We have considered the arguments. In our opinion, a suspension
          continuing for over a decade without a determination violates principles of natural
          justice. The order of compulsory retirement is set aside.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("Paste text"):
                    text_in = gr.Textbox(
                        label="Judgment or contract text",
                        lines=14,
                        value=SAMPLE_TEXT,
                        show_label=False,
                        placeholder="Paste at least 100 characters of legal text…",
                    )
                with gr.Tab("Upload file"):
                    file_in = gr.File(
                        label="PDF, DOCX or TXT",
                        file_types=[".pdf", ".docx", ".txt"],
                        type="filepath",
                    )
                    gr.HTML(
                        '<p class="ll-note">Scanned PDFs without a text layer cannot be read. '
                        'Paste the text in that case.</p>'
                    )

            with gr.Row():
                redact_cb = gr.Checkbox(label="Redact person names", value=False)
                strip_preamble_cb = gr.Checkbox(label="Strip the case-caption preamble", value=True)

            run_btn = gr.Button("Explain this document", variant="primary", size="lg")
            gr.HTML(
                '<p class="ll-note">A full judgment takes about 60–90 seconds on the free CPU tier. '
                'Keep this tab open while it runs.</p>'
            )

        with gr.Column(scale=7):
            stats_html = gr.HTML(INITIAL_RESULTS)
            with gr.Tabs():
                with gr.Tab("Citizen mode"):
                    citizen_out = gr.Textbox(
                        label="Plain-language summary",
                        lines=16,
                        interactive=False,
                        show_label=False,
                    )
                with gr.Tab("Lawyer mode"):
                    lawyer_out = gr.Textbox(
                        label="Structured legal summary",
                        lines=16,
                        interactive=False,
                        show_label=False,
                    )
                with gr.Tab("Citations"):
                    citations_out = gr.Markdown("_Citations appear here after analysis._")

            gr.HTML(
                '<div class="ll-export-row">'
                '<p>Export both summaries and the citations as a PDF.</p>'
                '<span class="ll-note">Opens the print dialog — choose “Save as PDF”.</span>'
                '</div>'
            )
            export_btn = gr.Button("Export PDF", variant="primary")

    print_html = gr.HTML("")

    gr.HTML(
        '<p class="ll-note" style="margin-top:32px">LegalLens explains documents for general '
        'understanding. It is not legal advice and does not replace a qualified lawyer.</p>'
    )

    run_btn.click(
        fn=summarize_ui,
        inputs=[file_in, text_in, redact_cb, strip_preamble_cb],
        outputs=[stats_html, lawyer_out, citizen_out, citations_out, print_html],
    )

    export_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { window.print(); }",
    )

    theme_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
          const root = document.documentElement;
          root.classList.toggle('dark');
          try { localStorage.setItem('ll-theme', root.classList.contains('dark') ? 'dark' : 'light'); } catch (e) {}
        }""",
    )

    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
          try {
            if (localStorage.getItem('ll-theme') === 'dark') document.documentElement.classList.add('dark');
          } catch (e) {}
        }""",
    )


if __name__ == "__main__":
    demo.queue(max_size=8)
    demo.launch(server_name="0.0.0.0", server_port=7860)

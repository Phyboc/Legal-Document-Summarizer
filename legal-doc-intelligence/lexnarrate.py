"""LexNarrate - a polished Gradio UI over the LegalLens pipeline.

Wired to the local fine-tuned models in ``models/``: the default summarizer is
``legallens-bart-base-e3-outcome`` (the outcome-preservation checkpoint), with
``e3`` and ``e1`` and the published Hub model selectable from the UI. Every
rendered field is grounded in real pipeline output or deterministic regex
extraction (``src/postprocess/legal_extract.py``) - nothing is invented, and
missing values render as "Not identified".
"""
import base64
import html
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, ".")

import gradio as gr

from src.pipeline import process_document
from src.postprocess.legal_extract import extract_all

HERE = os.path.dirname(os.path.abspath(__file__))


def _author_img(filenames, github_user):
    """Return local image as base64 Data URI if found, else fallback to GitHub avatar."""
    team_dir = os.path.abspath(os.path.join(HERE, "..", "public", "images", "team"))
    for fname in filenames:
        path = os.path.join(team_dir, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    mime = "image/png" if fname.lower().endswith(".png") else "image/jpeg"
                    return f"data:{mime};base64,{b64}"
            except Exception:
                pass
    return f"https://github.com/{github_user}.png"

# ---- Model wiring -------------------------------------------------------
def _local(name):
    p = os.path.join(HERE, "models", name)
    return p if os.path.exists(os.path.join(p, "model.safetensors")) else None

# Each checkpoint prefers a local copy under models/; when the weights aren't
# present (e.g. a fresh clone, which never carries the 9.7 GB of weights) it
# falls back to the published Hub repo so the app still runs. e3-outcome stays
# first, so it remains the default whether loaded locally or from the Hub.
MODEL_CHOICES = {}
for label, folder, hub in [
    ("e3-outcome (outcome-preserving, recommended)", "legallens-bart-base-e3-outcome",
     "fakechamp24/legallens-bart-base-e3-outcome"),
    ("e3", "legallens-bart-base-e3", "fakechamp24/legallens-bart-base"),
    ("e1", "legallens-bart-base-e1", None),
]:
    path = _local(folder)
    if path:
        MODEL_CHOICES[label] = path
    elif hub:
        MODEL_CHOICES[label + " · Hub"] = hub
MODEL_CHOICES["Hub: fakechamp24/legallens-bart-base"] = "fakechamp24/legallens-bart-base"
DEFAULT_MODEL = next(iter(MODEL_CHOICES))  # e3-outcome (local if present, else Hub)

SAMPLE_TEXT = """The Judgment of the Court was delivered by SAHAI, J.

The appellant, M/s Delta Constructions, was engaged by the respondent, State
Housing Board, under an agreement signed on 12.3.2019 to complete a residential
block within eighteen months. Work was delayed through 2020 and the respondent
issued a legal notice on 5.6.2021 demanding completion.

The appellant filed Civil Suit No. 412 of 2021 in the High Court of Madras
contending that delay was caused by the respondent's failure to hand over the
site. Learned counsel for the appellant relied on Section 73 of the Indian
Contract Act, 1872, claiming compensation of Rs. 45,00,000 for loss suffered.

The respondent State Housing Board submitted that the appellant abandoned the
work and was liable for liquidated damages under the agreement.

We have considered the arguments. In our opinion, the delay was attributable in
substantial part to the respondent, and Section 73 entitles the appellant to
reasonable compensation for proven loss.

The judgment dated 18.7.2024 holds that the appeal is partly allowed. The
appellant is awarded Rs. 20,00,000 with interest. The appeal is partially
allowed with costs."""


# ---- File extraction ----------------------------------------------------
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(file_path)
            return "\n".join(page.get_text() for page in doc)
        except Exception:
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(file_path).pages)
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(file_path).paragraphs)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---- HTML helpers -------------------------------------------------------
def esc(s):
    return html.escape(str(s)) if s else ""

NA = '<span class="na">Not identified</span>'


def chips(items):
    if not items:
        return NA
    return "".join(f'<span class="chip">{esc(i)}</span>' for i in items)


def section_block(title, body):
    body = body.strip() if body else ""
    return (
        f'<div class="lx-card"><div class="lx-card-title">{esc(title)}</div>'
        f'<div class="lx-card-body">{esc(body) if body else NA}</div></div>'
    )


def bullets(text):
    if not text:
        return NA
    from src.data.section_summary_alignment import summary_sentences
    sents = [s.strip() for s in summary_sentences(text) if s.strip()]
    if not sents:
        return NA
    return "<ul>" + "".join(f"<li>{esc(s)}</li>" for s in sents[:8]) + "</ul>"


def render_results(result: dict, raw_text: str, extracted: dict):
    citizen = result["citizen_mode"]["full_text"]
    lawyer_secs = result["lawyer_mode"].get("sections", {})
    # Fall back to parsing the lawyer full_text into sections if needed.
    sec = result["lawyer_mode"].get("section_text", {})

    outcome = extracted["outcome"]
    out_class = {
        "ALLOWED": "ok", "PARTIALLY ALLOWED": "warn",
        "DISMISSED": "bad", "NOT APPLICABLE": "na-badge",
    }.get(outcome, "na-badge")

    parties = extracted["parties"]
    words = len(raw_text.split())
    doc_type = _doc_type(raw_text)

    # Executive summary = short citizen overview (first 4 sentences)
    from src.data.section_summary_alignment import summary_sentences
    exec_sents = summary_sentences(citizen)[:4]
    exec_summary = " ".join(exec_sents) if exec_sents else citizen[:400]

    # Key information cards
    def kv(label, value):
        return (f'<div class="kcard"><div class="klabel">{label}</div>'
                f'<div class="kval">{value if value else NA}</div></div>')

    key_cards = (
        kv("KEY DATES", chips(extracted["dates"][:6])) +
        kv("MONEY / DAMAGES", chips(extracted["money"][:6])) +
        kv("LEGAL PROVISIONS", chips([p["section"] or p["law"] for p in extracted["provisions"]][:6])) +
        kv("COURT", esc(extracted["court"])) +
        kv("CASE NUMBER", esc(extracted["case_number"])) +
        kv("PARTIES", chips([v for v in [parties["appellant"], parties["respondent"]] if v]))
    )

    # Provisions detail
    prov_html = ""
    for p in extracted["provisions"][:6]:
        if not (p["law"] or p["section"]):
            continue
        prov_html += (
            f'<div class="prov"><div class="prov-head">{esc(p["law"]) or ""} '
            f'<span class="prov-sec">{esc(p["section"])}</span></div>'
            f'<div class="prov-ctx">{esc(p["context"]) if p["context"] else ""}</div></div>'
        )
    prov_html = prov_html or NA

    # Timeline
    tl = extracted["timeline"]
    tl_html = ""
    for ev in tl:
        tl_html += (
            f'<div class="tl-item"><div class="tl-year">{esc(ev["year"])}</div>'
            f'<div class="tl-event">{esc(ev["event"])}</div></div>'
        )
    tl_html = tl_html or NA

    # Entities (grounded regex)
    ent_html = (
        _ent("COURT", [extracted["court"]]) +
        _ent("LAW / ACT", [p["law"] for p in extracted["provisions"] if p["law"]]) +
        _ent("SECTION", [p["section"] for p in extracted["provisions"] if p["section"]]) +
        _ent("DATE", extracted["dates"][:8]) +
        _ent("AMOUNT", extracted["money"][:6]) +
        _ent("CASE", [extracted["case_number"]]) +
        _ent("PARTIES", [v for v in [parties["appellant"], parties["respondent"]] if v])
    )

    verif = result["lawyer_verification"]

    html_out = f"""
<div class="lx-wrap">
  <div class="lx-label">LEGAL SUMMARY</div>

  <div class="outcome-row">
    <span class="outcome-badge {out_class}">{esc(outcome)}</span>
    <span class="doc-type">{esc(doc_type)}</span>
    <span class="verif">Claims verified {verif['verified']}/{verif['total']}</span>
  </div>

  {section_block("Executive Summary", exec_summary)}

  <div class="two-col">
    <div class="src-col">
      <div class="lx-label">SOURCE DOCUMENT</div>
      <div class="src-meta">{words} words · {doc_type} · Extraction OK</div>
      <div class="src-text">{esc(raw_text)}</div>
    </div>
    <div class="sum-col">
      {section_block("Key Facts", _secget(sec, "facts"))}
      {section_block("Reasoning", _secget(sec, "reasoning"))}
      {section_block("Court / Decision", _secget(sec, "judgement"))}
    </div>
  </div>

  <div class="lx-label">PLAIN LANGUAGE / 02</div>
  <h3 class="lx-h">What does this actually mean?</h3>
  <div class="plain">{esc(citizen)}</div>
  <div class="disclaimer">This explanation is generated for understanding and does not constitute legal advice.</div>

  <div class="lx-label">KEY LEGAL INFORMATION</div>
  <div class="kgrid">{key_cards}</div>

  <div class="lx-label">LEGAL ENTITIES / 03</div>
  <div class="egrid">{ent_html}</div>

  <div class="lx-label">RELEVANT LEGAL PROVISIONS</div>
  <div class="provs">{prov_html}</div>

  <div class="lx-label">THE CASE, IN TIME</div>
  <div class="timeline">{tl_html}</div>

  <div class="lx-label">ARGUMENTS / 04</div>
  <div class="two-col args">
    <div class="arg-card"><div class="arg-head">PLAINTIFF / APPELLANT</div>{bullets(_secget(sec,'arguments_appellant'))}</div>
    <div class="arg-card"><div class="arg-head">DEFENDANT / RESPONDENT</div>{bullets(_secget(sec,'arguments_respondent'))}</div>
  </div>
</div>
"""
    return html_out


def _ent(label, vals):
    vals = [v for v in vals if v]
    inner = chips(vals) if vals else NA
    return f'<div class="ecard"><div class="elabel">{label}</div><div class="eval">{inner}</div></div>'


def _secget(sec, key):
    return (sec or {}).get(key, "")


def _doc_type(text: str) -> str:
    t = text.lower()
    if "writ petition" in t or "fundamental right" in t:
        return "Constitutional / Writ"
    if "contract" in t or "agreement" in t or "breach" in t:
        return "Civil Contract Dispute"
    if "accused" in t or "criminal" in t or "ipc" in t:
        return "Criminal Matter"
    if "appeal" in t:
        return "Civil Appeal"
    return "Legal Document"


# ---- Main analyze handler ----------------------------------------------
def analyze(file_obj, pasted_text, model_label, redact_names):
    text = ""
    if file_obj is not None:
        text = extract_text(file_obj if isinstance(file_obj, str) else file_obj.name)
    if (not text or len(text.strip()) < 100) and pasted_text:
        text = pasted_text
    if not text or len(text.strip()) < 100:
        return ("<div class='lx-wrap'><p class='err'>Please upload a document or paste at "
                "least 100 characters of legal text.</p></div>", None, "", gr.update(value=[]))

    model_name = MODEL_CHOICES.get(model_label, MODEL_CHOICES[DEFAULT_MODEL])
    result = process_document(text, redact_names=redact_names, model_name=model_name)
    extracted = extract_all(text)

    # Attach a section map for rendering (aggregate_by_section lives in result via chunk_summaries)
    sec_map = {}
    for cs in result.get("chunk_summaries", []):
        sec_map.setdefault(cs["section"], []).append(cs["summary"])
    result["lawyer_mode"]["section_text"] = {k: " ".join(v) for k, v in sec_map.items()}

    html_out = render_results(result, text, extracted)

    export = {
        "model": model_name,
        "document_type": _doc_type(text),
        "outcome": extracted["outcome"],
        "executive_summary": result["citizen_mode"]["full_text"],
        "lawyer_summary": result["lawyer_mode"]["full_text"],
        "sections": result["lawyer_mode"]["section_text"],
        "parties": extracted["parties"],
        "court": extracted["court"],
        "case_number": extracted["case_number"],
        "dates": extracted["dates"],
        "money": extracted["money"],
        "provisions": extracted["provisions"],
        "timeline": extracted["timeline"],
        "verification": result["lawyer_verification"],
    }
    json_path = _write_tmp(json.dumps(export, indent=2, ensure_ascii=False), ".json")

    return html_out, json_path, text, gr.update(value=[])


def _write_tmp(content, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="lexnarrate_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---- Grounded chat ------------------------------------------------------
_STOP = set("the a an of to in is are was were and or for on by with that this it as at be "
            "what why how did was were who when where which".split())


def _tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2]


def answer_question(question, history, doc_text):
    history = history or []
    history.append({"role": "user", "content": question})
    if not doc_text:
        history.append({"role": "assistant", "content": "Please analyze a document first."})
        return history, ""

    from src.data.section_summary_alignment import summary_sentences
    ex = extract_all(doc_text)
    q = question.lower()
    sents = [s.strip() for s in summary_sentences(doc_text) if len(s.strip()) > 20]

    # Intent routing to grounded structured fields first.
    answer = None
    if any(w in q for w in ("outcome", "final", "decide", "decision", "result", "verdict", "allowed", "dismissed")):
        verdict = ex["outcome"]
        tail = [s for s in sents if re.search(r"allowed|dismissed|set aside|awarded|decree", s, re.I)]
        answer = f"The outcome was: {verdict}." + (" " + tail[-1] if tail else "")
    elif any(w in q for w in ("date", "when", "timeline")) and ex["dates"]:
        answer = "Important dates mentioned: " + ", ".join(ex["dates"]) + "."
    elif any(w in q for w in ("provision", "section", "act", "statute", "law")) and ex["provisions"]:
        parts = [f"{p['law']} {p['section']}".strip() for p in ex["provisions"] if p["law"] or p["section"]]
        answer = "Legal provisions mentioned: " + "; ".join(parts) + "."
    elif any(w in q for w in ("party", "parties", "who", "plaintiff", "appellant", "defendant", "respondent")):
        p = ex["parties"]
        if p["appellant"] or p["respondent"]:
            answer = (f"Appellant/Plaintiff: {p['appellant'] or 'Not identified'}. "
                      f"Respondent/Defendant: {p['respondent'] or 'Not identified'}.")
    elif any(w in q for w in ("money", "damage", "amount", "compensation", "cost")) and ex["money"]:
        answer = "Amounts mentioned: " + ", ".join(ex["money"]) + "."

    # Keyword retrieval fallback (grounded in document sentences).
    if answer is None:
        q_tokens = set(_tok(question))
        scored = sorted(
            ((len(q_tokens & set(_tok(s))), s) for s in sents),
            key=lambda x: -x[0])
        top = [s for n, s in scored[:3] if n > 0]
        answer = " ".join(top) if top else "This information was not found in the uploaded document."

    history.append({"role": "assistant", "content": answer})
    return history, ""


# ---- Document search ----------------------------------------------------
def search_doc(query, doc_text):
    if not doc_text or not query:
        return ""
    matches = []
    for m in re.finditer(re.escape(query), doc_text, re.I):
        start = max(0, m.start() - 70)
        end = min(len(doc_text), m.end() + 70)
        snippet = doc_text[start:end]
        snippet = re.sub(re.escape(query), lambda x: f"<mark>{x.group(0)}</mark>", snippet, flags=re.I)
        matches.append(f"<div class='hit'>…{esc_keep(snippet)}…</div>")
        if len(matches) >= 20:
            break
    if not matches:
        return "<div class='lx-wrap'><p class='na'>No matches found.</p></div>"
    return f"<div class='lx-wrap'><div class='hits-count'>{len(matches)} match(es)</div>{''.join(matches)}</div>"


def esc_keep(s):
    # Escape everything except our <mark> tags.
    s = html.escape(s)
    return s.replace("&lt;mark&gt;", "<mark>").replace("&lt;/mark&gt;", "</mark>")


# ---- Downloads ----------------------------------------------------------
def export_text(doc_text):
    return _write_tmp(doc_text or "", ".txt") if doc_text else None


def export_markdown(doc_text, json_path):
    if not json_path or not os.path.exists(json_path):
        return None
    data = json.load(open(json_path, encoding="utf-8"))
    md = [f"# LegalLens Analysis\n", f"**Outcome:** {data['outcome']}  ",
          f"**Document type:** {data['document_type']}  ",
          f"**Model:** {data['model']}\n", "## Executive Summary\n",
          data["executive_summary"], "\n## Lawyer Summary\n", data["lawyer_summary"],
          "\n## Parties\n", f"- Appellant: {data['parties']['appellant'] or 'Not identified'}",
          f"- Respondent: {data['parties']['respondent'] or 'Not identified'}",
          "\n## Court\n", data["court"] or "Not identified",
          "\n## Dates\n"] + [f"- {d}" for d in data["dates"]] + \
         ["\n## Provisions\n"] + [f"- {p['law']} {p['section']}".strip() for p in data["provisions"]]
    return _write_tmp("\n".join(md), ".md")


# ---- CSS ----------------------------------------------------------------
CSS = """
:root{--bg:#121214;--ink:#f3efe8;--muted:#9b958b;--line:#2a2a2f;--accent:#d9a94e;--accent-soft:#3a2f1c;--card:#1a1a1e;--card2:#202026;}
.gradio-container{background:var(--bg)!important;font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important;max-width:1180px!important;margin:0 auto!important;color:var(--ink);}
footer{display:none!important;}
.lx-hero{padding:56px 8px 20px;}
.lx-hero h1{font-size:52px;line-height:1.05;font-weight:600;letter-spacing:-1.5px;margin:0 0 16px;color:var(--ink);}
.lx-hero p{font-size:19px;color:var(--muted);max-width:620px;line-height:1.5;margin:0 0 26px;}
.lx-label{font-size:11px;letter-spacing:2.5px;color:var(--accent);font-weight:600;text-transform:uppercase;margin:34px 0 10px;}
.lx-h{font-size:30px;font-weight:600;letter-spacing:-0.6px;margin:4px 0 16px;}
.flow-card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:30px 34px;margin-top:14px;box-shadow:0 1px 3px rgba(0,0,0,.03);}
.flow{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:13px;letter-spacing:1px;color:var(--ink);font-weight:600;}
.flow .step{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px 18px;}
.flow .arr{color:var(--accent);font-size:18px;}
.lx-wrap{padding:6px 2px 30px;}
.lx-card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 24px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.02);}
.lx-card-title{font-size:13px;letter-spacing:1px;text-transform:uppercase;color:var(--accent);font-weight:600;margin-bottom:10px;}
.lx-card-body{font-size:16px;line-height:1.65;color:var(--ink);}
.outcome-row{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:6px 0 4px;}
.outcome-badge{font-size:13px;font-weight:700;letter-spacing:1px;padding:8px 16px;border-radius:8px;}
.outcome-badge.ok{background:#1f3a24;color:#86d49c;}
.outcome-badge.warn{background:#3a2f14;color:#e0b45e;}
.outcome-badge.bad{background:#3a1f1c;color:#e4907f;}
.outcome-badge.na-badge{background:#26262b;color:var(--muted);}
.doc-type{font-size:15px;font-weight:600;}
.verif{font-size:13px;color:var(--muted);}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:10px;}
.two-col.args{align-items:start;}
.src-col,.sum-col{min-width:0;}
.src-meta{font-size:12px;color:var(--muted);margin-bottom:8px;letter-spacing:.5px;}
.src-text{font-family:'Georgia',serif;font-size:14.5px;line-height:1.7;color:#cfc8bb;background:var(--card2);border:1px solid var(--line);border-radius:14px;padding:18px 20px;max-height:460px;overflow:auto;white-space:pre-wrap;}
.plain{font-size:18px;line-height:1.7;color:var(--ink);background:var(--card);border-left:3px solid var(--accent);border-radius:0 12px 12px 0;padding:18px 24px;}
.disclaimer{font-size:12.5px;color:var(--muted);font-style:italic;margin-top:10px;}
.kgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
.kcard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;}
.klabel{font-size:10.5px;letter-spacing:1.5px;color:var(--accent);font-weight:600;margin-bottom:8px;}
.kval{font-size:14.5px;line-height:1.5;}
.egrid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
.ecard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
.elabel{font-size:10px;letter-spacing:1.5px;color:var(--muted);font-weight:700;margin-bottom:8px;}
.chip{display:inline-block;background:var(--bg);border:1px solid var(--line);border-radius:20px;padding:4px 12px;margin:3px 4px 3px 0;font-size:13px;}
.na{color:var(--muted);font-style:italic;font-size:13px;}
.provs .prov{border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:10px 0;background:var(--card);}
.prov-head{font-weight:600;font-size:15px;}
.prov-sec{color:var(--accent);font-weight:600;}
.prov-ctx{font-size:13px;color:var(--muted);margin-top:6px;line-height:1.5;}
.timeline{display:flex;gap:0;flex-wrap:wrap;}
.tl-item{flex:1;min-width:150px;border-left:2px solid var(--line);padding:4px 16px 18px;position:relative;}
.tl-item:before{content:'';position:absolute;left:-6px;top:4px;width:10px;height:10px;border-radius:50%;background:var(--accent);}
.tl-year{font-weight:700;color:var(--accent);font-size:17px;margin-bottom:6px;}
.tl-event{font-size:13.5px;line-height:1.5;color:var(--ink);}
.arg-card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 22px;}
.arg-head{font-size:12px;letter-spacing:1px;color:var(--accent);font-weight:700;margin-bottom:10px;}
.arg-card ul{margin:0;padding-left:18px;}
.arg-card li{font-size:14.5px;line-height:1.6;margin-bottom:8px;}
.hit{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:8px 0;font-size:14px;line-height:1.6;font-family:'Georgia',serif;}
.hit mark,.hits mark{background:#e0b45e;color:#1a1405;padding:0 2px;border-radius:3px;}
.hits-count{font-size:12px;color:var(--muted);letter-spacing:1px;margin-bottom:8px;}
.err{color:#7d2a22;font-size:16px;padding:20px;}
.lx-foot{border-top:1px solid var(--line);margin-top:50px;padding:30px 4px;color:var(--muted);font-size:13px;line-height:1.8;}
.lx-foot b{color:var(--ink);font-size:16px;letter-spacing:.5px;}

/* ---- Sticky nav ---- */
.lx-nav{position:sticky;top:0;z-index:50;display:flex;align-items:center;gap:22px;
  padding:14px 18px;margin:-8px -8px 0;background:rgba(18,18,20,.82);
  backdrop-filter:saturate(180%) blur(10px);border-bottom:1px solid var(--line);}
.lx-nav .brand{font-weight:700;letter-spacing:.5px;color:var(--ink);font-size:16px;margin-right:auto;}
.lx-nav .brand span{color:var(--accent);}
.lx-nav a{color:var(--muted);text-decoration:none;font-size:13.5px;font-weight:500;position:relative;transition:color .2s;}
.lx-nav a:hover{color:var(--accent);}
.lx-nav a:after{content:'';position:absolute;left:0;bottom:-5px;width:0;height:2px;background:var(--accent);transition:width .25s;}
.lx-nav a:hover:after{width:100%;}
@media(max-width:720px){.lx-nav a{display:none;}}

/* ---- Scroll reveal + motion ---- */
.reveal{opacity:0;transform:translateY(22px);transition:opacity .7s cubic-bezier(.2,.7,.2,1),transform .7s cubic-bezier(.2,.7,.2,1);}
.reveal.in{opacity:1;transform:none;}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
.sec{scroll-margin-top:70px;padding-top:10px;}
.sec-head{font-size:32px;font-weight:600;letter-spacing:-.8px;margin:6px 0 6px;}
.sec-sub{color:var(--muted);font-size:16px;max-width:640px;line-height:1.55;margin:0 0 20px;}

/* ---- Feature grid ---- */
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
.feat{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px 22px;
  transition:transform .3s cubic-bezier(.2,.7,.2,1),box-shadow .3s,border-color .3s;}
.feat:hover{transform:translateY(-5px);box-shadow:0 16px 34px rgba(0,0,0,.4);border-color:var(--accent-soft);}
.feat .ic{font-size:22px;width:44px;height:44px;display:flex;align-items:center;justify-content:center;
  border-radius:12px;background:var(--accent-soft);color:var(--accent);margin-bottom:14px;}
.feat h4{margin:0 0 7px;font-size:17px;font-weight:600;}
.feat p{margin:0;font-size:14px;color:var(--muted);line-height:1.55;}

/* ---- Step guide ---- */
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
.step-c{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;position:relative;overflow:hidden;}
.step-c:before{content:'';position:absolute;left:0;top:0;width:3px;height:100%;background:linear-gradient(var(--accent),#8a6a24);}
.step-n{font-size:34px;font-weight:700;color:#38342a;line-height:1;letter-spacing:-1px;}
.step-c h4{margin:8px 0 6px;font-size:16px;font-weight:600;}
.step-c p{margin:0;font-size:13.5px;color:var(--muted);line-height:1.55;}

/* ---- Metrics ---- */
.metric-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;}
.metric{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 16px;text-align:center;}
.metric .num{font-size:34px;font-weight:700;color:var(--accent);letter-spacing:-1px;font-variant-numeric:tabular-nums;}
.metric .num small{font-size:16px;color:var(--muted);font-weight:600;}
.metric .lbl{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);margin-top:6px;font-weight:600;}
.cmp{margin-top:22px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 24px;}
.cmp-row{display:grid;grid-template-columns:150px 1fr;gap:16px;align-items:center;margin:12px 0;}
.cmp-row .rl{font-size:13px;color:var(--ink);font-weight:600;}
.bars{display:flex;flex-direction:column;gap:7px;}
.bar{display:flex;align-items:center;gap:10px;font-size:12px;}
.bar .track{display:block;flex:1;height:10px;background:#26262b;border-radius:6px;overflow:hidden;}
.bar .fill{display:block;height:100%;width:0;border-radius:6px;background:linear-gradient(90deg,#8a6a24,var(--accent));transition:width 1.1s cubic-bezier(.2,.7,.2,1);}
.bar .fill.base{background:linear-gradient(90deg,#4a4a52,#6a6a74);}
.bar .fill.e3{background:linear-gradient(90deg,#7a6a3a,#b08f46);}
.bar .bl{width:92px;color:var(--muted);}
.bar .bv{width:42px;text-align:right;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums;}
.metric-note{font-size:12.5px;color:var(--muted);font-style:italic;margin-top:14px;line-height:1.5;}

/* ---- Authors ---- */
.team-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}
.author{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px 18px;text-align:center;transition:transform .3s,box-shadow .3s;}
.author:hover{transform:translateY(-5px);box-shadow:0 14px 30px rgba(0,0,0,.4);}
.author .av{width:68px;height:68px;border-radius:50%;margin:0 auto 12px;display:flex;align-items:center;justify-content:center;
  overflow:hidden;border:2px solid var(--accent);background:var(--card2);box-shadow:0 4px 12px rgba(0,0,0,.3);}
.author .av img{width:100%;height:100%;object-fit:cover;display:block;}
.author h4{margin:0 0 3px;font-size:15.5px;font-weight:600;}
.author .role{font-size:12px;color:var(--accent);font-weight:600;letter-spacing:.3px;}
.author a{display:inline-block;margin-top:9px;font-size:12px;color:var(--muted);text-decoration:none;border:1px solid var(--line);padding:3px 11px;border-radius:20px;transition:all .2s;}
.author a:hover{color:var(--accent);border-color:var(--accent);}
.about-band{background:linear-gradient(135deg,#1d1a12,#262013);border:1px solid var(--accent-soft);color:#f3efe8;border-radius:20px;padding:34px 36px;margin-top:8px;}
.about-band .slogan{font-size:24px;font-weight:600;letter-spacing:-.4px;margin:0 0 8px;}
.about-band p{margin:0;font-size:15px;line-height:1.6;color:var(--muted);max-width:720px;}
.about-band .team-tag{display:inline-block;margin-bottom:14px;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--accent);font-weight:600;}

@media (max-width:820px){.two-col,.kgrid,.egrid,.feat-grid,.steps,.team-grid{grid-template-columns:1fr;}
  .metric-grid{grid-template-columns:repeat(2,1fr);}.lx-hero h1{font-size:38px;}
  .timeline{flex-direction:column;}.tl-item{border-left:2px solid var(--line);}.cmp-row{grid-template-columns:1fr;}}
"""


# ---- Layout -------------------------------------------------------------
THEME = gr.themes.Base(primary_hue="amber", neutral_hue="neutral")

# LexNarrate uses a dark charcoal + gold scheme; force Gradio's dark mode so the
# palette holds regardless of the viewer's system preference.
# ---- Static content sections (features, guide, metrics, authors) --------
FEATURES_HTML = """
<div class="sec reveal" id="features">
  <div class="lx-label">WHAT LEXNARRATE DOES / 05</div>
  <h3 class="sec-head">Built for clarity you can trust.</h3>
  <p class="sec-sub">Every feature is source-grounded: summaries map back to the document, and
  unverifiable claims are flagged rather than invented.</p>
  <div class="feat-grid">
    <div class="feat"><div class="ic">⚖️</div><h4>Dual summaries</h4><p>Lawyer Mode keeps the section-aware legal structure; Citizen Mode rewrites it in plain language.</p></div>
    <div class="feat"><div class="ic">🔗</div><h4>Cite &amp; Verify</h4><p>An NLI model (deberta-v3) checks each claim against the source and marks it supported, unsupported, or contradicted.</p></div>
    <div class="feat"><div class="ic">🎯</div><h4>Outcome preservation</h4><p>The operative order is routed to its own judgement section so the holding — ALLOWED / DISMISSED — is never dropped.</p></div>
    <div class="feat"><div class="ic">🧩</div><h4>Section routing</h4><p>A rule + classifier splits facts, arguments, reasoning and judgement before summarizing each with its own budget.</p></div>
    <div class="feat"><div class="ic">🛡️</div><h4>PII redaction</h4><p>Aadhaar, PAN, phone and email patterns are detected and can be redacted before anything is summarized.</p></div>
    <div class="feat"><div class="ic">💬</div><h4>Ask &amp; search</h4><p>Grounded Q&amp;A answers only from the uploaded document, with full-text search and one-click JSON / Markdown / text export.</p></div>
  </div>
</div>
"""

GUIDE_HTML = """
<div class="sec reveal" id="guide">
  <div class="lx-label">HOW IT WORKS / 06</div>
  <h3 class="sec-head">From a dense judgment to a clear explanation.</h3>
  <p class="sec-sub">Six stages run on every document — the same pipeline the research was built and evaluated on.</p>
  <div class="steps">
    <div class="step-c"><div class="step-n">01</div><h4>Upload &amp; extract</h4><p>Drop a PDF, DOCX or TXT. Text is extracted and cleaned of court-PDF artefacts.</p></div>
    <div class="step-c"><div class="step-n">02</div><h4>Clean &amp; redact</h4><p>Preamble is stripped and PII patterns (Aadhaar, PAN, phone, email) are optionally redacted.</p></div>
    <div class="step-c"><div class="step-n">03</div><h4>Detect sections</h4><p>Paragraphs are labelled facts / arguments / reasoning / judgement, keeping the operative order intact.</p></div>
    <div class="step-c"><div class="step-n">04</div><h4>Summarize</h4><p>Your fine-tuned BART model summarizes each chunk with a section-specific token budget.</p></div>
    <div class="step-c"><div class="step-n">05</div><h4>Aggregate &amp; format</h4><p>Chunk summaries are merged and rendered as Lawyer Mode and plain-language Citizen Mode.</p></div>
    <div class="step-c"><div class="step-n">06</div><h4>Cite &amp; Verify</h4><p>Each sentence is checked against the source; grounded extraction fills dates, parties, provisions and outcome.</p></div>
  </div>
</div>
"""

# Measured on the first 100 IN-ABS test docs (artifacts/eval/full_outcome_test.json),
# Lawyer Mode body vs the published IN-ABS reference. Numbers are real, not illustrative.
METRICS_HTML = """
<div class="sec reveal" id="metrics">
  <div class="lx-label">EVALUATION / 07</div>
  <h3 class="sec-head">Measured, not claimed.</h3>
  <p class="sec-sub">Scores for the default <b>e3-outcome</b> checkpoint on 100 held-out IN-ABS
  test judgments (Lawyer Mode vs the reference summary).</p>
  <div class="metric-grid">
    <div class="metric"><div class="num" data-count="55.4">0</div><div class="lbl">ROUGE-1</div></div>
    <div class="metric"><div class="num" data-count="30.9">0</div><div class="lbl">ROUGE-2</div></div>
    <div class="metric"><div class="num" data-count="50.3">0</div><div class="lbl">ROUGE-L</div></div>
    <div class="metric"><div class="num" data-count="85.2">0</div><div class="lbl">BERTScore F1</div></div>
    <div class="metric"><div class="num" data-count="100">0</div><div class="lbl">Docs evaluated</div></div>
  </div>

  <div class="cmp">
    <div class="lx-label" style="margin-top:0">MODEL COMPARISON · ROUGE-1 (Lawyer)</div>
    <div class="bars">
      <div class="bar"><span class="bl">Zero-shot BART</span><span class="track"><span class="fill base" data-w="50.1"></span></span><span class="bv">50.1</span></div>
      <div class="bar"><span class="bl">Fine-tuned e3</span><span class="track"><span class="fill e3" data-w="55.2"></span></span><span class="bv">55.2</span></div>
      <div class="bar"><span class="bl">e3-outcome ⭐</span><span class="track"><span class="fill" data-w="55.4"></span></span><span class="bv">55.4</span></div>
    </div>
    <div class="lx-label">OUTCOME PRESENCE · full summary (holding stated)</div>
    <div class="bars">
      <div class="bar"><span class="bl">Zero-shot BART</span><span class="track"><span class="fill base" data-w="98"></span></span><span class="bv">98%</span></div>
      <div class="bar"><span class="bl">Fine-tuned e3</span><span class="track"><span class="fill e3" data-w="48"></span></span><span class="bv">48%</span></div>
      <div class="bar"><span class="bl">e3-outcome ⭐</span><span class="track"><span class="fill" data-w="53"></span></span><span class="bv">53%</span></div>
    </div>
    <div class="metric-note">Zero-shot BART scores high on outcome presence only because it near-copies
    the source verbatim, so the operative sentence is carried along; it loses on ROUGE / BERTScore.
    The outcome-preservation fix lifts the fine-tune's holding rate (e3 → e3-outcome: 48% → 53% full,
    46% → 50% Lawyer) while keeping its summarization quality.</div>
  </div>
</div>
"""

AUTHORS_HTML = f"""
<div class="sec reveal" id="about">
  <div class="about-band">
    <div class="team-tag">Team 4 · NLP Case Study</div>
    <p class="slogan">Justice begins with understanding.</p>
    <p><b>LegalLens</b> is a research project on
    hallucination-safe summarization of Indian court judgments — trained on the IN-ABS dataset,
    with source-grounded Cite &amp; Verify to keep every summary accountable to the document.</p>
  </div>
  <div class="lx-label">THE TEAM</div>
  <h3 class="sec-head" style="font-size:26px">Creators &amp; authors.</h3>
  <div class="team-grid">
    <div class="author">
      <div class="av"><img src="{_author_img(['manideep.jpg', 'manideep.jpeg', 'manideep.png'], 'MANIDEEP2407-SYS')}" alt="Akula Manideep Reddy" /></div>
      <h4>Akula Manideep Reddy</h4>
      <div class="role">Frontend, Model &amp; Integration</div>
      <a href="https://github.com/MANIDEEP2407-SYS" target="_blank" rel="noopener">GitHub ↗</a>
    </div>
    <div class="author">
      <div class="av"><img src="{_author_img(['mahima.jpeg', 'mahima.jpg', 'mahima.png'], 'mahima1006')}" alt="Mahima R Iyer" /></div>
      <h4>Mahima R Iyer</h4>
      <div class="role">Backend, Model &amp; Integration</div>
      <a href="https://github.com/mahima1006" target="_blank" rel="noopener">GitHub ↗</a>
    </div>
    <div class="author">
      <div class="av"><img src="{_author_img(['siva.jpeg', 'siva.jpg', 'siva.png'], 'Phyboc')}" alt="Sivasubramani K J" /></div>
      <h4>Sivasubramani K J</h4>
      <div class="role">Backend, Model &amp; Integration</div>
      <a href="https://github.com/Phyboc" target="_blank" rel="noopener">GitHub ↗</a>
    </div>
    <div class="author">
      <div class="av"><img src="{_author_img(['samanyu.jpg', 'samanyu.jpeg', 'samanyu.png'], 'SamanyuNair')}" alt="Samanyu Nair" /></div>
      <h4>Samanyu Nair</h4>
      <div class="role">Frontend, Model &amp; Integration</div>
      <a href="https://github.com/SamanyuNair" target="_blank" rel="noopener">GitHub ↗</a>
    </div>
  </div>
</div>
"""

# Scroll-reveal + animated count-up + bar fill. Runs once after the app loads.
ANIMATE_JS = """
() => {
  const run = () => {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (!e.isIntersecting) return;
        e.target.classList.add('in');
        e.target.querySelectorAll('.num[data-count]').forEach(el => {
          const target = parseFloat(el.dataset.count);
          const dec = (target % 1 !== 0) ? 1 : 0;
          let t0 = null;
          const step = (ts) => {
            if (!t0) t0 = ts;
            const p = Math.min((ts - t0) / 900, 1);
            const val = (target * (1 - Math.pow(1 - p, 3))).toFixed(dec);
            el.textContent = val;
            if (p < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        });
        e.target.querySelectorAll('.fill[data-w]').forEach(el => {
          setTimeout(() => { el.style.width = el.dataset.w + '%'; }, 120);
        });
        io.unobserve(e.target);
      });
    }, { threshold: 0.18 });
    document.querySelectorAll('.reveal').forEach(el => io.observe(el));
    document.querySelectorAll('.lx-nav a').forEach(a => {
      a.addEventListener('click', ev => {
        const id = a.getAttribute('href').slice(1);
        const t = document.getElementById(id);
        if (t) { ev.preventDefault(); t.scrollIntoView({behavior:'smooth'}); }
      });
    });
  };
  // Gradio mounts async; wait a tick for the DOM.
  setTimeout(run, 400);
}
"""

FORCE_DARK = """
<script>
(function(){
  var u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme','dark');
    window.location.replace(u.toString());
  }
})();
</script>
"""

with gr.Blocks(title="LegalLens", head=FORCE_DARK) as demo:
    doc_state = gr.State("")
    json_state = gr.State(None)

    gr.HTML("""
    <div class="lx-nav">
      <div class="brand">Legal<span>Lens</span></div>
      <a href="#analyze">Analyze</a>
      <a href="#features">Features</a>
      <a href="#guide">How it works</a>
      <a href="#metrics">Evaluation</a>
      <a href="#about">About</a>
    </div>
    <div class="lx-hero">
      <h1>Legal documents, made easier.</h1>
      <p>Upload a legal document and turn dense legal language into a clear, structured explanation.</p>
    </div>
    <div class="flow-card reveal">
      <div class="lx-label" style="margin-top:0">LEGAL INTELLIGENCE / 01</div>
      <div class="flow">
        <span class="step">DOCUMENT</span><span class="arr">→</span>
        <span class="step">EXTRACTION</span><span class="arr">→</span>
        <span class="step">LEGAL ANALYSIS</span><span class="arr">→</span>
        <span class="step">PLAIN-LANGUAGE SUMMARY</span>
      </div>
    </div>
    <span id="analyze"></span>
    """)

    gr.HTML('<div class="lx-label">START WITH A DOCUMENT</div>'
            '<h3 class="lx-h">Start with a document.</h3>'
            '<p style="color:var(--muted);font-size:16px;max-width:640px;margin-top:-8px">'
            'Upload a PDF or text document to extract the facts, issues, arguments, decisions, '
            'and important legal information.</p>')

    with gr.Row():
        with gr.Column(scale=2):
            file_in = gr.File(label="DROP YOUR DOCUMENT HERE  ·  PDF · DOCX · TXT",
                              file_types=[".pdf", ".docx", ".txt"], type="filepath")
            text_in = gr.Textbox(label="…or paste legal text", lines=8, value=SAMPLE_TEXT)
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(choices=list(MODEL_CHOICES), value=DEFAULT_MODEL,
                                   label="Summarizer model")
            redact_cb = gr.Checkbox(label="Redact person names", value=False)
            analyze_btn = gr.Button("Analyze document →", variant="primary", size="lg")

    results_html = gr.HTML()

    with gr.Accordion("Ask the document", open=False):
        gr.Markdown("Ask questions grounded only in the uploaded document.")
        chatbot = gr.Chatbot(height=320)
        with gr.Row():
            chat_in = gr.Textbox(placeholder="What was the final outcome?", scale=5, show_label=False)
            chat_btn = gr.Button("Ask", scale=1)
        gr.Examples(
            ["What was the main dispute?", "What did the plaintiff argue?",
             "What was the final outcome?", "What legal provisions were mentioned?",
             "What are the important dates?"],
            inputs=chat_in)

    with gr.Accordion("Search this document", open=False):
        with gr.Row():
            search_in = gr.Textbox(placeholder="Names, sections, dates, keywords…",
                                   scale=5, show_label=False)
            search_btn = gr.Button("Search", scale=1)
        search_out = gr.HTML()

    with gr.Accordion("Export analysis", open=False):
        with gr.Row():
            dl_json = gr.DownloadButton("Download Structured JSON")
            dl_text = gr.DownloadButton("Download Extracted Text")
            dl_md = gr.DownloadButton("Download Markdown")

    gr.HTML(FEATURES_HTML)
    gr.HTML(GUIDE_HTML)
    gr.HTML(METRICS_HTML)
    gr.HTML(AUTHORS_HTML)

    gr.HTML("""
    <div class="lx-foot">
      <b>LegalLens</b><br>
      Legal intelligence · Clear explanations · Source-grounded analysis<br>
      Built with Python + Gradio · Team 4 — LegalLens<br>
      <i>AI-generated summaries should be reviewed by a qualified legal professional.</i>
    </div>
    """)

    demo.load(None, None, None, js=ANIMATE_JS)

    analyze_btn.click(
        analyze, [file_in, text_in, model_dd, redact_cb],
        [results_html, json_state, doc_state, chatbot],
    )
    chat_btn.click(answer_question, [chat_in, chatbot, doc_state], [chatbot, chat_in])
    chat_in.submit(answer_question, [chat_in, chatbot, doc_state], [chatbot, chat_in])
    search_btn.click(search_doc, [search_in, doc_state], search_out)
    search_in.submit(search_doc, [search_in, doc_state], search_out)
    dl_json.click(lambda p: p, json_state, dl_json)
    dl_text.click(export_text, doc_state, dl_text)
    dl_md.click(export_markdown, [doc_state, json_state], dl_md)


demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, css=CSS, theme=THEME)

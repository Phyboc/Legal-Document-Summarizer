"""FastAPI endpoint for LegalLens."""
import io
import os
import sys
sys.path.insert(0, ".")
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.pipeline import process_document


def _extract_text(filename: str, content: bytes) -> tuple[str, str]:
    """Extract text from an uploaded file's bytes. Returns (text, error)."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", "PDF reading needs the `pypdf` package on the server."
        try:
            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            return "", f"Could not read that PDF: {exc}"
        return text, ""

    if ext == ".docx":
        try:
            import docx
        except ImportError:
            return "", "Word documents need the `python-docx` package on the server."
        try:
            document = docx.Document(io.BytesIO(content))
            text = "\n\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as exc:  # noqa: BLE001
            return "", f"Could not read that Word document: {exc}"
        return text, ""

    # .txt and anything else: decode as UTF-8, dropping undecodable bytes.
    return content.decode("utf-8", errors="ignore"), ""


def _build_response(result: dict, mode: str) -> dict:
    """Shape a process_document result for the API, honouring the mode filter."""
    response = {
        "n_paragraphs": result["n_paragraphs"],
        "n_chunks": result["n_chunks"],
        "sections_detected": result["sections_detected"],
        "is_finetuned": result["is_finetuned"],
    }
    if mode in ("lawyer", "both"):
        response["lawyer_mode"] = result["lawyer_mode"]
        response["lawyer_citations"] = result["lawyer_citations"]
        response["lawyer_verification"] = result["lawyer_verification"]
    if mode in ("citizen", "both"):
        response["citizen_mode"] = result["citizen_mode"]
        response["citizen_citations"] = result["citizen_citations"]
        response["citizen_verification"] = result["citizen_verification"]
    return response

app = FastAPI(title="LegalLens API", version="1.0")

# Allow the browser front ends (static page on Vercel / localhost, React dev
# server) to call the model API directly. CORS_ORIGINS can narrow this in
# production; the default is permissive for local development.
_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    text: str
    redact_names: bool = False
    strip_preamble: bool = True
    mode: str = "both"  # lawyer, citizen, both


class SummaryResponse(BaseModel):
    lawyer_mode: dict | None = None
    citizen_mode: dict | None = None
    lawyer_citations: list | None = None
    citizen_citations: list | None = None
    verification: dict | None = None
    n_paragraphs: int
    n_chunks: int
    sections_detected: list
    is_finetuned: bool


@app.get("/")
def root():
    return {"service": "LegalLens", "status": "up"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if not req.text or len(req.text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Input text too short")

    result = process_document(
        req.text,
        redact_names=req.redact_names,
        strip_preamble=req.strip_preamble,
    )

    return _build_response(result, req.mode)


@app.post("/summarize/upload")
async def summarize_upload(
    file: UploadFile = File(...),
    redact_names: bool = Form(False),
    strip_preamble: bool = Form(True),
    mode: str = Form("both"),
):
    content = await file.read()
    text, error = _extract_text(file.filename or "", content)
    if error:
        raise HTTPException(status_code=400, detail=error)
    if len(text.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="That file produced fewer than 100 characters of text. If it is a scanned PDF, paste the text instead.",
        )
    result = process_document(text, redact_names=redact_names, strip_preamble=strip_preamble)
    return _build_response(result, mode)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

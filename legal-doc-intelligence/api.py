"""FastAPI endpoint for LegalLens."""
import sys
sys.path.insert(0, ".")
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.pipeline import process_document

app = FastAPI(title="LegalLens API", version="1.0")


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

    response = {
        "n_paragraphs": result["n_paragraphs"],
        "n_chunks": result["n_chunks"],
        "sections_detected": result["sections_detected"],
        "is_finetuned": result["is_finetuned"],
    }

    if req.mode in ("lawyer", "both"):
        response["lawyer_mode"] = result["lawyer_mode"]
        response["lawyer_citations"] = result["lawyer_citations"]
        response["lawyer_verification"] = result["lawyer_verification"]

    if req.mode in ("citizen", "both"):
        response["citizen_mode"] = result["citizen_mode"]
        response["citizen_citations"] = result["citizen_citations"]
        response["citizen_verification"] = result["citizen_verification"]

    return response


@app.post("/summarize/upload")
async def summarize_upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    if len(text.strip()) < 100:
        raise HTTPException(status_code=400, detail="File content too short")
    result = process_document(text)
    return {
        "lawyer_mode": result["lawyer_mode"],
        "citizen_mode": result["citizen_mode"],
        "lawyer_verification": result["lawyer_verification"],
        "citizen_verification": result["citizen_verification"],
        "sections_detected": result["sections_detected"],
        "n_chunks": result["n_chunks"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

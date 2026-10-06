"""Document-level evaluation: score the product, not the chunk stage.

For each IN-ABS **test** document the full inference pipeline
(``src.pipeline.process_document``) is run with each model under comparison,
and the generated document summary is scored against the published IN-ABS
reference summary with ROUGE-1/2/L (and BERTScore with ``--bertscore``).

Every model is scored on the same documents in the same run, so the zero-shot
baseline and the fine-tuned model are directly comparable (guide §3.4).

Two outputs are scored per document:

    ``lawyer``  - Lawyer Mode body text (section titles stripped), i.e. what the
                  product shows, after its 700-word budget.
    ``full``    - every chunk summary concatenated and de-duplicated
                  (``aggregate_document``), i.e. the model's output before the
                  formatter's length budget.

Usage
-----
    python -m src.eval.evaluate --models facebook/bart-base models/legallens-bart-base \
        --sample 50 --output artifacts/eval/test50.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]
for _p in (_PKG_ROOT, _PKG_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.pipeline import process_document  # noqa: E402
from src.postprocess.aggregation import aggregate_document  # noqa: E402

logger = logging.getLogger("evaluate")


def lawyer_body(result: dict) -> str:
    return "\n".join(s["text"] for s in result["lawyer_mode"]["sections"])


def rouge(predictions: list[str], references: list[str]) -> list[dict]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    out = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        out.append({
            "rouge1": s["rouge1"].fmeasure,
            "rouge2": s["rouge2"].fmeasure,
            "rougeL": s["rougeLsum"].fmeasure,
        })
    return out


def bertscore(predictions: list[str], references: list[str]) -> list[float]:
    from bert_score import score

    # Long documents: the default roberta-large truncates at 512 tokens, which is
    # a known limitation of BERTScore on full judgement summaries - report it.
    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return [float(x) for x in f1]


def mean(values: list[float]) -> float:
    return round(100 * sum(values) / len(values), 2) if values else 0.0


def evaluate_model(model: str, docs: list[dict], use_bertscore: bool) -> dict:
    outputs = {"lawyer": [], "full": []}
    per_doc = []
    started = time.time()
    for i, doc in enumerate(docs, 1):
        result = process_document(doc["text"], model_name=model)
        lawyer = lawyer_body(result)
        full = aggregate_document(result["chunk_summaries"])
        outputs["lawyer"].append(lawyer)
        outputs["full"].append(full)
        per_doc.append({
            "doc_id": doc["doc_id"],
            "n_chunks": result["n_chunks"],
            "sections_detected": result["sections_detected"],
            "lawyer_words": len(lawyer.split()),
            "full_words": len(full.split()),
            "reference_words": len(doc["summary"].split()),
            "lawyer_summary": lawyer,
        })
        logger.info("[%s] %d/%d %s (%d chunks)", model, i, len(docs), doc["doc_id"], result["n_chunks"])

    references = [d["summary"] for d in docs]
    scores = {}
    for kind, preds in outputs.items():
        r = rouge(preds, references)
        for rec, s in zip(per_doc, r):
            rec[f"{kind}_rouge"] = {k: round(v, 4) for k, v in s.items()}
        scores[kind] = {metric: mean([s[metric] for s in r]) for metric in ("rouge1", "rouge2", "rougeL")}
        if use_bertscore:
            bs = bertscore(preds, references)
            for rec, b in zip(per_doc, bs):
                rec[f"{kind}_bertscore_f1"] = round(b, 4)
            scores[kind]["bertscore_f1"] = mean(bs)

    from src.model.summarizer import get_summarizer

    return {
        "model": model,
        "is_finetuned": get_summarizer(model).is_finetuned,
        "documents": len(docs),
        "seconds": round(time.time() - started, 1),
        "scores": scores,
        "mean_words": {
            "lawyer": round(sum(d["lawyer_words"] for d in per_doc) / len(per_doc), 1),
            "full": round(sum(d["full_words"] for d in per_doc) / len(per_doc), 1),
            "reference": round(sum(d["reference_words"] for d in per_doc) / len(per_doc), 1),
        },
        "per_document": per_doc,
    }


def load_test_docs(split: str, sample: int | None) -> list[dict]:
    from build_dataset import load_split, resolve_columns

    ds = load_split(split, limit=sample)
    text_col, summary_col, _ = resolve_columns(ds)
    # doc_id matches build_dataset / reference_summaries.jsonl: <split>_<position>.
    return [
        {"doc_id": f"{split}_{i:05d}", "text": row[text_col], "summary": row[summary_col]}
        for i, row in enumerate(ds)
        if row[text_col] and row[summary_col]
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Document-level evaluation against IN-ABS reference summaries")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model ids or local checkpoint paths; include facebook/bart-base for the zero-shot baseline")
    p.add_argument("--split", default="test", choices=("test", "validation"))
    p.add_argument("--sample", type=int, help="First N documents of the split (default: all)")
    p.add_argument("--bertscore", action="store_true", help="Also compute BERTScore (needs bert-score)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    docs = load_test_docs(args.split, args.sample)
    results = [evaluate_model(m, docs, args.bertscore) for m in args.models]

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "documents": len(docs),
        "note": "Document-level scores vs the published IN-ABS summary. "
                "'lawyer' = Lawyer Mode body (700-word budget); 'full' = all chunk summaries.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDocument-level results on {len(docs)} {args.split} documents")
    for r in results:
        for kind, s in r["scores"].items():
            print(f"  {r['model']:<40} {kind:<7} " + "  ".join(f"{k}={v}" for k, v in s.items()))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

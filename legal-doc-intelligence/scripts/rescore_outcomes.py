"""Score the outcome metrics of a saved evaluation run, without regenerating anything.

``src.eval.evaluate`` stores each document's Lawyer summary (and ROUGE/BERTScore),
but the outcome metrics were added after some runs were recorded. This re-reads the
stored Lawyer summaries, loads the same held-out references, and applies the current
``outcome_scores`` definition, so an older run can be compared on the same footing as
a new one. Nothing is generated and no model is loaded.

Usage (from legal-doc-intelligence/):

    python scripts/rescore_outcomes.py --eval artifacts/eval/test100_e3.json \\
        --label e3 --output reports/outcome_eval/e3_outcomes.json

The reference is matched by ``doc_id`` (``test_<position>``), so the stored run must
have been produced on the same split prefix (``test[:N]``).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import load_split, resolve_columns  # noqa: E402
from src.eval.evaluate import outcome_scores  # noqa: E402


def _mean(values: list[float]) -> float | None:
    return round(100 * sum(values) / len(values), 2) if values else None


def rescore(eval_path: Path, split: str, label: str) -> dict:
    run = json.loads(eval_path.read_text(encoding="utf-8"))["results"][0]
    per_doc = run["per_document"]
    n = len(per_doc)

    dataset = load_split(split, limit=n)
    _, summary_col, _ = resolve_columns(dataset)
    references = {}
    for index, row in enumerate(dataset):
        references[f"{split}_{index:05d}"] = row.get(summary_col) or ""
    # Plain IN-ABS ids are test_00000 style; match on the same scheme as the run.
    predictions, refs, ids = [], [], []
    for doc in per_doc:
        doc_id = doc["doc_id"]
        if doc_id not in references:
            continue
        predictions.append(doc["lawyer_summary"])
        refs.append(references[doc_id])
        ids.append(doc_id)

    scores = outcome_scores(predictions, refs)
    presence = [s["present"] for s in scores]
    recalls = [s["recall"] for s in scores if s["recall"] is not None]
    return {
        "label": label,
        "source_run": str(eval_path),
        "model": run.get("model"),
        "split": split,
        "documents_scored": len(ids),
        "scope": "lawyer",
        "outcome_present_pct": _mean(presence),
        "outcome_recall_pct": _mean(recalls),
        "documents_with_reference_disposition": len(recalls),
        "definition": (
            "present = Lawyer summary contains a sentence matched by section_detector.is_disposition; "
            "recall = mean token recall of the reference's disposition sentences (>=3-letter tokens), "
            "over documents whose reference has a disposition"
        ),
        "rescored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "per_document": [
            {"doc_id": d, "present": s["present"], "recall": s["recall"]} for d, s in zip(ids, scores)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    result = rescore(args.eval, args.split, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_document"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

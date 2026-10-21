Title: Preserve the court's operative order by routing it to the judgement section

Base branch note: this branch was cut from review2 (37eeeef). Target review2 as
the PR base (or confirm main is meant to absorb review2) so the diff is just the
three commits below, not the whole review2 lineage.

---

## Problem

LegalLens drops the court's operative order (the holding) — the single most
important sentence of a legal summary. ROUGE/BERTScore barely notice, because the
order is one or two sentences of a long summary.

## Root cause (confirmed by audit)

The order is **not** lost to filtering — the old 25-char short-sentence filter was
already gone, and 90% of operative-order sentences survive into the training
targets. The order survives but is attached to the **wrong section**: the
Cross-Encoder scores a restated order about as highly as the order itself, so
`align_summary_to_sections` routed it to a facts or reasoning chunk. On 100
validation docs, only 5 of 45 final-order sentences reached a `judgement` chunk.
A chunk-level model trained this way never learns the order belongs to the
`judgement` section the inference formatter protects. Two compounding causes:
`is_disposition` missed common forms (35 of 45 orders), and training used a
`full_document` fallback label inference never produced.

## Changes

- **section_detector**: broaden `is_disposition` (modal/adverb between subject and
  verb, "declared invalid", first-person past); add `is_lower_court_history` /
  `is_operative_order` to separate a recited lower-court order from this court's
  own; add `build_section_groups` as the single grouping rule shared with inference.
- **section_summary_alignment**: `align_summary_to_sections` takes `is_operative`;
  an operative sentence is labelled and routed to a `judgement` chunk only when a
  judgement chunk clears the threshold (source must support it) — no fabrication.
- **pipeline / build_dataset**: both use `build_section_groups`, so training and
  inference see identical section labels; `full_document` fallback retired.
- **Audit tooling**: `scripts/audit_outcome_survival.py` (stage trace),
  `scripts/rescore_outcomes.py` (offline outcome metrics for a saved run).

## Before / after (100 IN-ABS validation docs)

| | before | after |
|---|---:|---:|
| final orders reaching a judgement chunk | 5 / 45 | **13 / 45** |
| final orders labelled judgement | 8 | **18** |
| outcome sentences retained in a target | 62 / 69 | 64 / 69 |
| `is_disposition` misses | 53 | **33** |

Full table: `reports/outcome_preservation_before_after.csv`.

## Evaluation status

Measured (committed artifacts, test[:100], one consistent detector):

| Lawyer body | zero-shot | existing e3 | corrected |
|---|---:|---:|---:|
| ROUGE-1 / L | 50.9 / 44.7 | 55.2 / 49.6 | not run |
| outcome presence | 78% | 34% | not run |
| outcome recall | 34% | 16% | not run |

Zero-shot presence/recall exceed the fine-tune because the copier carries orders
verbatim while e3 compresses and drops them — the exact gap this fix targets. The
**corrected checkpoint is not trained** (CPU-only machine); reproducible
rebuild/train/eval commands are in `reports/model_evaluation.md`.

## Tests

`python -m pytest -q` → 88 passed, 1 skipped. 21 new offline regression tests in
`tests/test_outcome_preservation.py` cover short orders, modal/adverb forms,
negation, lower-court history, permission-not-order, OCR corruption, multi-matter,
chunk-boundary and budget-overflow cases, threshold-respecting routing, and
train/inference grouping parity. End-to-end CLI smoke test produced a correct
JUDGEMENT block.

## Known limitations / follow-up

- No corrected checkpoint or A/B/C generation eval yet (GPU required). Do not
  promote a model on this branch; the fix is validated at the data level only.
- Frontend is mock-only (`src/mockData.js`) and does not call the backend.
- 22 oblique relief phrasings still missed by `is_disposition` (left conservative
  rather than over-matched); 2 oblique orders remain unassigned.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

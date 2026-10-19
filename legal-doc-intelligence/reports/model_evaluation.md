# Model evaluation — LegalLens outcome-preservation work

**Measured on this machine:** CPU-only PyTorch (no CUDA available to Python), so
no model was trained and no new generation was run. The zero-shot and existing
fine-tuned columns are reproduced from committed evaluation artifacts
(`artifacts/eval/test100_e1.json`, `test100_e3.json`), scored on the first 100
IN-ABS **test** documents, full pipeline, vs the published IN-ABS summary. The
**corrected fine-tuned** column is `not run` — it requires a GPU rebuild + retrain.

Outcome presence/recall were recomputed offline from each run's stored Lawyer
summaries with the current (broadened) `is_disposition` detector, so all columns
use one definition (`scripts/rescore_outcomes.py`).

## Comparison table

| Metric | Zero-shot BART | Existing fine-tuned (e3) | Corrected fine-tuned | Status / interpretation |
|---|---:|---:|---:|---|
| ROUGE-1 (Lawyer) | 50.93 | 55.15 | not run | measured, committed artifacts |
| ROUGE-2 (Lawyer) | 24.96 | 30.75 | not run | measured |
| ROUGE-L (Lawyer) | 44.65 | 49.64 | not run | measured |
| BERTScore F1 (Lawyer) | 83.51 | 85.04 | not run | measured |
| ROUGE-1 (full) | 52.86 | 58.80 | not run | measured |
| ROUGE-2 (full) | 30.05 | 34.04 | not run | measured |
| ROUGE-L (full) | 45.11 | 51.07 | not run | measured |
| BERTScore F1 (full) | 83.67 | 85.13 | not run | measured |
| Outcome presence — Lawyer | **78.0%** | **34.0%** | not run | measured; see note A |
| Outcome recall — Lawyer | 34.15% | 15.71% | not run | measured; mean over 31 docs with a reference disposition |
| Outcome presence — full | not run | not run | not run | needs re-generation (full-chunk body not stored) |
| Outcome agreement | not run | not run | not run | detector has no sign/agreement head yet |
| NLI entailment rate | not run | not run | not run | needs generation + NLI pass |
| Mean NLI — true/unrelated/reversed | not run | not run | not run | calibration exists (`artifacts/citation_calibration.json`); per-run not generated |
| Mean BERTScore between perturbed runs | not run | not run | not run | perturbation harness not run |
| Pairs with unchanged outcome | not run | not run | not run | — |
| Toxicity flags after review | not run | not run | not run | — |
| PII matches after review | 0 patterns on val100 build | 0 | not run | pattern PII only; build_stats pii_pattern_counts all 0 |
| Outputs within length limits — Lawyer | not run | ~pass (mean 557.9 w, budget 500–700) | not run | aggregate words within budget; per-doc compliance not tabulated |
| Outputs within length limits — Citizen | not run | not run | not run | — |
| Novel-bigram rate | not run | ~0.16 (copy ratio 0.84, per forensic report) | not run | from data audit, not a per-run eval |

**Note A — the central result.** With one consistent detector, zero-shot bart-base
has *higher* outcome presence (78% vs 34%) and recall (34% vs 16%) on the Lawyer
body than the existing fine-tuned model. This is not evidence that zero-shot is
better: it near-copies large spans of the input, so an operative sentence is
often carried verbatim, while the fine-tuned model compresses and — because its
training targets filed the order under facts/reasoning rather than `judgement` —
drops it under the formatter's budget. ROUGE/BERTScore reward the fine-tune;
outcome presence exposes the product gap. This is exactly what the data fix
(routing orders to `judgement`) is designed to correct, and it is the hypothesis a
corrected retrain must test.

## What the fix changed, and what it did not

- **Validated at the data level** (`reports/outcome_preservation_before_after.csv`):
  final-order sentences reaching a `judgement` chunk 5 → 13 of 45; labelled
  `judgement` 8 → 18; detector misses 53 → 33. 88 tests pass (21 new).
- **Not yet validated at the model level.** No corrected checkpoint exists, so the
  claim "the corrected model states the order more often" is a hypothesis, not a
  measured result. Do not promote any model on the strength of this report.

## Reproduce the blocked columns (GPU)

```bash
cd legal-doc-intelligence
# 1. Rebuild the full dataset with the fix (hours on CPU; faster on GPU)
python scripts/build_dataset.py --full --output_dir artifacts/training_data/full_v2
# 2. Train the corrected checkpoint (do NOT overwrite e3)
python -m src.model.train --data artifacts/training_data/full_v2 \
  --base_model facebook/bart-base --output_dir models/legallens-bart-base-outcome-v2 \
  --epochs 3 --lr 3e-5 --batch_size 4 --grad_accum 8 --max_target_length 448 --seed 42
# 3. One-run A/B/C on the SAME 100 test docs, both base and both fine-tunes
LEGAL_NLI_MODEL=models/nli-deberta-v3-xsmall python -m src.eval.evaluate \
  --models facebook/bart-base models/legallens-bart-base-e3 models/legallens-bart-base-outcome-v2 \
  --split test --sample 100 --bertscore --output artifacts/eval/abc_v2.json
# 4. Rescore outcomes for each
python scripts/rescore_outcomes.py --eval artifacts/eval/abc_v2.json --label v2 \
  --output reports/outcome_eval/v2_outcomes.json
```

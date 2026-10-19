# Release checklist — outcome-preservation work

Status legend: [x] done/verified · [~] partial · [ ] blocked/not done

## Data and code
- [x] Root cause confirmed by audit (relevance-only routing; detector under-coverage; history conflation; grouping drift). `reports/outcome_data_audit.md`
- [x] Short-sentence filter confirmed already absent (ruled out).
- [x] Operative orders routed to judgement when the source supports it; no fabrication when it does not.
- [x] Train and inference share one grouping rule (`build_section_groups`).
- [x] Regression tests added and passing (88 passed, 1 skipped; 21 new in `tests/test_outcome_preservation.py`).
- [x] Before/after survival measured on val100 (`reports/outcome_preservation_before_after.csv`).

## Dataset rebuild and training (GPU)
- [ ] Full dataset rebuilt with the fix (`--full`). Blocked: CPU-only; hours-to-days here.
- [ ] Corrected checkpoint `legallens-bart-base-outcome-v2` trained. Blocked: no GPU torch.
- [ ] Existing e3 checkpoint preserved (do not overwrite). Policy stated; nothing overwritten.

## Evaluation (GPU)
- [x] Zero-shot and existing e3 ROUGE/BERTScore reproduced from committed artifacts.
- [x] Outcome presence/recall (Lawyer) rescored offline under one detector for zero-shot and e3.
- [ ] Corrected model A/B/C on the same 100 test docs. Blocked.
- [ ] Outcome agreement, per-run NLI, perturbation stability, toxicity, Citizen length compliance. Not run.

## Release gates (cannot be evaluated until a corrected model exists)
- [ ] Outcome presence and recall improve materially over e3.
- [ ] Outcome agreement does not regress.
- [ ] No fabricated outcomes in the reviewed regression suite. (Tests assert no fabrication in the routing logic.)
- [ ] Factual support / contradiction rates do not materially regress.
- [~] Both modes pass length/formatting tests (aggregate Lawyer words in budget; per-doc + Citizen not tabulated).
- [x] PII / evidence-verification unit tests pass (existing suite green).
- [x] No critical runtime failures (end-to-end CLI smoke test produced a correct JUDGEMENT block).

## Integration
- [x] Backend defaults to the fine-tuned checkpoint (config.BART_MODEL), not base.
- [x] Backend runs end-to-end with the fixed pipeline (smoke test).
- [ ] Frontend wired to the backend. Blocked / not done: the React UI is mock-only (`src/mockData.js`); it never calls `/summarize`. Real wiring + result-pane redesign is a separate task.

## Verdict
Do NOT promote any model to production on this branch. The fix is validated at the
data level only. A corrected retrain and the A/B/C evaluation are required before
the release gates can be judged.

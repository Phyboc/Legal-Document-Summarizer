LegalLens — Forensic Audit Report

Read-only investigation. No files were modified, nothing was installed, retrained, or uploaded. All counts below were computed directly from the on-disk artifacts and the dataset JSONL; model weights were hashed but never loaded or run (I did not launch new generation — Phase 7 relies on the committed eval artifacts, per the “no expensive jobs without permission” constraint).

1. Executive verdict

Conditionally ready — as the chunk-level summarizer inside the LegalLens pipeline, with two fixes first. Not ready as a drop-in document summarizer.

The fine-tune is real and the headline claims on the model card reconcile to the artifacts: the published checkpoint is genuinely the epoch-3 best checkpoint, it beats facebook/bart-base zero-shot on every document-level metric the project reports, and the data/training pipeline is unusually well-instrumented and honest about its own weaknesses. There is no evidence of the classic catastrophic bugs (swapped inputs/targets, empty labels, silent truncation, accidental base-model publication, wrong checkpoint).

But the user’s symptom — “reproduces the opening of the judgment, omits later content, stops mid-sentence at the limit” — is confirmed and reproduced from a committed artifact, and it is not a training failure. It is the product of three independent, individually-fixable causes:

The pseudo-targets are ~84% extractive (mean lexical copy ratio 0.84, median 0.87). The model correctly learned the task it was given: copy. (Finding D1)
Per-chunk generation is capped at 100–250 tokens with forced_eos_token_id=2, so when the extractive output reaches the cap it is cut mid-sentence and an EOS is forced. (Finding I1)
facebook/bart-base is a denoising autoencoder, not a summarizer — zero-shot it near-copies the input — and the CLI/API/Gradio all default to it unless LEGAL_BART_MODEL is set. A test run that didn’t set that variable was testing the base model. (Finding I2)

Smallest sensible next step is not retraining: raise/repair the generation configuration and make the entry points load the fine-tuned checkpoint. Retraining with less-extractive targets is a larger, optional, second-order improvement.

2. Verified pipeline

Source → published checkpoint, every stage grounded in code that was demonstrably executed (artifacts exist for each):

percins/IN-ABS (official train/validation/test splits)
  → build_dataset.py  (reuses the inference preprocessing: preprocess → PII redact
      → split_paragraphs → detect_sections → chunk_document)                  scripts/build_dataset.py:309-461
  → per-chunk Cross-Encoder alignment of each reference-summary sentence       src/data/section_summary_alignment.py:326-439
      (cross-encoder/ms-marco-MiniLM-L-6-v2, sigmoid-normalized,               src/shared/cross_encoder.py:26-46
       threshold 0.40 + relative margin 0.90)                                  config.py:15-16
  → aligned_target = summary sentences whose primary chunk is this chunk,      src/data/section_summary_alignment.py:442-466
      joined in reference-summary order, greedy to 448 tokens
  → artifacts/training_data/full/{train,validation,test}.jsonl                 (31029 / 4198 / 6133 records on disk)
  → src/model/train.py  (BART-base, Seq2SeqTrainer, 3 epochs, best=rougeL)     src/model/train.py:219-287
  → models/legallens-bart-base-e3/  (== checkpoint-2910, verified by SHA256)
  → model card README → fakechamp24/legallens-bart-base                        models/legallens-bart-base-e3/README.md
Inference: src/pipeline.py → summarizer per chunk → aggregate → Lawyer/Citizen → Cite&Verify

Published-checkpoint identity, proven: models/legallens-bart-base-e3/model.safetensors is byte-identical (SHA256 0f17f55d…04b4) to checkpoint-2910/model.safetensors, and the trainer state records best_global_step: 2910, best_metric: 34.72. So the top-level saved model is the epoch-3 best checkpoint — “Best checkpoint = last epoch” is true. The one unverifiable link is the last arrow: no model-upload step exists in the repo (the notebook trains/evaluates e3 but never pushes it), so I cannot confirm from here that the HF repo’s weights equal local e3. See §10.

3. Dataset audit

Counts reconcile exactly with the model card. Computed from the JSONL on disk vs artifacts/training_data/full/build_stats.json:

Claim (card)	Artifact	Match
31,029 train / 4,198 validation records	wc -l train/val = 31029 / 4198	✓
5,346 / 712 judgements	build_stats documents_per_split	✓
17.9% sentences unassigned	unassigned_rate 0.1787	✓
Arguments thin, 4.6%	(1297+601)/41360 = 4.59%	✓
max input 1024 / target 448	length_validation max 1023 / 448, 0 violations	✓

Quality findings (31,029 train records inspected programmatically + 30 stratified samples read):

Targets are highly extractive — the central dataset problem. Lexical copy ratio over a 2,000-record sample: mean 0.84, median 0.87, p90 0.97; 39% of targets copy ≥90% of their words from the source chunk, only 3% are substantially abstractive (≤0.5). This is inherent to the method: IN-ABS headnotes are themselves near-extractive, and the Cross-Encoder matches a summary sentence to the chunk it is lexically closest to, so the target is almost a copy. (src/data/section_summary_alignment.py:271-282)
Targets are clean and complete. 0 empty targets; only 144/31,029 (0.5%) don’t end in terminal punctuation — build_aligned_target only ever stops at whole-sentence boundaries (section_summary_alignment.py:442-466), so the training targets are not themselves truncated mid-sentence. The mid-sentence symptom is a generation artifact, not a data artifact.
Near-duplicate targets: benign. 79 identical (chunk_text, aligned_target) pairs and 109 targets occurring >1×; spot-checks show these are short operative orders (“The appeal is dismissed.”) legitimately recurring across judgments.
Alignment scores are saturated (build_stats: mean 0.9347, median 0.9891, p90 0.9992). The sigmoid-normalized MS-MARCO relevance score sits near 1.0 for almost every on-topic pair, so the 0.40 absolute threshold does little work; the relative margin (0.90) is what actually prevents every sentence being assigned to every section. Multi-label rate 21.5%, 1.07 labels/sentence — reasonable.

Representative examples (document IDs real):

GOOD_TARGET — train_00406 chunk 5 (judgement): chunk “There will, therefore, be judgment in terms of the award… interest on judgment at 9%…” → target restates exactly that, complete, copy 0.92. A valid extractive operative-order target.
TOO_EXTRACTIVE — train_01581 chunk 0 (facts): target is two fact sentences lifted almost verbatim (copy 0.83); typical of the bulk of the set.
MISMATCHED_TARGET (likely) — train_04718 chunk 8 (judgement): chunk is a dismissal order (“We accordingly dismiss the appeal… no orders as to costs”) but the target is an unrelated sentence about “the effect of clause 13 A of the Order…” ending in a stray ). The alignment attached the wrong summary sentence to an operative-order chunk. (Confidence: Possible — the chunk preview is truncated, so I can’t fully exclude a later-in-chunk match.)
UNSUPPORTED/noisy — train_00508 chunk 19 (arguments_appellant): target is a raw citation list (“Union of India vs Bhim Singh… [1953] S.C.R. 486”). Training on these teaches the model to emit citation strings, which the card itself says it garbles.
4. Split and leakage audit
Official IN-ABS splits preserved; partitioned by document before chunking. The builder asserts document-id disjointness every run and it holds: doc_id_overlaps: {} (build_dataset.py:522-547). Document IDs are <split>_<index>, so a chunk cannot cross a split boundary by construction. No builder-introduced leakage.
10 inherited content duplicates — real, disclosed, not hidden. split_isolation.cross_split_content_duplicates: 10 — identical source judgments appearing in two official splits: 4 train↔validation, 6 train↔test (e.g. train_00353↔test_00307). This is leakage inside percins/IN-ABS, surfaced by SHA-1 content hashing (build_dataset.py:548-571), not created here. Impact on the reported test metrics is small (≤6 of 100 test docs partly seen in training) but non-zero, so the document-level numbers are very slightly optimistic.
No test references touch training or model selection. Checkpoint selection uses chunk-level ROUGE on the validation pseudo-targets only (train.py:235-236); the test split is used solely by src/eval/evaluate.py. Held-out integrity is sound apart from the 10 inherited dupes.
5. Training audit (actual vs model card)

Every card claim checks out against training_meta.json / trainer_state.json:

Card	Actual	
3 epochs, lr 3e-5, eff. batch 32 (4×8)	epochs 3.0, lr 3e-05, batch 4, grad_accum 8	✓
warmup 6%, wd 0.01, label smoothing 0.1, bf16	identical	✓
max input 1024 / target 448	identical; length_report shows 0 inputs/targets truncated	✓
Best = last epoch	rougeL 33.2 → 34.56 → 34.72; best=step 2910	✓
Chunk-val ROUGE 38.74/22.37/34.72	eval_metrics_chunk_level identical	✓
Label masking / loss: DataCollatorForSeq2Seq + text_target tokenization (train.py:80-86, 250) — correct -100 masking, no manual error. Tokenizer == base BART, saved alongside weights.
Early stopping (patience 2) configured but never fired (loss improved each epoch).
One real anomaly — logged training loss ≈ 25, eval cross-entropy ≈ 2.96 (trainer_state.json: step losses 36→24, train_loss 25.33; eval_loss 2.96). An ~8× scale gap. Evidence that it is a reporting artifact of label smoothing under transformers 5.8.0 (the version stamped in the configs), not a broken run: step loss decreases monotonically, eval_loss falls each epoch (3.04→2.98→2.96), chunk ROUGE rises, and the model beats zero-shot downstream. Flag it, don’t panic about it. (Finding T1)
Checkpoint parity confirmed (SHA256, §2). No accidental base-model publication locally.
6. Inference audit — the mid-sentence truncation, explained

Entry points are consistent but all default to the base model. cli.py:12, api.py:45, gradio_app.py:33, and src/eval/evaluate.py:114 all call process_document(...). Only evaluate.py passes model_name; CLI/API/Gradio pass none, so get_summarizer() resolves BART_MODEL = os.getenv("LEGAL_BART_MODEL", "facebook/bart-base") (config.py:5, summarizer.py:19-26). Without LEGAL_BART_MODEL set, the product runs zero-shot bart-base. This is almost certainly why “the base model also performed poorly on the same example” — the default path is the base model.

The 606-token input → ~151 words → mid-sentence stop is confirmed from artifacts/baselines/zero_shot.json. That file’s generated fields show bart-base emitting a near-verbatim copy of each chunk’s opening and stopping mid-clause: e.g. doc test_00000 chunk 0 — chunk 664 words, generated 206 words ending “…had not been”; test_00002 — 617-word chunk, 171 words ending “…on the advice of the Secretary of”. The mechanism, fully grounded:

summarize_chunk passes max_length = the section profile cap — facts/default 200, reasoning 250, judgement 100 tokens — with num_beams=4, no_repeat_ngram_size=3, early_stopping=True (summarizer.py:7-14, 52-62).
The model (extractive) copies the chunk’s leading sentences. It reaches max_length before emitting a natural EOS.
forced_eos_token_id=2 (config.json, generation_config.json) forces an EOS at the final position → the output ends mid-sentence at the cap. ~151–217 words ≈ 190–210 tokens ≈ the 200-token facts/default cap. ✔ matches the report exactly.

Why max_new_tokens “didn’t help”: the code never uses max_new_tokens — it uses max_length per profile (summarizer.py:58). Separately, the saved generation_config.json has max_length: 20 (the bart-base pretraining default) for both e1 and e3. Any caller relying on defaults, or setting max_new_tokens alongside that max_length, hits a conflict (HF resolves/ warns and can clamp). (Finding I3)

“Omits later content” has two paths: (a) at document level each chunk is summarized independently but the extractive model reproduces only each chunk’s opening; (b) if the whole judgment is fed to the model directly (model-card snippet, not the pipeline), input is truncated at 1024 tokens (summarizer.py:53) so everything after ~1024 tokens is invisible — the model is a chunk-level summarizer being asked to do document-level work.

Postprocessing is sound and not a hidden cause: dedupe_sentences (abbreviation-aware) removes overlap dupes; allocate_budget protects the judgement section first and trims others (mode_formatter.py:54-88) — a genuine improvement over front-trimming. Cite & Verify caveat: copied/verbatim sentences short-circuit to entailment=1.0, "supported" (cite_verify.py:102-123, 200-202). So a high verified_rate partly reflects the extractive copying rather than independent correctness — don’t read it as a quality guarantee.

7. Evaluation audit (from committed artifacts; no new runs)

The project’s committed A/B/C/D comparison on the first 100 held-out IN-ABS test judgments, full pipeline, scored vs the published summary:

Model	Output	R1	R2	RL	BERTScore	mean words
bart-base zero-shot	Lawyer	50.93	24.96	44.65	83.51	591.7
e3 (this model)	Lawyer	55.15	30.75	49.64	85.04	557.9
bart-base zero-shot	all chunks	52.86	30.05	45.11	83.67	1361.2
e3	all chunks	58.80	34.04	51.07	85.13	756.3
e1 (1 epoch)	Lawyer	53.86	29.48	48.03	84.92	543.2
Card numbers reconcile. Zero-shot column = artifacts/eval/test100_e1.json (which scored bart-base + e1 together on the same 100 docs); “this model” = test100_e3.json. Both use test[:100] and deterministic beam decode → comparable.
Reproducible result: fine-tuning yields a real, consistent gain (+4–6 ROUGE-1, +1.5 BERTScore) and roughly halves the raw output verbosity (zero-shot “all chunks” 1361 words → e3 756, nearer the 827-word reference).
Methodology caveats to record, not silently accept:
The e3 run (test100_e3.json) contains only e3 — the zero-shot baseline was not re-run in the same process, contrary to evaluate.py’s docstring (“same run”). Practically comparable, but the card pairs numbers from two runs. (Finding E1)
ROUGE/BERTScore reward lexical overlap, which rewards the very copying that is the product problem — high ROUGE here is partly an extractiveness score. The repo’s own outcome_scores (does the summary carry the court’s order?) is the better signal and should be surfaced on the card.
The 10 inherited cross-split dupes slightly inflate test scores (§4).
I did not launch a fresh generation eval (loads a 558 MB checkpoint + downloads IN-ABS test split = an expensive job needing your go-ahead). To reproduce A/B/C/D yourself:
LEGAL_NLI_MODEL=models/nli-deberta-v3-xsmall python -m src.eval.evaluate \
  --models facebook/bart-base models/legallens-bart-base-e3 \
  --split test --sample 10 --bertscore --output artifacts/eval/reproduce10.json
8. Ranked findings
ID	Severity	Confidence	Finding
I1	Critical	Confirmed	Mid-sentence truncation
D1	High	Confirmed	Targets ~84% extractive → model copies
I2	High	Confirmed	Entry points default to base model
I3	Medium	Confirmed	Saved generation_config.json max_length:20
E1	Medium	Strong	Zero-shot baseline from a different eval run
D2	Medium	Strong	Inherited 10 cross-split dupes in test metrics
D3	Medium	Possible	Occasional mismatched/citation-list targets
T1	Low	Strong	Logged train loss ≈8× eval loss (reporting artifact)
P1	High	Confirmed	Chunk-level model ≠ document summarizer (product gap)

I1 — Generation hits max_length and forced_eos cuts mid-sentence. Evidence: summarizer.py:7-14,55-62; config.json/generation_config.json forced_eos_token_id:2; zero_shot.json outputs ending mid-clause at ~200 tokens. Impact: every chunk near its cap ends in a broken sentence. Fix: raise profile max_length (facts/default→~320, reasoning→~400, judgement→~160), prefer max_new_tokens, and set length_penalty≈2.0 + min_length tuned; consider dropping forced_eos or raising caps so EOS is reached naturally. Code only, no retrain. Verify: regenerate the 606-token example; output ends on terminal punctuation under the cap.

D1 — Pseudo-targets are copies. Evidence: copy ratio mean 0.84/median 0.87/p90 0.97 (31k records); saturated alignment scores; section_summary_alignment.py:271-282. Impact: the model’s extractiveness is learned-correct behavior, not a bug — abstractiveness cannot be generated at inference if it was never supervised. Fix (optional, larger): paraphrase/compress targets, or filter out ≥0.95-copy targets, or add an abstractive reference source; this is the only change that needs a dataset rebuild + retrain. Verify: rebuild a 100-doc slice, confirm median copy ratio drops, retrain-smoke, inspect outputs.

I2 — CLI/API/Gradio run bart-base unless LEGAL_BART_MODEL is set. Evidence: config.py:5, summarizer.py:19-26, cli.py:12, api.py:45, gradio_app.py:33. Impact: the default product is the near-identity copier; “base model also poor” is this. Fix: default BART_MODEL to the fine-tuned repo/local path, or make entry points fail loudly when the base model is loaded. Code only. Verify: cli.py summarize with no env var reports is_finetuned: true.

I3 — generation_config.json max_length:20. Evidence: both e1/e3 configs. Impact: latent — any .generate() without explicit max_length yields 20-token output; conflicts with a caller’s max_new_tokens. Fix: write a sane generation_config.json (max_length≈400, min_length, length_penalty, num_beams=4) into the published model. Code/artifact only. Verify: bare model.generate(**inputs) produces a full summary.

E1 — baseline from a separate run. Fix: re-run both models in one evaluate.py invocation for the card. Verify: single JSON contains both models. D2 — disclose the 10 dupes on the card (already in build_stats). D3 — audit alignment of judgement/arguments targets, consider a per-target copy-ratio / length-sanity filter. T1 — document the loss-reporting quirk; no action needed.

P1 — architecture vs product requirement. Lawyer Mode (500–700 w) and Citizen Mode (150–250 w) are assembled in the pipeline, not produced by the model; the model is strictly chunk-level and extractive. It satisfies “section-aware chunk summaries aggregated into modes,” not “faithful abstractive document summary.” Integrate it as the former with eyes open, or treat abstractive quality as a separate roadmap item.

9. Action plan (smallest-first; retraining is not step 1)
Fix generation (I1, I3) — raise per-section max_length/switch to max_new_tokens, add length_penalty, repair the saved generation_config.json. Verify: the 606-token example ends cleanly; no mid-sentence cut. (hours, code only)
Make the fine-tuned model the default (I2) — change config.BART_MODEL default or hard-fail on base. Verify: is_finetuned: true with no env var.
Tidy evaluation/card (E1, D2, T1) — one-run A/B, disclose the 10 inherited dupes and the loss-reporting quirk, surface outcome_recall next to ROUGE.
Verify the published checkpoint (§10) — confirm HF weights == local e3; add a scripted, logged upload.
Only then, if abstractiveness is required (D1, P1) — rebuild targets to be less extractive and retrain. Measure against step 1’s improved baseline so you don’t retrain to fix what was a generation cap.

Do not re-report test-set numbers that were used to pick any of these changes as unbiased test results; hold out a fresh slice if you tune on test.

10. Missing evidence
Published-model identity. No model-upload step in the repo (notebook trains/evaluates e3 but never pushes; build_dataset.py only pushes the dataset). Cannot byte-verify fakechamp24/legallens-bart-base == local e3. Resolve (read-only):
python -c "from huggingface_hub import hf_hub_download; import hashlib; p=hf_hub_download('fakechamp24/legallens-bart-base','model.safetensors'); print(hashlib.sha256(open(p,'rb').read()).hexdigest())"
# expect 0f17f55d2366b0eb057359da685f08679d329e648a1c8a1845185d3875ec04b4
The 606-token example itself. I reproduced the mechanism from zero_shot.json; the exact input the user saw isn’t committed. Drop it in and run the evaluate.py command in §7 to capture per-chunk generations.
Full training stdout (the tokenize_splits truncation warnings, backend line). training_meta.json captures the summary; raw logs would confirm the alignment backend was CrossEncoder at train time (build_stats confirms it at build time: backend: "CrossEncoder", not the mock stub).
Live token-level Phase 3 counts. I used build_stats real-tokenizer validation (max target 448, 0 truncated) + word-level copy ratios rather than loading the BART tokenizer over 31k records; load the tokenizer if you want exact per-record token ratios.
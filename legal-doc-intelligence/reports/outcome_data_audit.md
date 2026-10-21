# Outcome-preservation data audit

**Scope.** 100 IN-ABS validation documents, built with the section-aligned
pipeline (CPU Cross-Encoder), at `HEAD` (37eeeef) and after the fix (4daca2c).
Reproduce with `scripts/audit_outcome_survival.py`. All counts are heuristic and
provisional until a person reviews the sample (`reports/outcome_audit/*/outcome_sentences.csv`).

## What the question actually was

The symptom under investigation is "LegalLens drops the court's operative order
(the holding)". The forensic report (`../../LegalLens — Forensic Audit Report.MD`)
had already established, and this audit confirms, that the *earlier* suspected
cause — a 25-character short-sentence filter — **no longer exists**. `git log -S
_MIN_SENTENCE_CHARS` shows it was removed before the current head; the splitter
now keeps every non-empty sentence, so "The appeal is dismissed." survives
segmentation. The real losses are elsewhere.

## Where outcomes actually go (baseline, HEAD)

Outcome sentences are found by audit-side cues (`scripts/audit_outcome_survival.py:OUTCOME_CUES`),
deliberately independent of the training-side detector `is_disposition`, so the
audit cannot grade the detector with the detector's own rules. Sentences naming a
lower court in the past tense ("the High Court dismissed the petition") are
flagged as history, not final orders.

| Stage | Count | Share |
|---|---:|---:|
| segmented and retained in a training target | 62 | 89.9% |
| unassigned (no section cleared the threshold) | 5 | 7.2% |
| lost to the 448-token target budget | 2 | 2.9% |

So **survival into the targets is not the problem** — 90% of outcome sentences
are kept. The problem is *where* they are kept.

Of the **45 final-order sentences** (lower-court history excluded):

| | baseline | after fix |
|---|---:|---:|
| labelled `judgement` | 8 | **18** |
| primary chunk is a `judgement` chunk | 5 | **13** |
| retained in some target | 42 | 43 |
| missed by `is_disposition` | 35 | **22** |

The order survives but is attached to a **facts or reasoning chunk** two times
out of three, because the Cross-Encoder scores a *restated* order (e.g. a headnote
sentence that echoes the facts) about as highly as the order itself. A chunk-level
model trained on this never learns that the operative order belongs to the
`judgement` section — which is the section the inference formatter protects first.

## Confirmed root causes

1. **Relevance-only routing (primary).** `align_summary_to_sections` attached each
   sentence to its single highest-scoring chunk. For an operative order that chunk
   is usually facts/reasoning, so the order is trained into the wrong section.
   *Fixed:* operative orders are now routed to a `judgement` chunk when the source
   supports one (a judgement chunk above threshold); otherwise unchanged.

2. **Detector under-coverage.** `is_disposition` recognised only adjacent
   subject-verb forms ("the appeal is allowed") and missed "must accordingly be
   allowed", "is accordingly set aside", "declared invalid", first-person past
   ("we dismissed"). 35 of 45 final orders were invisible to it. *Fixed:* broadened
   patterns; misses 35 → 22 (the remainder are genuinely oblique, e.g. "entitled to
   a refund", left for the conservative path rather than over-matched).

3. **Lower-court history conflation.** A recited lower-court order matches the same
   verbs as a final order. *Fixed:* `is_lower_court_history` / `is_operative_order`
   separate them, and history is not routed to `judgement`.

4. **Train/inference grouping drift.** The build used a `full_document` fallback
   label the inference pipeline never produced. *Fixed:* both call the shared
   `section_detector.build_section_groups`.

## Not a cause (ruled out)

- **Short-sentence filter** — removed before HEAD; verified absent.
- **Target truncation** — only 1–2 of 69 outcome sentences hit the 448-token
  budget, and `build_aligned_target` stops on sentence boundaries and reports
  overflow rather than cutting mid-order.
- **Missing judgement records** — 99/100 (now 100/100) documents have a judgement
  section detected. The ~half-of-documents "no judgement record" figure in the
  build stats is `judgement_outcome == no_judgement_sentence`: the reference
  summary simply states no disposition the aligner could place, not a missing field.

## Remaining failures in the sample

- 2 unassigned final orders (`validation_00000`, `validation_00034`) are oblique
  ("This order was confirmed by the higher income tax authorities") and below the
  alignment threshold — correctly conservative, not fabricated.
- 22 final orders still missed by `is_disposition` are oblique relief phrasings;
  the audit records them so the gap is visible rather than hidden.

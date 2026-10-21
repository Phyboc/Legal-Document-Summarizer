# Outcome survival audit: baseline (HEAD 37eeeef)

- Documents audited: 100 (0 failed)
- Documents with a judgement section detected: 99
- Documents with a reference outcome (audit cues): 40 (40.0% of documents)
- Outcome documents with >=1 outcome retained in a training target: 36 (90.0%)
- Outcome sentences: 69

| Stage | Count | Share |
|---|---:|---:|
| retained | 62 | 89.86% |
| unassigned | 5 | 7.25% |
| budget_overflow | 2 | 2.9% |

- Training-detector (`is_disposition`) misses among outcome sentences: 53
- Outcome sentences with lower-court context (ambiguous, flagged): 24

Heuristic counts. Stage labels come from exact sentence matching against the
alignment record; outcome sentences come from audit-side cues. Provisional until
a person has reviewed a sample.

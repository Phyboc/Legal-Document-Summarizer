# Outcome survival audit: after fix (operative routing + shared grouping + detector coverage)

- Documents audited: 100 (0 failed)
- Documents with a judgement section detected: 100
- Documents with a reference outcome (audit cues): 40 (40.0% of documents)
- Outcome documents with >=1 outcome retained in a training target: 36 (90.0%)
- Outcome sentences: 69

| Stage | Count | Share |
|---|---:|---:|
| retained | 64 | 92.75% |
| unassigned | 4 | 5.8% |
| budget_overflow | 1 | 1.45% |

- Training-detector (`is_disposition`) misses among outcome sentences: 33
- Outcome sentences with lower-court context (ambiguous, flagged): 24
- Excluding lower-court context (45 sentences): retained 95.56%, labelled judgement 18, primary chunk in a judgement section 13

Heuristic counts. Stage labels come from exact sentence matching against the
alignment record; outcome sentences come from audit-side cues. Provisional until
a person has reviewed a sample.

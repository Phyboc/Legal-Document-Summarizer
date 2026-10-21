# Dataset diagnosis - `artifacts\training_data\val100`

* records: **645** {'train': 0, 'validation': 645, 'test': 0}
* documents: **100**

## Gate results

| Gate | Value | Limit | Result |
| --- | --- | --- | --- |
| `unassigned_rate_within_band` | 0.2863 | 0.2 | **FAIL** |
| `no_impossible_targets` | 21 | 0 | **FAIL** |
| `judgement_signal_present` | 0.5 | 0.1 | PASS |

## Sections

| Section | Records | Share | Target words | Chunk words | Best score |
| --- | --- | --- | --- | --- | --- |
| `reasoning` | 387 | 60.0% | 92 | 621 | 0.987 |
| `facts` | 173 | 26.8% | 98 | 591 | 0.988 |
| `judgement` | 50 | 7.8% | 20 | 64 | 0.986 |
| `arguments_appellant` | 16 | 2.5% | 39 | 54 | 0.983 |
| `full_document` | 10 | 1.6% | 180 | 624 | 0.982 |
| `arguments_respondent` | 9 | 1.4% | 17 | 80 | 0.989 |

## Findings

* targets longer than their chunk: **18** (2.8%)
* targets with no prose: **3**
* target content-word coverage of chunk: n=645 mean=0.7517 median=0.7857 p10=0.5 p90=0.9818 max=1.0
* records below 20% coverage: **5**
* matched sentences under 5 words: 64 (2.8%)
* fragment-only records: **9**
* alignment scores (matched): n=2291 mean=0.9291 median=0.9872 p10=0.7467 p90=0.9991 max=1.0
* share of matched scores >= 0.9: 80.5%
* unassigned best scores: n=930 mean=0.0577 median=0.0064 p10=0.0003 p90=0.2183 max=0.393
* reference-summary coverage: n=100 mean=0.7424 median=0.756 p10=0.5652 p90=0.9231 max=1.0
* documents keeping judgement signal: 50/100 (50.0%)
* identical chunk text across splits: 0
* same-document chunk pairs over 10% overlap: 12

## Impossible targets (first examples)

* `validation_00016` chunk 1 (`facts`): 163 target words vs 138 chunk words
* `validation_00020` chunk 5 (`arguments_appellant`): 206 target words vs 121 chunk words
* `validation_00026` chunk 2 (`facts`): 381 target words vs 256 chunk words


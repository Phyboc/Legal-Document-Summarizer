# Dataset diagnosis - `artifacts\training_data\full`

* records: **385** {'train': 124, 'validation': 131, 'test': 130}
* documents: **60**

## Gate results

| Gate | Value | Limit | Result |
| --- | --- | --- | --- |
| `no_duplicate_record_keys` | 0 | 0 | PASS |
| `no_cross_split_duplicate_text` | 0 | 0 | PASS |
| `no_target_over_model_window` | 0 | 0 | PASS |
| `unassigned_rate_within_band` | 0.2621 | 0.2 | **FAIL** |
| `no_impossible_targets` | 13 | 0 | **FAIL** |
| `judgement_signal_present` | 0.5 | 0.1 | PASS |

## Sections

| Section | Records | Share | Target words | Chunk words | Best score |
| --- | --- | --- | --- | --- | --- |
| `reasoning` | 225 | 58.4% | 109 | 622 | 0.986 |
| `facts` | 100 | 26.0% | 100 | 598 | 0.986 |
| `judgement` | 30 | 7.8% | 22 | 79 | 0.993 |
| `full_document` | 19 | 4.9% | 154 | 618 | 0.976 |
| `arguments_appellant` | 10 | 2.6% | 20 | 42 | 0.951 |
| `arguments_respondent` | 1 | 0.3% | 173 | 141 | 0.973 |

## Findings

* targets longer than their chunk: **11** (2.9%)
* targets with no prose: **2**
* target content-word coverage of chunk: n=385 mean=0.7462 median=0.7759 p10=0.5 p90=0.96 max=1.0
* records below 20% coverage: **2**
* matched sentences under 5 words: 64 (4.4%)
* fragment-only records: **7**
* alignment scores (matched): n=1467 mean=0.9226 median=0.9857 p10=0.7401 p90=0.999 max=1.0
* share of matched scores >= 0.9: 78.0%
* unassigned best scores: n=526 mean=0.0648 median=0.0107 p10=0.0003 p90=0.2308 max=0.3998
* reference-summary coverage: n=60 mean=0.7425 median=0.75 p10=0.6 p90=0.8889 max=1.0
* documents keeping judgement signal: 30/60 (50.0%)
* identical chunk text across splits: 0
* same-document chunk pairs over 10% overlap: 4

## Section coverage (documents, not records)

| Section | Documents | Share of documents |
| --- | --- | --- |
| `arguments_appellant` | 10 | 16.7% |
| `arguments_respondent` | 1 | 1.7% |
| `facts` | 56 | 93.3% |
| `full_document` | 2 | 3.3% |
| `judgement` | 30 | 50.0% |
| `reasoning` | 58 | 96.7% |

* documents with no records at all: **0**
* structure-fallback documents (no section signal): 2
* judgement outcomes recorded by the builder: {'retained': 30, 'no_judgement_sentence': 26, 'no_judgement_section': 2, 'sentences_assigned_elsewhere': 2}

## Impossible targets (first examples)

* `train_00003` chunk 10 (`judgement`): 70 target words vs 67 chunk words
* `train_00005` chunk 2 (`reasoning`): 196 target words vs 185 chunk words
* `train_00006` chunk 8 (`arguments_appellant`): 245 target words vs 142 chunk words


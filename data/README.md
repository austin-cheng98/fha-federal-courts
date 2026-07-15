# Data Status

All inputs required for the frozen analysis are included in this directory. No network access or API key is needed for the default pipeline.

## Included Data

| Path | Rows | Description |
|---|---:|---|
| `raw/bulk_fha_cases.jsonl` | 757 | Canonical NOS-443 opinion-cluster metadata |
| `processed/paper_corpus.jsonl` | 751 | Full-text records recoverable from the canonical population |
| `external/housing_panel.csv` | 36 | Twelve circuits across ACS vintages 2012, 2017, and 2022 |
| `validation/gold_human_codings.json` | 93 | Hand-coded diagnostic overlap; 30 cases have second-pass coding |
| `validation/excluded_non_nos443.jsonl` | 180 | Candidate rows excluded by exact NOS normalization |

## Field Notes

### Opinion corpus

`paper_corpus.jsonl` contains one JSON object per line. Important fields are `cluster_id`, `case_name`, `court_id`, `circuit`, `court_level`, `year`, `nature_of_suit`, `text`, and `text_source`. Circuit is a geographic aggregation key, not the court that issued the opinion.

The 757-row metadata file is the retrieval denominator. Six canonical clusters do not have recoverable full text, so they are not included in text extraction.

### Housing panel

`housing_panel.csv` contains `circuit`, `year`, `dissimilarity_index`, and `denial_rate`. The dissimilarity measure is derived from Census ACS B03002 counts and is the only housing outcome used in the feasibility merge. HMDA denial rate is supplementary and is not used as a within-unit outcome.

The ACS vintages are intentionally non-overlapping. The legal merge produces eight matched cells, all in 2022; the pipeline records this limitation and does not estimate real-data fixed effects.

### Validation data

The hand-coded file is an enriched, choice-based diagnostic set rather than a probability sample. It supports construct-level precision, recall, F1, kappa, and cue-sparsity checks. The exclusion file preserves the 180 candidate rows rejected by the exact NOS-443 normalizer.

## Generated Files

`run_pipeline.py`, `score_goldset.py`, and `validation_robustness.py` write derived files to ignored `data/processed/` and `outputs/` paths. Generated files are not source inputs and are not committed.

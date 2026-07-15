# Data

The repository keeps only frozen inputs needed for the paper’s analysis.

| Path | Contents |
|---|---|
| `raw/bulk_fha_cases.jsonl` | 757 canonical NOS-443 opinion-cluster metadata rows |
| `processed/paper_corpus.jsonl` | 751 rows with recoverable full text |
| `external/housing_panel.csv` | 36 circuit-year ACS/HMDA rows for the feasibility check |
| `validation/gold_human_codings.json` | 93-case hand-coded overlap, including 30 second-pass cases |
| `validation/excluded_non_nos443.jsonl` | 180 excluded candidate rows |

The housing panel uses ACS vintages 2012, 2017, and 2022. Only eight substantive legal-corpus cells match, all in 2022, so no real-data fixed-effects estimate is reported. Derived files are written to ignored paths by the reproducibility scripts.

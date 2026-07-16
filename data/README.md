# Data

Frozen inputs for the default offline pipeline.

| Path | Rows | Description |
|---|---:|---|
| `raw/bulk_fha_cases.jsonl` | 757 | NOS-443 opinion-cluster metadata |
| `processed/paper_corpus.jsonl` | 751 | Full-text opinion records |
| `external/housing_panel.csv` | 36 | Circuit-year housing panel |
| `validation/gold_human_codings.json` | 93 | Hand-coded validation set |
| `validation/excluded_non_nos443.jsonl` | 180 | Excluded candidate records |

Generated files are ignored and can be rebuilt with the scripts in `../scripts/`.

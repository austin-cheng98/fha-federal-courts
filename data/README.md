# Data

The default workflow uses only committed local files. `data/processed/*.csv` and
`outputs/` are generated and ignored.

| Path | Role |
|---|---|
| `raw/bulk_fha_cases.jsonl` | Frozen NOS-443 candidate metadata |
| `processed/paper_corpus.jsonl` | Canonical opinion records with recovered text |
| `external/housing_panel.csv` | Circuit-year ACS dissimilarity and HMDA inputs |
| `validation/gold_human_codings.json` | Human-coded overlap and second-pass cases |
| `validation/excluded_non_nos443.jsonl` | Excluded candidate records |
| `validation/CODEBOOK.md` | Coding definitions and evidence-quote rules |
| `validation/random_sample_index.json` | Seeded 150-case draw (`20260720`) |
| `validation/llm_labels_3pass.json` | Frozen pass-level LLM labels |
| `validation/llm_majority_votes.json` | Frozen majority votes and self-consistency |

Run `make reproduce` from the repository root to rebuild derived files and reports.

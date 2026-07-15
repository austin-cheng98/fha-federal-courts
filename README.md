# FHA federal-court corpus

Frozen data and deterministic code for measuring Fair Housing Act signals in federal district-court opinions. The inputs contain 757 canonical NOS-443 clusters, 751 recoverable full texts, a 417-case rule-positive corpus, a 36-row housing feasibility panel, and a 93-case hand-coded validation set.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_pipeline.py
python scripts/score_goldset.py
python scripts/validation_robustness.py
```

`run_pipeline.py` writes derived feature, doctrine, FEII, panel, table, and summary files under ignored `outputs/` and `data/processed/` paths. The real housing merge has eight matched cells in one year; the pipeline records that feasibility result and does not report a real-data fixed-effects estimate. Use `--source synthetic` only for the seeded estimator check.

## Inputs

- `data/raw/bulk_fha_cases.jsonl`: 757-row NOS-443 population metadata.
- `data/processed/paper_corpus.jsonl`: 751 full-text records.
- `data/external/housing_panel.csv`: ACS/HMDA circuit-year feasibility input.
- `data/validation/gold_human_codings.json`: 93-case diagnostic overlap, including 30 second-pass cases.
- `data/validation/excluded_non_nos443.jsonl`: 180-row exclusion audit.

The code is deterministic for the frozen inputs: TF-IDF, LSA, and K-means use fixed settings; the synthetic estimator check uses a fixed seed. CourtListener, Census, HMDA, live-harvest, and figure-generation utilities are not included.

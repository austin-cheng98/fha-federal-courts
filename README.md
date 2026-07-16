# FHA Federal-Court Corpus

Reproducible code and frozen inputs for measuring Fair Housing Act signals in federal district-court opinions.

## Contents

- `src/fha/`: extraction, clustering, FEII, housing merge, and Schelling scenario modules.
- `scripts/`: pipeline, validation, and plotting entry points.
- `data/`: frozen CourtListener-derived inputs and validation files.
- `tests/`: offline regression tests.

## Setup

```bash
git clone https://github.com/austin-cheng98/fha-federal-courts.git
cd fha-federal-courts
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce

```bash
python scripts/run_pipeline.py
python scripts/score_goldset.py
python scripts/validation_robustness.py
python scripts/run_schelling.py
python scripts/plot_schelling.py
python scripts/plot_additional.py
PYTHONPATH=src pytest -q
```

Generated files are written to ignored `data/processed/` and `outputs/` paths.

## Data

Included inputs:

- `data/raw/bulk_fha_cases.jsonl`
- `data/processed/paper_corpus.jsonl`
- `data/external/housing_panel.csv`
- `data/validation/gold_human_codings.json`
- `data/validation/excluded_non_nos443.jsonl`

## License

MIT. See `CITATION.cff` for citation metadata.

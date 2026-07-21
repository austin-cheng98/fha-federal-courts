# FHA Federal-Court Corpus

Code and frozen inputs for measuring Fair Housing Act signals in federal district-court opinions.

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/austin-cheng98/fha-federal-courts.git
cd fha-federal-courts
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Reproduce

The complete workflow is offline. The LLM labels used by the paper are frozen in
`data/validation/`; reproduction does not make an API call.

```bash
make reproduce
```

This runs the deterministic pipeline, validation checks, random-sample verification,
frozen-LLM scoring, and prevalence correction. Generated tables and reports are written
under ignored `data/processed/` and `outputs/` paths.

To run individual checks:

```bash
make test
python3 scripts/draw_random_sample.py
python3 scripts/score_llm_baseline.py
python3 scripts/analyze_prevalence.py
```

`draw_random_sample.py --write` overwrites the committed sample index and should only be
used when the input frame is intentionally changed.

## Repository map

- `src/fha/` — extraction, FEII, housing feasibility inputs, the LLM baseline, the cross-circuit doctrinal-split test, and the Schelling gatekeeping model. A released doctrinal-embedding module is included but not used in the sorting analysis.
- `scripts/` — reproducible entry points used by `make reproduce`.
- `data/` — frozen corpus, housing panel, human coding, codebook, and LLM artifacts.
- `docs/LLM_BASELINE.md` — provenance and verification details for the frozen LLM baseline.
- `tests/` — offline regression tests.

## License

MIT. See `CITATION.cff` for citation metadata.

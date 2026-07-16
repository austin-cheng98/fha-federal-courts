# FHA Federal-Court Corpus

**Deterministic text measurement of Fair Housing Act signals in federal district-court opinions.**

## Overview

This repository contains a frozen CourtListener-derived corpus, a rule-based legal-text extractor, descriptive regime clustering, a circuit-year legal-signal index, an illustrative Schelling segregation coupling, and validation code. The unit of analysis is the opinion cluster. Circuit is used as a geographic aggregation key; the corpus is district-court only.

The released data contain 757 canonical NOS-443 clusters, 751 recoverable full texts, 417 rule-positive substantive clusters, and a 93-case hand-coded diagnostic overlap. The housing input is a feasibility check rather than a reported causal estimate: only eight substantive legal-corpus cells match, all in 2022.

## Analysis Framework

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         FHA MEASUREMENT FRAMEWORK                        │
├───────────────────────┬───────────────────────┬─────────────────────────┤
│  Corpus                │  Text measurement      │  Aggregation            │
│  NOS-443 clusters      │  Claims and duties     │  Circuit × year         │
│  751 full texts        │  Frameworks/remedies   │  FEII descriptive index │
│  417 substantive       │  Outcome cues          │  Housing feasibility    │
│  Schelling coupling    │  FEII → tolerance     │  Illustrative dynamics  │
└───────────────────────┴───────────────────────┴─────────────────────────┘
```

## Features

### Corpus construction

- Exact or description-equivalent NOS-443 normalization.
- District-court circuit mapping with an exclusion audit.
- Frozen metadata and full-text inputs for offline execution.

### Legal-text measurement

- Multi-label indicators for disparate treatment, disparate impact, reasonable accommodation, refusal/steering, and zoning/land use.
- Framework, proof-standard, precedent, remedy, disposition, and settlement cues.
- Character-offset evidence spans for inspectability.
- Negation sensitivity as a validation option.

### Descriptive structure

- TF-IDF → truncated SVD/LSA → K-means with fixed settings.
- Three regimes ordered by mean legal-signal breadth: weak, moderate, and strict.
- Circuit-year regime composition, transitions, and divergence tables.

### FEII and housing boundary

- Equal-weight standardized components for opinion volume, shrunken outcome-cue rate, and remedy-cue intensity.
- ACS Black–White dissimilarity and HMDA denial-rate inputs at the circuit-year level.
- Automatic feasibility guard: the eight-cell, one-year real merge receives a note rather than a fixed-effects estimate.

### Illustrative generative coupling

- A two-group Schelling model uses circuit-mean FEII to map into one neighborhood-tolerance parameter.
- The 20 × 20 Moore-neighborhood grid, 10% vacancies, 250-cycle cap, 40 replications, and seed are fixed in code.
- The simulation is a mechanism probe, not a calibrated housing model or causal estimate.

## Installation

Requirements: Python 3.11 or newer.

```bash
git clone https://github.com/austin-cheng98/fha-federal-courts.git
cd fha-federal-courts
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Reproducing Results

### Step 1: Run the frozen-data pipeline

```bash
python scripts/run_pipeline.py
```

This extracts the 417 substantive records, writes case features, regime tables, FEII cells, the housing merge, and a feasibility summary. Derived files are written under ignored `data/processed/` and `outputs/` paths.

### Step 2: Recompute diagnostic extraction metrics

```bash
python scripts/score_goldset.py
```

This regenerates machine labels from the current extractor and compares them with the 93-case hand-coded overlap, including the 30-case second-pass reliability check.

### Step 3: Run sensitivity checks

```bash
python scripts/validation_robustness.py
```

This writes negation, era, FEII-component, regime-term, and case-mix diagnostics to ignored `outputs/validation/` files. The real housing branch remains non-estimable.

### Step 4: Run tests

```bash
PYTHONPATH=src pytest -q
```

The test suite checks court mapping, FHA identification, extraction, standardized FEII construction, and the seeded synthetic estimator oracle (`BETA_TRUE = -0.70`). The synthetic check is software validation, not evidence about the real corpus.

### Step 5: Run the illustrative model

```bash
python scripts/run_schelling.py
python scripts/plot_schelling.py
python scripts/plot_additional.py
```

These commands write the deterministic circuit-level scenarios and the three Schelling/FEII figures used in the paper. FEII changes only the tolerance parameter; the mapping endpoints are stipulated for illustration and are not fitted to the housing panel.

## Data Sources

| Source | Repository input | Use |
|---|---|---|
| CourtListener bulk-derived snapshot | `data/raw/bulk_fha_cases.jsonl` | 757-row canonical NOS-443 population denominator |
| CourtListener recoverable text | `data/processed/paper_corpus.jsonl` | 751 full-text opinion clusters |
| Census ACS | `data/external/housing_panel.csv` | Black–White dissimilarity at 2012, 2017, and 2022 vintages |
| CFPB HMDA aggregation | `data/external/housing_panel.csv` | Supplementary 2022 denial-rate field |
| Hand coding | `data/validation/gold_human_codings.json` | 93-case diagnostic overlap and 30 second-pass cases |
| Exclusion audit | `data/validation/excluded_non_nos443.jsonl` | 180 candidate rows rejected by exact NOS normalization |

The shipped housing panel has 36 circuit-year rows. Joining it to the substantive opinion corpus leaves eight cells, all in 2022, so the repository does not report a real-data TWFE, bootstrap, event study, or DiD estimate.

## Project Structure

```text
fha-federal-courts/
├── LICENSE
├── README.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/bulk_fha_cases.jsonl
│   ├── processed/paper_corpus.jsonl
│   ├── external/housing_panel.csv
│   └── validation/
│       ├── excluded_non_nos443.jsonl
│       └── gold_human_codings.json
├── scripts/
│   ├── run_pipeline.py
│   ├── run_schelling.py
│   ├── plot_schelling.py
│   ├── plot_additional.py
│   ├── score_goldset.py
│   └── validation_robustness.py
├── src/fha/
│   ├── classify.py
│   ├── config.py
│   ├── doctrine.py
│   ├── econometrics.py
│   ├── extract.py
│   ├── feii.py
│   ├── housing.py
│   ├── pipeline.py
│   ├── reference.py
│   ├── reports.py
│   ├── schelling.py
│   └── synth.py
└── tests/test_pipeline.py
```

## Current Results

| Quantity | Value |
|---|---:|
| Canonical NOS-443 clusters | 757 |
| Recoverable full texts | 751 |
| Rule-positive substantive clusters | 417 |
| Hand-coded diagnostic overlap | 93 |
| Second-pass cases | 30 |
| Housing input rows | 36 |
| Matched substantive housing cells | 8 |
| Matched housing vintages | 2022 only |

The extractor is designed to measure textual signals, not to adjudicate legal merits. Claim indicators are multi-label, outcome cues are directional and sparse, and circuit summaries are descriptive. The housing merge is retained to document the data boundary rather than to support causal interpretation.

## License and Citation

Code is released under the MIT License. See `CITATION.cff` for citation metadata.

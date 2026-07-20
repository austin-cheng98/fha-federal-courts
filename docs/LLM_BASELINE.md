# Frozen LLM baseline

The paper's LLM comparison is a frozen, offline artifact. The repository does not call a
model during reproduction.

## Provenance

- Model: `claude-opus-4-8`
- Passes: three independent labels per opinion; per-field majority vote reported
- Human overlap: 93 cases from `data/validation/gold_human_codings.json`
- Random draw: 150 cases from the 658-case non-overlap frame, seed `20260720`
- Returned jobs: 728 of 729; the missing pass is cluster `9672005`, whose two returned
  passes agree on all fields

The human overlap is the only set with a reference label. The random-draw labels are model
silver labels used for robustness and prevalence checks.

## Verify

From the repository root, after installation:

```bash
python3 scripts/draw_random_sample.py
python3 scripts/score_llm_baseline.py
python3 scripts/analyze_prevalence.py
```

The first command verifies the committed sample index. The second verifies that recomputed
majority votes match `data/validation/llm_majority_votes.json`. The third reports the
prevalence correction and exact paired tests.

Derived CSV and JSON files are written to ignored `outputs/paper/validation/`.

The committed inputs are `data/validation/llm_labels_3pass.json`,
`data/validation/llm_majority_votes.json`, `data/validation/random_sample_index.json`, and
`data/validation/CODEBOOK.md`.

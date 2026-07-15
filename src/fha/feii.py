from __future__ import annotations

import numpy as np
import pandas as pd

REMEDY_WEIGHTS = {"remedy_injunction": 1.0, "remedy_civil_penalty": 0.9,
                  "remedy_declaratory": 0.6, "remedy_damages": 0.5,
                  "remedy_attorneys_fees": 0.2}
DEFAULT_COMPONENT_WEIGHTS = {"opinion_volume": 1 / 3,
                             "outcome_cue_rate": 1 / 3,
                             "remedy_cue_intensity": 1 / 3}


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def remedy_cue_intensity_row(row: pd.Series) -> float:
    total = sum(REMEDY_WEIGHTS.values())
    return float(sum(w * row.get(k, 0) for k, w in REMEDY_WEIGHTS.items()) / total)

def aggregate(features: pd.DataFrame, unit: str = "circuit",
              weights: dict | None = None) -> pd.DataFrame:
    df = features.copy()
    df = df[df[unit].notna() & df["year"].notna()]
    df["remedy_cue_intensity"] = df.apply(remedy_cue_intensity_row, axis=1)

    outcome_col = "outcome_cue" if "outcome_cue" in df.columns else "plaintiff_win"
    grp = df.groupby([unit, "year"])
    panel = grp.agg(
        n_opinions=("cluster_id", "count"),
        n_resolved=(outcome_col, lambda s: int(s.notna().sum())),
        win_sum=(outcome_col, lambda s: float(s.dropna().sum())),
        mean_enforcement=("enforcement_strength", "mean"),
        mean_strictness=("doctrinal_strictness", "mean"),
        remedy_cue_intensity=("remedy_cue_intensity", "mean"),
        share_disparate_impact=("claim_disparate_impact", "mean"),
    ).reset_index()






    grand = (panel["win_sum"].sum() / max(panel["n_resolved"].sum(), 1))
    k = 4.0
    panel["outcome_cue_rate"] = ((panel["win_sum"] + k * grand) /
                                  (panel["n_resolved"] + k))
    comps = ["opinion_volume", "outcome_cue_rate", "remedy_cue_intensity"]
    panel["opinion_volume"] = np.log1p(panel["n_opinions"])
    base = dict(weights or DEFAULT_COMPONENT_WEIGHTS)
    w = {c: base.get(c, 0.0) for c in comps}
    wsum = sum(w.values()) or 1.0
    w = {c: v / wsum for c, v in w.items()}
    z = pd.DataFrame({c: _zscore(panel[c]) for c in comps})
    panel["FEII"] = sum(w[c] * z[c] for c in comps)
    for c in comps:
        panel[f"z_{c}"] = z[c]
    return panel.rename(columns={unit: "unit"}).assign(unit_type=unit)

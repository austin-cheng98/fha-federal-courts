"""
Federal Enforcement Intensity Index: aggregate opinion-cluster features to
circuit x year and combine z-scored components (opinion volume,
directional outcome-cue rate, remedy-cue intensity) into a standardized
descriptive enforcement proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REMEDY_WEIGHTS = {"remedy_injunction": 1.0, "remedy_civil_penalty": 0.9,
                  "remedy_declaratory": 0.6, "remedy_damages": 0.5,
                  "remedy_attorneys_fees": 0.2}
DEFAULT_COMPONENT_WEIGHTS = {"opinion_volume": 1 / 3,
                             "outcome_cue_rate": 1 / 3,
                             "remedy_cue_intensity": 1 / 3,
                             "docket_share": 0.25}


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def remedy_cue_intensity_row(row: pd.Series) -> float:
    """Normalized weighted presence of remedy terms, not awarded relief."""
    total = sum(REMEDY_WEIGHTS.values())
    return float(sum(w * row.get(k, 0) for k, w in REMEDY_WEIGHTS.items()) / total)


# Backward-compatible name for downstream users of the original package.
remedy_severity_row = remedy_cue_intensity_row


def aggregate(features: pd.DataFrame, unit: str = "circuit",
              housing_docket: pd.DataFrame | None = None,
              weights: dict | None = None) -> pd.DataFrame:
    """Build the circuit(or metro) x year FEII panel.

    housing_docket: optional frame [unit, year, n_housing_cases] giving the
    denominator for docket_share (e.g. all NOS-443 federal decisions). If absent,
    the index uses the three active components documented above.
    """
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
    # Directional outcome cue: empirical-Bayes shrinkage toward the grand cue
    # rate. Party-role resolution is not available in the source corpus, so
    # this component is not called a plaintiff success rate.
    # weighted by the number of RESOLVED cases in the cell. This avoids both
    # naive global-mean imputation (which biases low-volume cells to the mean)
    # and dropping cells with few resolved outcomes. k = pseudo-count.
    grand = (panel["win_sum"].sum() / max(panel["n_resolved"].sum(), 1))
    k = 4.0
    panel["outcome_cue_rate"] = ((panel["win_sum"] + k * grand) /
                                  (panel["n_resolved"] + k))
    panel["plaintiff_success"] = panel["outcome_cue_rate"]  # legacy alias

    # Docket share is optional; without an external denominator it is omitted
    # rather than double-counting the opinion-volume component.
    comps = ["opinion_volume", "outcome_cue_rate", "remedy_cue_intensity"]
    if housing_docket is not None:
        panel = panel.merge(housing_docket, on=[unit, "year"], how="left")
        panel["docket_share"] = (panel["n_opinions"] /
                                 panel["n_housing_cases"].replace(0, np.nan))
        panel["docket_share"] = panel["docket_share"].fillna(0)
        comps = ["opinion_volume", "docket_share", "outcome_cue_rate",
                 "remedy_cue_intensity"]

    panel["opinion_volume"] = np.log1p(panel["n_opinions"])
    # Legacy aliases retained for consumers of the first release; the paper
    # and new tables use the unambiguous names above.
    panel["n_filings"] = panel["n_opinions"]
    panel["filings"] = panel["opinion_volume"]
    panel["remedy_severity"] = panel["remedy_cue_intensity"]
    base = dict(weights or DEFAULT_COMPONENT_WEIGHTS)
    if "outcome_cue_rate" not in base and "plaintiff_success" in base:
        base["outcome_cue_rate"] = base["plaintiff_success"]
    w = {c: base.get(c, 0.0) for c in comps}
    wsum = sum(w.values()) or 1.0
    w = {c: v / wsum for c, v in w.items()}        # renormalize to the used set
    z = pd.DataFrame({c: _zscore(panel[c]) for c in comps})
    panel["FEII"] = sum(w[c] * z[c] for c in comps)
    for c in comps:
        panel[f"z_{c}"] = z[c]
    return panel.rename(columns={unit: "unit"}).assign(unit_type=unit)

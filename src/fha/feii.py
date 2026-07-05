"""
Federal Enforcement Intensity Index: aggregate case features to circuit x
year and combine z-scored components (filings, docket_share,
plaintiff_success, remedy_severity) into a standardized index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REMEDY_WEIGHTS = {"remedy_injunction": 1.0, "remedy_civil_penalty": 0.9,
                  "remedy_declaratory": 0.6, "remedy_damages": 0.5,
                  "remedy_attorneys_fees": 0.2}
DEFAULT_COMPONENT_WEIGHTS = {"filings": 0.25, "docket_share": 0.25,
                             "plaintiff_success": 0.25, "remedy_severity": 0.25}


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def remedy_severity_row(row: pd.Series) -> float:
    return float(sum(w * row.get(k, 0) for k, w in REMEDY_WEIGHTS.items()))


def aggregate(features: pd.DataFrame, unit: str = "circuit",
              housing_docket: pd.DataFrame | None = None,
              weights: dict | None = None) -> pd.DataFrame:
    """Build the circuit(or metro) x year FEII panel.

    housing_docket: optional frame [unit, year, n_housing_cases] giving the
    denominator for docket_share (e.g. all NOS-443 federal decisions). If absent,
    docket_share falls back to within-year share of FHA filings across cells.
    """
    df = features.copy()
    df = df[df[unit].notna() & df["year"].notna()]
    df["remedy_severity"] = df.apply(remedy_severity_row, axis=1)

    grp = df.groupby([unit, "year"])
    panel = grp.agg(
        n_filings=("cluster_id", "count"),
        n_resolved=("plaintiff_win", lambda s: int(s.notna().sum())),
        win_sum=("plaintiff_win", lambda s: float(s.dropna().sum())),
        mean_enforcement=("enforcement_strength", "mean"),
        mean_strictness=("doctrinal_strictness", "mean"),
        remedy_severity=("remedy_severity", "mean"),
        share_disparate_impact=("claim_disparate_impact", "mean"),
    ).reset_index()
    # plaintiff success: empirical-Bayes shrinkage toward the grand win rate,
    # weighted by the number of RESOLVED cases in the cell. This avoids both
    # naive global-mean imputation (which biases low-volume cells to the mean)
    # and dropping cells with few resolved outcomes. k = pseudo-count.
    grand = (panel["win_sum"].sum() / max(panel["n_resolved"].sum(), 1))
    k = 4.0
    panel["plaintiff_success"] = ((panel["win_sum"] + k * grand) /
                                  (panel["n_resolved"] + k))

    # docket share + component list (drop docket_share in the fallback, where it
    # is just a monotone transform of n_filings and would double-count volume).
    comps = ["filings", "plaintiff_success", "remedy_severity"]
    if housing_docket is not None:
        panel = panel.merge(housing_docket, on=[unit, "year"], how="left")
        panel["docket_share"] = (panel["n_filings"] /
                                 panel["n_housing_cases"].replace(0, np.nan))
        panel["docket_share"] = panel["docket_share"].fillna(0)
        comps = ["filings", "docket_share", "plaintiff_success", "remedy_severity"]

    panel["filings"] = np.log1p(panel["n_filings"])
    base = weights or DEFAULT_COMPONENT_WEIGHTS
    w = {c: base.get(c, 0.0) for c in comps}
    wsum = sum(w.values()) or 1.0
    w = {c: v / wsum for c, v in w.items()}        # renormalize to the used set
    z = pd.DataFrame({c: _zscore(panel[c]) for c in comps})
    panel["FEII"] = sum(w[c] * z[c] for c in comps)
    for c in comps:
        panel[f"z_{c}"] = z[c]
    return panel.rename(columns={unit: "unit"}).assign(unit_type=unit)

"""
Robustness checks: exclude_scotus(), placebo_statute(), lagged_effects(),
and alternative_grouping() (Census-style regions instead of circuits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import econometrics as ec
from . import feii as feii_mod
# Census-style region grouping for the alternative-grouping robustness check.
REGIONS = {
    "1": "Northeast", "2": "Northeast", "3": "Northeast",
    "4": "South", "5": "South", "11": "South",
    "6": "Midwest", "7": "Midwest", "8": "Midwest",
    "9": "West", "10": "West", "DC": "DC",
}


def exclude_scotus(features: pd.DataFrame, housing: pd.DataFrame,
                   y: str = "dissimilarity_index") -> dict:
    """Drop supreme-court cases, rebuild FEII, re-run the TWFE."""
    sub = features[features["court_level"] != "supreme"]
    panel_feii = feii_mod.aggregate(sub, unit="circuit")
    panel = ec.build_panel(panel_feii, housing)
    res = ec.twfe(panel, y=y, x="FEII")
    res["check"] = "exclude_scotus"
    return res


def placebo_statute(panel: pd.DataFrame, y: str = "dissimilarity_index",
                    B: int = 499, seed: int = 1) -> dict:
    """Randomization-inference placebo.

    Permute FEII within unit B times to build the null distribution of the TWFE
    t-statistic, then report placebo_p = share of |t_placebo| >= |t_real|. A
    single permutation (the old version) is just one draw and uninformative.

    Caveat: this reshuffles the SAME enforcement series rather than substituting a
    genuine non-housing statute (e.g. Title VII / ADA Title II enforcement, listed
    in SETTINGS.placebo_statutes). A true placebo-statute test requires harvesting
    those non-housing dockets and is the recommended next step; this within-unit
    permutation tests only that the FEII<->housing link is not an artifact of the
    panel's marginal structure.
    """
    real = ec.twfe(panel, y=y, x="FEII")
    real_t = abs(real.get("t", np.nan))
    rng = np.random.default_rng(seed)
    df = panel.copy()
    ts = []
    for _ in range(B):
        df["FEII_placebo"] = df.groupby("unit")["FEII"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        r = ec.twfe(df, y=y, x="FEII_placebo")
        t = r.get("t")
        if t is not None and not (isinstance(t, float) and np.isnan(t)):
            ts.append(abs(t))
    ts = np.array(ts)
    placebo_p = (float((ts >= real_t).mean())
                 if len(ts) and not np.isnan(real_t) else float("nan"))
    return {"check": "placebo_randomization_inference",
            "real_coef": real.get("coef"), "real_t": real.get("t"),
            "placebo_p": placebo_p,
            "placebo_t_mean": float(ts.mean()) if len(ts) else float("nan"),
            "B": int(len(ts))}


def lagged_effects(panel: pd.DataFrame, y: str = "dissimilarity_index",
                   lags: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Y_t on FEII_{t-k} for each k (with FE)."""
    df = panel.sort_values(["unit", "year"]).copy()
    rows = []
    for k in lags:
        df[f"FEII_lag{k}"] = df.groupby("unit")["FEII"].shift(k)
        sub = df.dropna(subset=[f"FEII_lag{k}", y])
        if sub["unit"].nunique() < 2 or len(sub) < 20:
            continue
        r = ec.twfe(sub, y=y, x=f"FEII_lag{k}")
        rows.append({"lag": k, "coef": round(r["coef"], 4), "se": round(r["se"], 4),
                     "p": round(r["p"], 4), "n": r["n"]})
    return pd.DataFrame(rows)


def alternative_grouping(features: pd.DataFrame, housing: pd.DataFrame,
                         y: str = "dissimilarity_index") -> dict:
    """Re-run with circuits collapsed into Census-style regions."""
    feat = features.copy()
    feat["region"] = feat["circuit"].map(REGIONS).fillna("Other")
    feat_r = feat.rename(columns={"circuit": "_circ_orig", "region": "circuit"})
    panel_feii = feii_mod.aggregate(feat_r, unit="circuit")

    h = housing.copy()
    h["region"] = h["circuit"].map(REGIONS).fillna("Other")
    h_r = (h.groupby(["region", "year"])[y].mean().reset_index()
           .rename(columns={"region": "circuit"}))
    panel = ec.build_panel(panel_feii, h_r)
    res = ec.twfe(panel, y=y, x="FEII")
    res["check"] = "alternative_grouping(region)"
    return res


def run_all(features: pd.DataFrame, panel: pd.DataFrame, housing: pd.DataFrame,
            y: str = "dissimilarity_index") -> dict:
    return {
        "exclude_scotus": exclude_scotus(features, housing, y),
        "placebo_statute": placebo_statute(panel, y),
        "lagged_effects": lagged_effects(panel, y),
        "alternative_grouping": alternative_grouping(features, housing, y),
    }

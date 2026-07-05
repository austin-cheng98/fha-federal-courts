"""
Identification and inference: twfe() two-way fixed effects (SE clustered by
unit), event_study(), wild_cluster_bootstrap() for few clusters, and
supporting did_2x2() / circuit_compare() helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def build_panel(feii_panel: pd.DataFrame, housing: pd.DataFrame,
                unit_left: str = "unit", unit_right: str = "circuit") -> pd.DataFrame:
    """Merge FEII (unit x year) with housing outcomes (unit x year)."""
    h = housing.rename(columns={unit_right: "unit"})
    panel = feii_panel.merge(h, on=["unit", "year"], how="inner")
    return panel.sort_values(["unit", "year"]).reset_index(drop=True)


def _tidy(res, term: str) -> dict:
    return {"coef": float(res.params.get(term, np.nan)),
            "se": float(res.bse.get(term, np.nan)),
            "t": float(res.tvalues.get(term, np.nan)),
            "p": float(res.pvalues.get(term, np.nan)),
            "ci_low": float(res.conf_int().loc[term, 0]) if term in res.params else np.nan,
            "ci_high": float(res.conf_int().loc[term, 1]) if term in res.params else np.nan,
            "n": int(res.nobs), "r2": float(res.rsquared)}


MIN_RESID_DF = 5    # minimum residual df before a FE regression is trusted


def _degeneracy_note(panel: pd.DataFrame, n_regressors: int) -> dict | None:
    """Return a {note,...} dict if the panel is too thin for the FE regression
    (more parameters than data leaves singular / over-fit estimates), else None.
    Guards the tiny-real-corpus case where TWFE would silently return garbage."""
    n_units = panel["unit"].nunique()
    n_years = panel["year"].nunique()
    n_params = (n_units - 1) + (n_years - 1) + n_regressors + 1  # +intercept
    if len(panel) < n_params + MIN_RESID_DF:
        return {"coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n": int(len(panel)),
                "r2": np.nan,
                "note": (f"insufficient panel for FE inference: n={len(panel)}, "
                         f"params={n_params}, units={n_units}, years={n_years}")}
    return None


# Cluster-robust inference referenced to t(G-1), not the normal -- essential
# with the ~12 federal circuits, where the z-reference is badly anti-conservative.
_CLUSTER = lambda g: {"groups": g, "use_t": True}


def twfe(panel: pd.DataFrame, y: str, x: str = "FEII",
         controls: list[str] | None = None, cluster: str = "unit") -> dict:
    """7.1 core regression: two-way FE, cluster-robust SE (t(G-1) reference)."""
    guard = _degeneracy_note(panel, 1 + (len(controls) if controls else 0))
    if guard:
        guard.update({"model": "twfe", "y": y, "x": x, "controls": controls or []})
        return guard
    ctrl = (" + " + " + ".join(controls)) if controls else ""
    formula = f"{y} ~ {x}{ctrl} + C(unit) + C(year)"
    res = smf.ols(formula, data=panel).fit(
        cov_type="cluster", cov_kwds=_CLUSTER(panel[cluster]))
    out = _tidy(res, x)
    out.update({"model": "twfe", "y": y, "x": x, "controls": controls or [],
                "n_clusters": int(panel[cluster].nunique())})
    return out


def wild_cluster_bootstrap(panel: pd.DataFrame, y: str, x: str = "FEII",
                           controls: list[str] | None = None,
                           cluster: str = "unit", B: int = 999,
                           seed: int = 0) -> dict:
    """Wild cluster restricted bootstrap-t for H0: beta_x = 0 (Cameron-Gelbach-
    Miller). With only ~12 circuit clusters this is the trustworthy inference;
    the t(G-1) p-value is a fallback. Rademacher cluster weights, residuals from
    the model with x imposed to zero. Returns the bootstrap p-value."""
    ctrl = (" + " + " + ".join(controls)) if controls else ""
    full = smf.ols(f"{y} ~ {x}{ctrl} + C(unit) + C(year)", data=panel).fit(
        cov_type="cluster", cov_kwds=_CLUSTER(panel[cluster]))
    t_obs = float(full.tvalues.get(x, np.nan))
    # restricted fit: impose beta_x = 0 by omitting x
    rest = smf.ols(f"{y} ~ 1{ctrl} + C(unit) + C(year)", data=panel).fit()
    yhat = rest.fittedvalues.to_numpy()
    uhat = rest.resid.to_numpy()
    clusters = panel[cluster].to_numpy()
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    df = panel.copy()
    n_ge = 0
    n_ok = 0
    for _ in range(B):
        w = dict(zip(uniq, rng.integers(0, 2, len(uniq)) * 2 - 1))  # +/-1
        wv = np.array([w[g] for g in clusters], dtype=float)
        df["_ystar"] = yhat + wv * uhat
        fb = smf.ols(f"_ystar ~ {x}{ctrl} + C(unit) + C(year)", data=df).fit(
            cov_type="cluster", cov_kwds=_CLUSTER(df[cluster]))
        tb = fb.tvalues.get(x, np.nan)
        if tb is not None and not np.isnan(tb):
            n_ok += 1
            if abs(tb) >= abs(t_obs):
                n_ge += 1
    return {"x": x, "coef": float(full.params.get(x, np.nan)), "t_obs": t_obs,
            "wild_p": (n_ge / n_ok if n_ok else float("nan")),
            "B": int(n_ok), "n_clusters": int(len(uniq))}


def derive_treatment(panel: pd.DataFrame, shock_year: int,
                     measure: str = "FEII") -> dict:
    """Split units high/low by their EX ANTE (pre-shock) mean of `measure`.

    Using only year < shock_year keeps assignment exogenous to the post-period
    outcome. The earlier rule split on the post-minus-pre *change* in FEII, which
    sorts units on a realized consequence of the shock -- a regression-to-the-mean
    / selection-on-the-outcome bias that manufactures a spurious treated x post
    effect (confirmed by Monte Carlo). Never reference year >= shock_year here.
    """
    pre = panel[panel.year < shock_year].groupby("unit")[measure].mean().dropna()
    if pre.empty:
        return {}
    cut = pre.median()
    return {u: int(v > cut) for u, v in pre.items()}


def event_study(panel: pd.DataFrame, y: str, shock_year: int,
                treated: dict | None = None, k_min: int = -5, k_max: int = 5,
                measure: str = "FEII") -> pd.DataFrame:
    """7.2 dynamic DiD. Returns coefficient path by event time (k=-1 omitted)."""
    df = panel.copy()
    if treated is None:
        treated = derive_treatment(df, shock_year, measure)
    df["treated"] = df["unit"].map(treated).fillna(0).astype(int)
    df["k"] = (df["year"] - shock_year).clip(lower=k_min, upper=k_max)
    # event-time dummies interacted with treated, reference k=-1
    terms = []
    for k in range(k_min, k_max + 1):
        if k == -1:
            continue
        col = f"D_{'m' if k < 0 else 'p'}{abs(k)}"
        df[col] = ((df["k"] == k) & (df["treated"] == 1)).astype(int)
        terms.append((k, col))
    if _degeneracy_note(df, len(terms) + 1):     # event-time dummies + treated
        out = pd.DataFrame([{"k": -1, "coef": 0.0, "se": 0.0,
                             "ci_low": 0.0, "ci_high": 0.0}])
        out.attrs["note"] = "insufficient panel for event study"
        out.attrs["shock_year"] = shock_year
        return out
    formula = (f"{y} ~ " + " + ".join(c for _, c in terms) +
               " + treated + C(unit) + C(year)")
    res = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds=_CLUSTER(df["unit"]))
    rows = [{"k": -1, "coef": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0}]
    for k, col in terms:
        rows.append({"k": k, "coef": float(res.params.get(col, np.nan)),
                     "se": float(res.bse.get(col, np.nan)),
                     "ci_low": float(res.conf_int().loc[col, 0]) if col in res.params else np.nan,
                     "ci_high": float(res.conf_int().loc[col, 1]) if col in res.params else np.nan})
    out = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    out.attrs["n_treated"] = int(sum(treated.values()))
    out.attrs["shock_year"] = shock_year
    return out


def did_2x2(panel: pd.DataFrame, y: str, shock_year: int,
            treated: dict | None = None, measure: str = "FEII") -> dict:
    """Canonical 2x2 DiD: treated x post interaction."""
    df = panel.copy()
    if treated is None:
        treated = derive_treatment(df, shock_year, measure)
    df["treated"] = df["unit"].map(treated).fillna(0).astype(int)
    df["post"] = (df["year"] >= shock_year).astype(int)
    res = smf.ols(f"{y} ~ treated*post", data=df).fit(
        cov_type="cluster", cov_kwds=_CLUSTER(df["unit"]))
    out = _tidy(res, "treated:post")
    out.update({"model": "did_2x2", "y": y, "shock_year": shock_year,
                "n_treated": int(sum(treated.values())),
                "n_clusters": int(df["unit"].nunique())})
    return out


def circuit_compare(features: pd.DataFrame, panel: pd.DataFrame,
                    y: str) -> pd.DataFrame:
    """7.3 cross-circuit DESCRIPTIVE comparison table.

    NOTE: the attached corr_* values are raw cross-sectional correlations over
    only ~12 circuits -- they are descriptive context, NOT a causal estimate, and
    are confounded by everything that differs across circuits (the within-circuit
    TWFE/event study is the identification, not this). They are also partly
    mechanical: mean_FEII and mean_{y} are circuit means of series that already
    co-move by construction. Treat as exploratory only.
    """
    by_circ = (features.groupby("circuit")
               .agg(n_cases=("cluster_id", "count"),
                    mean_strictness=("doctrinal_strictness", "mean"),
                    plaintiff_success=("plaintiff_win", lambda s: s.dropna().mean()),
                    share_disparate_impact=("claim_disparate_impact", "mean"))
               .reset_index())
    outcome = panel.groupby("unit")[[y, "FEII"]].mean().reset_index()
    outcome = outcome.rename(columns={"unit": "circuit", y: f"mean_{y}",
                                      "FEII": "mean_FEII"})
    tab = by_circ.merge(outcome, on="circuit", how="left")
    tab.attrs["corr_strictness_outcome"] = float(
        tab["mean_strictness"].corr(tab[f"mean_{y}"]))
    tab.attrs["corr_FEII_outcome"] = float(tab["mean_FEII"].corr(tab[f"mean_{y}"]))
    return tab.round(4)

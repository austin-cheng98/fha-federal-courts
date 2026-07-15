import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def build_panel(feii_panel, housing):
    panel = feii_panel.merge(
        housing.rename(columns={"circuit": "unit"}),
        on=["unit", "year"], how="inner")
    return panel.sort_values(["unit", "year"]).reset_index(drop=True)


def twfe(panel, y, x="FEII"):
    units = panel["unit"].nunique()
    years = panel["year"].nunique()
    parameters = (units - 1) + (years - 1) + 2
    if len(panel) < parameters + 5:
        return {
            "coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
            "n": int(len(panel)),
            "note": (f"insufficient panel for FE inference: n={len(panel)}, "
                     f"params={parameters}, units={units}, years={years}"),
        }
    model = smf.ols(f"{y} ~ {x} + C(unit) + C(year)", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["unit"], "use_t": True})
    return {
        "coef": float(model.params[x]),
        "se": float(model.bse[x]),
        "t": float(model.tvalues[x]),
        "p": float(model.pvalues[x]),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
    }

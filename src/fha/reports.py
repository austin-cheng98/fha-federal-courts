"""
Output writers: doctrinal map, enforcement effectiveness, and law-in-action
gap. Figures to outputs/figures, tables to outputs/tables (matplotlib
optional).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _maybe_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# (1) doctrinal map
def doctrinal_map_outputs(doctrine_map: pd.DataFrame,
                          divergence: pd.DataFrame) -> dict:
    doctrine_map.to_csv(config.TABLES / "doctrinal_map.csv", index=False)
    divergence.to_csv(config.TABLES / "doctrinal_divergence.csv", index=False)
    paths = {"doctrinal_map_csv": str(config.TABLES / "doctrinal_map.csv"),
             "divergence_csv": str(config.TABLES / "doctrinal_divergence.csv")}

    plt = _maybe_plt()
    if plt is not None and "share_strict" in doctrine_map:
        pivot = (doctrine_map.pivot_table(index="circuit", columns="year",
                 values="share_strict", aggfunc="mean").sort_index())
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r",
                       vmin=0, vmax=1)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        xt = list(pivot.columns)
        step = max(1, len(xt) // 12)
        ax.set_xticks(range(0, len(xt), step), [xt[i] for i in range(0, len(xt), step)])
        ax.set_title("Federal FHA doctrinal map: share of cases in 'strict' regime")
        ax.set_xlabel("year"); ax.set_ylabel("circuit")
        fig.colorbar(im, ax=ax, label="share strict")
        fig.tight_layout()
        p = config.FIGURES / "doctrinal_map_heatmap.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        paths["doctrinal_map_png"] = str(p)
    return paths


# (2) enforcement effectiveness
def effectiveness_outputs(twfe_res: dict, event_study: pd.DataFrame,
                          did2x2: dict | None = None) -> dict:
    eff = pd.DataFrame([twfe_res])
    eff.to_csv(config.TABLES / "effectiveness_twfe.csv", index=False)
    event_study.to_csv(config.TABLES / "event_study.csv", index=False)
    paths = {"twfe_csv": str(config.TABLES / "effectiveness_twfe.csv"),
             "event_study_csv": str(config.TABLES / "event_study.csv")}
    if did2x2:
        pd.DataFrame([did2x2]).to_csv(config.TABLES / "did_2x2.csv", index=False)
        paths["did_2x2_csv"] = str(config.TABLES / "did_2x2.csv")

    plt = _maybe_plt()
    if plt is not None and len(event_study):
        fig, ax = plt.subplots(figsize=(8, 5))
        es = event_study.sort_values("k")
        ax.errorbar(es["k"], es["coef"],
                    yerr=1.96 * es["se"].fillna(0), fmt="o-", capsize=3)
        ax.axhline(0, color="grey", lw=0.8); ax.axvline(-0.5, color="red", ls="--", lw=0.8)
        ax.set_title("Event study: housing outcome around FHA enforcement shock")
        ax.set_xlabel("years since shock (k)"); ax.set_ylabel("effect on outcome")
        fig.tight_layout()
        p = config.FIGURES / "event_study.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        paths["event_study_png"] = str(p)
    return paths


# (3) law-in-action gap
def law_in_action_gap(features: pd.DataFrame, panel: pd.DataFrame,
                      outcome: str = "dissimilarity_index",
                      improvement_is_decrease: bool = True) -> pd.DataFrame:
    """Gap = z(doctrinal strictness) - z(observed housing improvement), per circuit.

    Positive gap  => courts talk strict but housing barely moves (law-in-action gap).
    Negative gap  => housing improved more than doctrine alone would predict.

    Caveat: z-scoring over only ~12 circuits is coarse and the gap is a DESCRIPTIVE
    contrast of two standardized series, not a tested quantity; the "observed
    improvement" is a noisy first-to-last difference. Read it as exploratory.
    """
    strict = (features.groupby("circuit")["doctrinal_strictness"].mean()
              .rename("strictness"))
    # observed change = first-to-last change in the housing outcome per unit
    chg = []
    for u, g in panel.sort_values("year").groupby("unit"):
        g = g.dropna(subset=[outcome])
        if len(g) >= 2:
            delta = g[outcome].iloc[-1] - g[outcome].iloc[0]
            improvement = -delta if improvement_is_decrease else delta
            chg.append({"circuit": u, "observed_improvement": improvement})
    chg = pd.DataFrame(chg).set_index("circuit")["observed_improvement"]
    tab = pd.concat([strict, chg], axis=1).dropna()

    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd else s * 0
    tab["strictness_z"] = z(tab["strictness"])
    tab["improvement_z"] = z(tab["observed_improvement"])
    tab["law_in_action_gap"] = (tab["strictness_z"] - tab["improvement_z"]).round(3)
    tab = tab.round(4).reset_index().rename(columns={"index": "circuit"})
    tab.to_csv(config.TABLES / "law_in_action_gap.csv", index=False)
    return tab.sort_values("law_in_action_gap", ascending=False)


def write_summary(report: dict) -> str:
    """Write only the run facts needed to reproduce the paper boundary."""
    run = report.get("Run", {})
    twfe = report.get("Step 7 TWFE (FEII -> segregation)", {})
    lines = ["# FHA federal-courts pipeline -- run summary", ""]
    if run:
        lines.extend([
            f"- source: {run.get('source')}",
            f"- cases: {run.get('cases')}",
            f"- FEII cells: {run.get('FEII cells')}",
            f"- housing panel rows: {run.get('panel rows')}",
        ])
    if isinstance(twfe, dict) and twfe.get("note"):
        lines.append(f"- real-data inference: not estimated ({twfe['note']})")
    elif isinstance(twfe, dict):
        lines.append("- estimator check: completed on a synthetic or feasible panel")
    lines.append("")
    out = config.OUTPUTS / "SUMMARY.md"
    out.write_text("\n".join(lines))
    return str(out)

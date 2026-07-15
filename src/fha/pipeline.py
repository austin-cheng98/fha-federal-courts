"""
End-to-end pipeline orchestrator: run(source='synthetic'|'live'|'existing',
real_housing=...).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config, feii as feii_mod, doctrine, econometrics as ec
from . import reports, robustness, synth
from .classify import build_clean_corpus
from .courtlistener import CourtListenerClient
from .extract import extract_corpus
from .reference import FHA_SHOCKS

OUTCOME = "dissimilarity_index"
# Event-study shock = the SCOTUS shock in FHA_SHOCKS (Inclusive Communities, 2015).
SHOCK_YEAR = next((s["year"] for s in FHA_SHOCKS if s.get("kind") == "scotus"), 2015)
MIN_CASES_FOR_INFERENCE = 8


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def collect_corpus(source: str, live_kwargs: dict | None = None) -> list[dict]:
    """Get raw case records."""
    if source == "synthetic":
        synth.generate(**(live_kwargs or {}))
        return _load_jsonl(config.RAW / "synthetic_corpus.jsonl")
    if source == "live":
        lk = {"query": '"fair housing act"', "max_pages": 10,
              "fetch_text": True, **(live_kwargs or {})}
        client = CourtListenerClient()
        recs = [r.__dict__ for r in client.harvest(**lk)]
        out = config.RAW / "live_corpus.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return recs
    if source == "existing":
        p = (live_kwargs or {}).get("path", config.RAW / "live_corpus.jsonl")
        return _load_jsonl(p)
    raise ValueError(f"unknown source: {source}")


def run(source: str = "synthetic", live_kwargs: dict | None = None,
        real_housing: bool = False, embedding_backend: str | None = None) -> dict:
    report: dict = {"source": source}

    # collect + identify (clean FHA-only corpus)
    raw = collect_corpus(source, live_kwargs)
    clean, id_report = build_clean_corpus(
        raw, use_ml=False, require_nos443=(source != "synthetic"))
    if source == "synthetic":          # synthetic corpus is all FHA by construction
        clean = raw
        id_report["n_kept"] = len(clean)
    report["step2_identification"] = id_report

    # extract structured variables
    if not clean:
        raise ValueError(f"Step 2 produced 0 FHA cases from {len(raw)} inputs; "
                         f"identification report={id_report}")
    feat = extract_corpus(clean)
    if feat.empty or "circuit" not in feat.columns:
        raise ValueError("Step 3 extraction produced no usable feature rows")
    feat = feat[feat["circuit"].notna() & feat["year"].notna()].reset_index(drop=True)
    feat.to_csv(config.PROCESSED / "case_features.csv", index=False)
    report["step3_n_cases"] = len(feat)
    if len(feat) < MIN_CASES_FOR_INFERENCE:
        report["warning"] = (
            f"only {len(feat)} cases with circuit+year; Steps 5-7 econometrics "
            f"are underpowered/degenerate and the FE regressions will return a "
            f"'note' instead of estimates. Harvest more cases (scripts/collect_live.py).")

    # doctrinal embeddings + clustering + map
    texts = [r.get("text", "") for r in clean if r.get("circuit") and r.get("year")]
    step4 = doctrine.run_step4(feat, texts, backend=embedding_backend)
    feat = step4["features"]
    step4["doctrine_map"].to_csv(config.PROCESSED / "doctrine_map.csv", index=False)
    # regime transitions + cross-circuit divergence (doctrinal-map edges)
    step4["transitions"].to_csv(config.PROCESSED / "doctrine_transitions.csv", index=False)
    step4["divergence"].to_csv(config.PROCESSED / "doctrine_divergence.csv", index=False)
    report["step4_cluster_info"] = step4["cluster_info"]

    # FEII
    panel_feii = feii_mod.aggregate(feat, unit="circuit")
    panel_feii.to_csv(config.PROCESSED / "feii_panel.csv", index=False)
    report["step5_feii_cells"] = len(panel_feii)

    # housing outcomes
    from . import housing as housing_mod
    hp = housing_mod.load_housing_panel(real=real_housing)
    report["step6_housing_rows"] = len(hp)

    # identification
    panel = ec.build_panel(panel_feii, hp)
    panel.to_csv(config.PROCESSED / "analysis_panel.csv", index=False)
    twfe_res = ec.twfe(panel, y=OUTCOME, x="FEII")
    # A thin real panel is a feasibility result, not an estimand. Keep the
    # seeded synthetic estimator check, but do not manufacture downstream
    # event-study or DiD outputs when the FE guard returns a note.
    if "note" not in twfe_res:
        report["step7_twfe_wild_bootstrap"] = ec.wild_cluster_bootstrap(
            panel, y=OUTCOME, x="FEII", B=999)
        es = ec.event_study(panel, y=OUTCOME, shock_year=SHOCK_YEAR)
        did2 = ec.did_2x2(panel, y=OUTCOME, shock_year=SHOCK_YEAR)
    else:
        es = pd.DataFrame(columns=["k", "coef", "se", "ci_low", "ci_high"])
        es.attrs["note"] = "not estimated: housing panel failed the FE feasibility guard"
        did2 = {"model": "did_2x2",
                "note": "not estimated: housing panel failed the FE feasibility guard"}
    cc = ec.circuit_compare(feat, panel, y=OUTCOME)
    report["step7_twfe"] = twfe_res
    report["step7_did_2x2"] = did2
    report["step7_circuit_compare_corr"] = {
        "strictness_vs_outcome": cc.attrs.get("corr_strictness_outcome"),
        "FEII_vs_outcome": cc.attrs.get("corr_FEII_outcome")}

    # outputs
    reports.doctrinal_map_outputs(step4["doctrine_map"], step4["divergence"])
    reports.effectiveness_outputs(twfe_res, es, did2)
    gap = reports.law_in_action_gap(feat, panel, outcome=OUTCOME)
    report["step9_law_in_action_gap"] = gap

    # robustness
    report["step10_robustness"] = robustness.run_all(feat, panel, hp, y=OUTCOME)

    # narrative summary
    reports.write_summary({
        "Run": {"source": source, "cases": len(feat),
                "FEII cells": len(panel_feii), "panel rows": len(panel)},
        "Step 2 identification": id_report,
        "Step 4 regimes": step4["cluster_info"]["sizes"],
        "Step 7 TWFE (FEII -> segregation)": twfe_res,
        "Step 7 DiD 2x2": did2,
        "Step 9 law-in-action gap (top)": gap.head(6),
    })
    report["outputs_dir"] = str(config.OUTPUTS)
    return report


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    rep = run(source=src)
    print(json.dumps({k: (v if not isinstance(v, pd.DataFrame) else "<df>")
                      for k, v in rep.items()}, indent=2, default=str)[:2000])

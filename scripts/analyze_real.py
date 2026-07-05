#!/usr/bin/env python3
"""
Run the analysis on the real corpus and the real Census/HMDA panel; print a
summary and write outputs/ tables and figures.

  python scripts/analyze_real.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from fha import config, doctrine, feii, econometrics as ec, reports  # noqa: E402
from fha.classify import build_clean_corpus  # noqa: E402
from fha.extract import extract_corpus  # noqa: E402

OUTCOME = "dissimilarity_index"
SEICSHNAYDRE_WINRATE = 0.195   # 18/92, external benchmark


def main():
    # Prefer a freshly harvested full-text corpus; fall back to the shipped
    # census corpus so the analysis runs on a clean checkout.
    corpus = config.RAW / "fha_text_corpus.jsonl"
    if not corpus.exists():
        corpus = config.PROCESSED / "paper_corpus.jsonl"
    if not corpus.exists():
        sys.exit("No corpus found. Use the shipped data/processed/paper_corpus.jsonl, "
                 "or harvest one first with scripts/fetch_bulk_text.py.")
    recs = [json.loads(l) for l in corpus.open(encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("text")]
    print(f"=== FHA federal-courts analysis — REAL corpus ({corpus.name}) ===")
    print(f"full-text federal FHA opinions: {len(recs)} "
          f"(Seicshnaydre 2013 = 92; pharma-ML = 698)")

    clean, idrep = build_clean_corpus(recs, use_ml=False)
    feat = extract_corpus(clean)
    feat = feat[feat.circuit.notna() & feat.year.notna()].reset_index(drop=True)
    print(f"confirmed FHA-substantive: {len(feat)}")
    feat.to_csv(config.PROCESSED / "real_case_features.csv", index=False)

    # doctrine
    texts = [r.get("text", "") for r in clean
             if r.get("circuit") and r.get("year")]
    s4 = doctrine.run_step4(feat, texts[:len(feat)])
    feat = s4["features"]

    # FEII
    pf = feii.aggregate(feat, unit="circuit")

    # REAL housing panel (Census dissimilarity + HMDA), pulled earlier
    hp_path = config.EXTERNAL / "housing_panel.csv"
    have_housing = hp_path.exists()
    econ = None
    if have_housing:
        hp = pd.read_csv(hp_path)
        panel = ec.build_panel(pf, hp)
        if len(panel) >= 12:
            econ = ec.twfe(panel, y=OUTCOME, x="FEII")
            if "note" not in econ:
                econ["wild"] = ec.wild_cluster_bootstrap(panel, y=OUTCOME, x="FEII", B=499)

    # ---- report ----
    print("\n--- CLAIM MIX (share of cases) ---")
    for c in [x for x in feat if x.startswith("claim_")]:
        print(f"  {c.replace('claim_',''):24} {feat[c].mean():5.0%}")

    pw = feat.plaintiff_win.dropna()
    print(f"\n--- OUTCOMES ---")
    print(f"  plaintiff win rate: {pw.mean():.0%} (n={len(pw)} clear holdings) "
          f"| Seicshnaydre benchmark: {SEICSHNAYDRE_WINRATE:.0%}")
    print(f"  burden frameworks: {feat.burden_framework.value_counts().to_dict()}")
    print(f"  remedies: injunction {feat.remedy_injunction.mean():.0%}, "
          f"damages {feat.remedy_damages.mean():.0%}")

    print(f"\n--- DOCTRINAL REGIMES (clustered on opinion text) ---")
    print(f"  {s4['cluster_info']['sizes']}")
    print(f"  mean strictness/regime: {s4['cluster_info']['mean_strictness_by_regime']}")

    print(f"\n--- CIRCUIT-LEVEL DOCTRINAL VARIATION (core deliverable) ---")
    g = (feat.groupby("circuit")
         .agg(n=("cluster_id", "count"), strictness=("doctrinal_strictness", "mean"),
              impact=("claim_disparate_impact", "mean"),
              win=("plaintiff_win", lambda s: s.dropna().mean())).round(2))
    print(g.sort_values("n", ascending=False).to_string())

    print(f"\n--- ENFORCEMENT INTENSITY (FEII) ---")
    print(f"  circuits span FEII {pf.FEII.min():.2f} to {pf.FEII.max():.2f}")

    print(f"\n--- HOUSING-OUTCOME LINK (Step 7) ---")
    if econ is None:
        print("  panel too thin to estimate (expected at this N / G=12) — "
              "descriptive results above are the substantive findings")
    elif "note" in econ:
        print(f"  {econ['note']}")
    else:
        wp = econ.get("wild", {}).get("wild_p")
        print(f"  TWFE FEII->segregation: coef={econ['coef']:+.4f} "
              f"t(G-1) p={econ['p']:.3f}" + (f", wild-bootstrap p={wp:.3f}" if wp else ""))
        print("  (interpret with caution — circuit-level inference is low-powered)")

    # outputs
    reports.doctrinal_map_outputs(s4["doctrine_map"], s4["divergence"])
    s4["doctrine_map"].to_csv(config.PROCESSED / "real_doctrine_map.csv", index=False)
    print(f"\noutputs written to {config.OUTPUTS}/ (doctrinal map csv + figure)")


if __name__ == "__main__":
    main()

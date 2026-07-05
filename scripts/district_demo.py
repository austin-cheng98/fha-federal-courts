#!/usr/bin/env python3
"""
Compare circuit (~12 clusters) vs district (~90) inference on the synthetic
DGP: more clusters versus noisier per-cell FEII.

  python scripts/district_demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from fha import synth, feii, econometrics as ec  # noqa: E402
from fha.extract import extract_corpus  # noqa: E402

OUTCOME = "dissimilarity_index"


def _fit(panel, label):
    r = ec.twfe(panel, y=OUTCOME, x="FEII")
    wb = ec.wild_cluster_bootstrap(panel, y=OUTCOME, x="FEII", B=999)
    print(f"  {label:9s}: clusters={r.get('n_clusters'):>3}  "
          f"coef={r['coef']:+.4f}  t(G-1) p={r['p']:.4f}  "
          f"wild-bootstrap p={wb['wild_p']:.4f}")
    return r, wb


def main():
    truth = synth.generate_district()
    print(f"synthetic DGP: BETA_TRUE={truth['BETA_TRUE']}, "
          f"{truth['n_districts']} districts, {truth['n_cases']} cases\n")
    recs = [json.loads(l) for l in open(truth["corpus"])]
    feat = extract_corpus(recs)
    hp = pd.read_csv(truth["housing_panel"])

    # district x year  (~90 clusters)
    pf_d = feii.aggregate(feat, unit="court_id")
    panel_d = ec.build_panel(pf_d, hp, unit_right="court_id")

    # same cases + housing aggregated up to circuit x year  (~12 clusters)
    pf_c = feii.aggregate(feat, unit="circuit")
    hp_c = hp.groupby(["circuit", "year"])[OUTCOME].mean().reset_index()
    panel_c = ec.build_panel(pf_c, hp_c, unit_right="circuit")

    cc = len(recs) / max(len(panel_c), 1)
    dc = len(recs) / max(len(panel_d), 1)
    print(f"TWFE  segregation ~ FEII + unit FE + year FE  (true effect < 0):")
    print(f"  (circuit ~{cc:.0f} cases/cell, district ~{dc:.0f} cases/cell)")
    _fit(panel_c, "CIRCUIT")
    _fit(panel_d, "DISTRICT")
    print("\n=> More clusters does NOT automatically mean more power: the finer")
    print("   district unit splits the same corpus into noisier FEII cells, and")
    print("   the attenuation can outweigh the cluster-count gain. Pick the unit")
    print("   that balances cases/cell against cluster count for YOUR corpus.")


if __name__ == "__main__":
    main()

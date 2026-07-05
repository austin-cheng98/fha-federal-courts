#!/usr/bin/env python3
"""
Validation and robustness suite. Writes outputs/paper/validation/:
negation_sensitivity.json, feii_leave_one_out.csv, era_stability.csv,
regime_top_terms.csv, circuit_casemix.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fha import config, feii, econometrics as ec  # noqa: E402
from fha.classify import build_clean_corpus       # noqa: E402
from fha.extract import extract_corpus            # noqa: E402

OUT = config.OUTPUTS / "paper" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

CLAIMS = ["claim_disparate_treatment", "claim_disparate_impact",
          "claim_zoning_exclusionary", "claim_refusal_rent_sell",
          "claim_reasonable_accommodation"]


def load_clean():
    recs = [json.loads(l) for l in
            (config.PROCESSED / "paper_corpus.jsonl").open() if l.strip()]
    recs = [r for r in recs if r.get("text")]
    clean, _ = build_clean_corpus(recs, use_ml=False)
    return clean


def features(clean, negation):
    feat = extract_corpus(clean, negation=negation)
    return feat[feat.circuit.notna() & feat.year.notna()].reset_index(drop=True)


def summarize(feat):
    pw = feat.plaintiff_win.dropna()
    return {
        "n": len(feat),
        "claim_shares": {c.replace("claim_", ""): round(float(feat[c].mean()), 4)
                         for c in CLAIMS},
        "framework": feat.burden_framework.value_counts(normalize=True)
                        .round(4).to_dict(),
        "strictness_mean": round(float(feat.doctrinal_strictness.mean()), 4),
        "plaintiff_win": round(float(pw.mean()), 4),
        "n_holdings": int(len(pw)),
    }


def run_twfe(feat, weights=None):
    feat = feat.copy()
    feat["year"] = feat["year"].astype(int)
    feat["circuit"] = feat["circuit"].astype(str)
    pf = feii.aggregate(feat, unit="circuit", weights=weights)
    pf["unit"] = pf["unit"].astype(str)
    pf["year"] = pf["year"].astype(int)
    hp = pd.read_csv(config.EXTERNAL / "housing_panel.csv")
    hp["circuit"] = hp["circuit"].astype(str)
    hp["year"] = hp["year"].astype(int)
    panel = ec.build_panel(pf, hp)
    res = ec.twfe(panel, y="dissimilarity_index", x="FEII")
    if "note" in res:
        return {"note": res["note"]}
    w = ec.wild_cluster_bootstrap(panel, y="dissimilarity_index", x="FEII")
    return {"coef": round(res["coef"], 4), "se": round(res["se"], 4),
            "p": round(res["p"], 4), "wild_p": round(w["wild_p"], 4),
            "n": res["n"], "clusters": res["n_clusters"]}


def main():
    clean = load_clean()

    # ---- 1. negation sensitivity ------------------------------------------
    base = features(clean, negation=False)
    neg = features(clean, negation=True)
    sens = {"baseline": summarize(base), "negation": summarize(neg)}
    sens["baseline"]["twfe"] = run_twfe(base)
    sens["negation"]["twfe"] = run_twfe(neg)
    # max absolute movement across the headline quantities
    d_claims = {k: round(sens["negation"]["claim_shares"][k]
                         - sens["baseline"]["claim_shares"][k], 4)
                for k in sens["baseline"]["claim_shares"]}
    sens["delta_claim_shares_pp"] = d_claims
    sens["delta_win_rate_pp"] = round(
        sens["negation"]["plaintiff_win"] - sens["baseline"]["plaintiff_win"], 4)
    sens["delta_strictness"] = round(
        sens["negation"]["strictness_mean"] - sens["baseline"]["strictness_mean"], 4)
    json.dump(sens, (OUT / "negation_sensitivity.json").open("w"), indent=1)
    print("== negation sensitivity ==")
    print(" claim-share deltas (pp):", d_claims)
    print(" win rate:", sens["baseline"]["plaintiff_win"], "->",
          sens["negation"]["plaintiff_win"],
          f"(n {sens['baseline']['n_holdings']} -> {sens['negation']['n_holdings']})")
    print(" strictness:", sens["baseline"]["strictness_mean"], "->",
          sens["negation"]["strictness_mean"])
    print(" TWFE:", sens["baseline"]["twfe"], "->", sens["negation"]["twfe"])

    # ---- 2. FEII leave-one-out --------------------------------------------
    rows = [{"spec": "baseline (filings+success+remedy)", **run_twfe(base)}]
    for drop in ("filings", "plaintiff_success", "remedy_severity"):
        w = {c: 0.25 for c in ("filings", "docket_share",
                               "plaintiff_success", "remedy_severity")}
        w[drop] = 0.0
        rows.append({"spec": f"drop {drop}", **run_twfe(base, weights=w)})
    loo = pd.DataFrame(rows)
    loo.to_csv(OUT / "feii_leave_one_out.csv", index=False)
    print("\n== FEII leave-one-out TWFE ==")
    print(loo.to_string(index=False))

    # ---- 3. era stability --------------------------------------------------
    base = base.copy()
    base["era"] = pd.cut(base.year, [0, 2014, 2019, 3000],
                         labels=["pre-2015", "2015-2019", "2020+"])
    recs_era = []
    for era, g in base.groupby("era", observed=True):
        pw = g.plaintiff_win.dropna()
        recs_era.append({
            "era": era, "n": len(g),
            **{c.replace("claim_", ""): round(float(g[c].mean()), 3) for c in CLAIMS},
            "strictness": round(float(g.doctrinal_strictness.mean()), 3),
            "win_rate": round(float(pw.mean()), 3) if len(pw) else None,
            "n_holdings": int(len(pw)),
        })
    era_df = pd.DataFrame(recs_era)
    era_df.to_csv(OUT / "era_stability.csv", index=False)
    print("\n== era stability ==")
    print(era_df.to_string(index=False))

    # ---- 4. regime top terms ----------------------------------------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans

    texts = [r.get("text", "") for r in clean
             if r.get("circuit") and r.get("year")][:len(base)]
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                          max_df=0.9, stop_words="english", max_features=50000)
    X = vec.fit_transform(texts)
    k = min(100, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, random_state=0)
    emb = normalize(svd.fit_transform(X))
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(emb)
    strict = base.doctrinal_strictness.to_numpy()[:len(texts)]
    order = np.argsort([strict[km.labels_ == c].mean() for c in range(3)])
    names = {order[0]: "weak", order[1]: "moderate", order[2]: "strict"}
    vocab = np.array(vec.get_feature_names_out())
    tt = []
    for c in range(3):
        loading = km.cluster_centers_[c] @ svd.components_
        top = vocab[np.argsort(loading)[::-1][:15]]
        tt.append({"regime": names[c], "n": int((km.labels_ == c).sum()),
                   "top_terms": ", ".join(top)})
    tt_df = pd.DataFrame(tt).sort_values("regime")
    tt_df.to_csv(OUT / "regime_top_terms.csv", index=False)
    print("\n== regime top terms ==")
    for _, r in tt_df.iterrows():
        print(f" {r.regime} (n={r.n}): {r.top_terms}")

    # ---- 5. case-mix-adjusted circuit strictness ---------------------------
    import statsmodels.formula.api as smf
    df = base.rename(columns={c: c.replace("claim_", "c_") for c in CLAIMS})
    cvars = [c.replace("claim_", "c_") for c in CLAIMS]
    m = smf.ols("doctrinal_strictness ~ C(circuit) + " + " + ".join(cvars),
                data=df).fit()
    raw = base.groupby("circuit").doctrinal_strictness.mean()
    base_circ = sorted(base.circuit.unique())[0]
    adj = {base_circ: 0.0}
    for c in sorted(base.circuit.unique()):
        key = f"C(circuit)[T.{c}]"
        if key in m.params:
            adj[c] = m.params[key]
    adj = pd.Series(adj) + m.params["Intercept"] + sum(
        m.params[v] * df[v].mean() for v in cvars)
    ra_only = base[base.claim_reasonable_accommodation == 1] \
        .groupby("circuit").doctrinal_strictness.mean()
    out = pd.DataFrame({"raw": raw.round(3), "case_mix_adjusted": adj.round(3),
                        "ra_only": ra_only.round(3)})
    out["raw_rank"] = out.raw.rank(ascending=False).astype(int)
    out["adj_rank"] = out.case_mix_adjusted.rank(ascending=False).astype(int)
    out.to_csv(OUT / "circuit_casemix.csv")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(out.raw, out.case_mix_adjusted)
    rho_ra, _ = spearmanr(out.raw.reindex(ra_only.index).dropna(),
                          ra_only.dropna())
    print("\n== circuit strictness: raw vs case-mix adjusted ==")
    print(out.to_string())
    print(f" Spearman raw vs adjusted: {rho:.3f} | raw vs RA-only: {rho_ra:.3f}")
    json.dump({"spearman_raw_adj": round(float(rho), 3),
               "spearman_raw_ra": round(float(rho_ra), 3)},
              (OUT / "casemix_summary.json").open("w"), indent=1)


if __name__ == "__main__":
    main()

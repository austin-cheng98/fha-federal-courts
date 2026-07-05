#!/usr/bin/env python3
"""
Generate the paper's figures, tables, and key_numbers.json from the frozen
corpus (data/processed/paper_corpus.jsonl), the population metadata, and the
Census/HMDA housing panel.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from fha import config, doctrine, feii, econometrics as ec
from fha.classify import build_clean_corpus
from fha.extract import extract_corpus

FIG = config.OUTPUTS / "paper" / "figures"
TAB = config.OUTPUTS / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
                 "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                 "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight"})
NAVY, RUST, TEAL, GOLD = "#1f3b57", "#b5482e", "#2a8a7d", "#caa53d"
CLAIM_LABELS = {"disparate_treatment": "Disparate\ntreatment", "disparate_impact": "Disparate\nimpact",
                "zoning_exclusionary": "Zoning /\nexclusionary", "refusal_rent_sell": "Refusal /\nsteering",
                "reasonable_accommodation": "Reasonable\naccommodation"}
K = {}   # key numbers for the prose


def savefig(name):
    plt.savefig(FIG / name); plt.close()
    print("  fig:", name)


# load + extract
recs = [json.loads(l) for l in (config.PROCESSED / "paper_corpus.jsonl").open()
        if l.strip()]
recs = [r for r in recs if r.get("text")]
clean, _ = build_clean_corpus(recs, use_ml=False)
feat = extract_corpus(clean)
feat = feat[feat.circuit.notna() & feat.year.notna()].reset_index(drop=True)
texts = [r.get("text", "") for r in clean if r.get("circuit") and r.get("year")][:len(feat)]
emb = doctrine.embed(texts)
labels, cinfo = doctrine.cluster_regimes(emb, feat.doctrinal_strictness.to_numpy())
feat["regime"] = labels
pop = [json.loads(l) for l in (config.RAW / "bulk_fha_cases.jsonl").open() if l.strip()]

K["n_text"] = len(recs); K["n_substantive"] = len(feat); K["n_population"] = len(pop)
CLAIMS = [c for c in feat if c.startswith("claim_")]
K["claim_shares"] = {c.replace("claim_", ""): round(float(feat[c].mean()), 3) for c in CLAIMS}
pw = feat.plaintiff_win.dropna()
K["plaintiff_win"] = round(float(pw.mean()), 3); K["n_holdings"] = int(len(pw))
K["regimes"] = cinfo["sizes"]; K["regime_strictness"] = cinfo["mean_strictness_by_regime"]
print(f"corpus n={len(recs)} text, {len(feat)} substantive, pop={len(pop)}")

# Fig 1 — corpus over time
yr = feat.groupby("year").size()
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.bar(yr.index, yr.values, color=NAVY, width=0.9)
ax.axvline(2015, color=RUST, ls="--", lw=1.4)
ax.text(2015.3, ax.get_ylim()[1] * 0.85, "Inclusive\nCommunities\n(2015)", color=RUST, fontsize=8)
ax.set_xlabel("year"); ax.set_ylabel("FHA opinions"); ax.set_title("(a) Corpus by filing year")
ax.set_xlim(1988, 2027)
savefig("f1_corpus_time.png")

# Fig 2 — claim distribution
order = sorted(CLAIMS, key=lambda c: feat[c].mean())
vals = [feat[c].mean() for c in order]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.barh([CLAIM_LABELS[c.replace("claim_", "")] for c in order], vals, color=TEAL)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
ax.set_xlabel("share of cases"); ax.set_xlim(0, max(vals) + 0.1)
ax.set_title("(b) FHA claim-type prevalence")
savefig("f2_claims.png")

# Fig 3 — representativeness (sample vs population): circuit + decade
def shares(rows, key):
    from collections import Counter
    c = Counter(key(r) for r in rows); t = sum(c.values())
    return {k: v / t for k, v in c.items()}
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
for ax, (nm, key, od) in zip(axes, [
        ("circuit", lambda r: str(r.get("circuit")),
         ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC"]),
        ("decade", lambda r: f"{(r.get('year') or 0)//10*10}s",
         ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"])]):
    ps = shares(pop, key); ss = shares([r for r in recs], key)
    x = np.arange(len(od)); w = 0.4
    ax.bar(x - w/2, [ps.get(k, 0) for k in od], w, label="population (937)", color=NAVY)
    ax.bar(x + w/2, [ss.get(k, 0) for k in od], w, label="sample", color=GOLD)
    ax.set_xticks(x); ax.set_xticklabels(od, rotation=45 if nm == "circuit" else 0, fontsize=8)
    ax.set_title(f"({'c' if nm=='circuit' else 'd'}) {nm}: sample vs population")
    ax.set_ylabel("share")
axes[0].legend(fontsize=8, frameon=False)
savefig("f3_representativeness.png")

# Fig 4 — doctrinal strictness + disparate-impact recognition by circuit
g = (feat.groupby("circuit").agg(n=("cluster_id", "count"),
     strict=("doctrinal_strictness", "mean"), impact=("claim_disparate_impact", "mean"),
     win=("plaintiff_win", lambda s: s.dropna().mean())).reset_index())
g = g[g.n >= 5].sort_values("strict")
fig, ax = plt.subplots(figsize=(7.4, 3.4))
xx = np.arange(len(g))
ax.bar(xx - 0.2, g.strict, 0.4, label="doctrinal strictness", color=NAVY)
ax.bar(xx + 0.2, g.impact, 0.4, label="disparate-impact share", color=RUST)
ax.set_xticks(xx); ax.set_xticklabels("C" + g.circuit.astype(str), fontsize=9)
ax.set_ylabel("index / share"); ax.legend(fontsize=8, frameon=False)
ax.set_title("(e) Cross-circuit doctrinal variation")
savefig("f4_circuit_strictness.png")

# Fig 5 — doctrinal regimes in embedding space (2D) + strictness
from sklearn.decomposition import PCA
xy = PCA(n_components=2, random_state=0).fit_transform(emb)
cmap = {"strict": RUST, "moderate": GOLD, "weak": TEAL}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.5), gridspec_kw={"width_ratios": [1.4, 1]})
for reg in ["weak", "moderate", "strict"]:
    m = feat.regime == reg
    a1.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.6, color=cmap[reg],
               label=f"{reg} (n={int(m.sum())})")
a1.set_xlabel("doctrinal embedding dim 1"); a1.set_ylabel("dim 2")
a1.legend(fontsize=8, frameon=False); a1.set_title("(f) Doctrinal regimes (text embedding)")
rs = cinfo["mean_strictness_by_regime"]
a2.bar(list(rs.keys()), list(rs.values()), color=[cmap[k] for k in rs])
a2.set_ylabel("mean strictness"); a2.set_title("(g) Regime strictness")
savefig("f5_regimes.png")

# Fig 6 — doctrinal map heatmap (circuit x decade strictness)
feat["decade"] = (feat.year // 10 * 10).astype(int)
piv = feat.pivot_table(index="circuit", columns="decade", values="doctrinal_strictness", aggfunc="mean")
order_c = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC"]
piv = piv.reindex([c for c in order_c if c in piv.index])
fig, ax = plt.subplots(figsize=(6.8, 4.0))
im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=0, vmax=0.7)
ax.set_yticks(range(len(piv.index)), ["C" + str(c) for c in piv.index])
ax.set_xticks(range(len(piv.columns)), [f"{c}s" for c in piv.columns])
ax.set_title("(h) Doctrinal-strictness map (circuit × decade)")
fig.colorbar(im, ax=ax, label="mean strictness", shrink=0.8)
savefig("f6_doctrine_map.png")

# Fig 7 — temporal: disparate-impact recognition over time (Inclusive Communities)
tt = feat[feat.year >= 1995].copy()
tt["period"] = np.where(tt.year < 2015, "pre-2015", "2015+")
roll = tt.groupby("year")["claim_disparate_impact"].mean().rolling(3, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(roll.index, roll.values, color=NAVY, lw=1.8)
ax.axvline(2015, color=RUST, ls="--", lw=1.3)
pre = tt[tt.year < 2015]["claim_disparate_impact"].mean()
post = tt[tt.year >= 2015]["claim_disparate_impact"].mean()
ax.axhline(pre, xmax=0.66, color=TEAL, ls=":", lw=1); ax.axhline(post, xmin=0.66, color=GOLD, ls=":", lw=1)
ax.text(2002, pre + .02, f"pre-2015 mean {pre:.0%}", color=TEAL, fontsize=8)
ax.text(2016, post + .02, f"2015+ mean {post:.0%}", color=GOLD, fontsize=8)
ax.set_ylabel("disparate-impact share (3-yr avg)"); ax.set_xlabel("year")
ax.set_title("(i) Disparate-impact claims around Inclusive Communities")
savefig("f7_temporal.png")
K["impact_pre2015"] = round(float(pre), 3); K["impact_post2015"] = round(float(post), 3)

# Fig 8 — plaintiff win rate by claim type + benchmark
wr = {}
for c in CLAIMS:
    sub = feat[feat[c] == 1]["plaintiff_win"].dropna()
    if len(sub) >= 3:
        wr[c.replace("claim_", "")] = (sub.mean(), len(sub))
fig, ax = plt.subplots(figsize=(6.8, 3.2))
ks = list(wr.keys()); v = [wr[k][0] for k in ks]
ax.bar(ks, v, color=TEAL)
ax.axhline(K["plaintiff_win"], color=NAVY, ls="-", lw=1.2, label=f"overall {K['plaintiff_win']:.0%}")
ax.axhline(0.195, color=RUST, ls="--", lw=1.2, label="Seicshnaydre (2013) 20%")
for i, k in enumerate(ks):
    ax.text(i, v[i] + .01, f"{v[i]:.0%}\n(n={wr[k][1]})", ha="center", fontsize=7)
ax.set_xticklabels([k.replace("_", "\n") for k in ks], fontsize=8)
ax.set_ylabel("plaintiff win rate"); ax.legend(fontsize=8, frameon=False)
ax.set_title("(j) Plaintiff success by claim type")
savefig("f8_winrate.png")

# Fig 9 — FEII by circuit
pf = feii.aggregate(feat, unit="circuit")
fc = pf.groupby("unit")["FEII"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7.2, 3.0))
colors = [RUST if x < 0 else NAVY for x in fc.values]
ax.bar(["C" + str(c) for c in fc.index], fc.values, color=colors)
ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("mean FEII (z-score)")
ax.set_title("(k) Federal Enforcement Intensity Index by circuit")
savefig("f9_feii.png")

# Econometrics + tables
hp = pd.read_csv(config.EXTERNAL / "housing_panel.csv")
panel = ec.build_panel(pf, hp)
econ = ec.twfe(panel, y="dissimilarity_index", x="FEII")
if "note" not in econ:
    wb = ec.wild_cluster_bootstrap(panel, y="dissimilarity_index", x="FEII", B=999)
    econ["wild_p"] = wb["wild_p"]
K["twfe"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in econ.items()
             if k in ("coef", "se", "t", "p", "wild_p", "n", "n_clusters", "note")}

# Table 1 — corpus / representativeness
from collections import Counter
def tvd(a, b):
    keys = set(a) | set(b); return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
tvd_circ = tvd(shares(pop, lambda r: r.get("circuit")), shares(recs, lambda r: r.get("circuit")))
tvd_dec = tvd(shares(pop, lambda r: (r.get("year") or 0)//10), shares(recs, lambda r: (r.get("year") or 0)//10))
import math
N = len(pop); n = len(recs)
moe = 1.96 * math.sqrt(.25 / n) * math.sqrt((N - n) / (N - 1))
K["tvd_circuit"] = round(tvd_circ, 3); K["tvd_decade"] = round(tvd_dec, 3); K["moe"] = round(moe, 3)
pd.DataFrame([
    ["Full-text opinions analyzed", n],
    ["Substantively-FHA (citation+keyword filter)", len(feat)],
    ["Population (federal NOS-443 FHA clusters)", N],
    ["Coverage of population", f"{n/N:.0%}"],
    ["Circuits represented", feat.circuit.nunique()],
    ["Year span", f"{int(feat.year.min())}-{int(feat.year.max())}"],
    ["Margin of error (50/50 proportion, 95%, FPC)", f"±{moe:.1%}"],
    ["Total-variation distance, circuit (sample vs pop)", f"{tvd_circ:.3f}"],
    ["Total-variation distance, decade (sample vs pop)", f"{tvd_dec:.3f}"],
], columns=["quantity", "value"]).to_csv(TAB / "t1_corpus.csv", index=False)

# Table 2 — circuit comparison
g2 = (feat.groupby("circuit").agg(
        n=("cluster_id", "count"), strictness=("doctrinal_strictness", "mean"),
        impact=("claim_disparate_impact", "mean"), accommodation=("claim_reasonable_accommodation", "mean"),
        win=("plaintiff_win", lambda s: s.dropna().mean())).reset_index())
g2 = g2.merge(pf.groupby("unit")["FEII"].mean().rename("FEII").reset_index().rename(columns={"unit": "circuit"}),
              on="circuit", how="left")
g2 = g2.sort_values("strictness", ascending=False).round(3)
g2.to_csv(TAB / "t2_circuits.csv", index=False)

# Table 3 — regime characteristics
reg_tab = feat.groupby("regime").agg(
    n=("cluster_id", "count"), strictness=("doctrinal_strictness", "mean"),
    impact=("claim_disparate_impact", "mean"),
    hud_framework=("reason_hud_burden_shifting", "mean"),
    mcdonnell=("reason_mcdonnell_douglas", "mean"),
    win=("plaintiff_win", lambda s: s.dropna().mean())).round(3)
reg_tab.to_csv(TAB / "t3_regimes.csv")

# Table 4 — regression
pd.DataFrame([K["twfe"]]).to_csv(TAB / "t4_regression.csv", index=False)

(config.OUTPUTS / "paper" / "key_numbers.json").write_text(json.dumps(K, indent=2, default=str))
print("\nKEY NUMBERS:"); print(json.dumps(K, indent=1, default=str))
print(f"\nAll assets in {config.OUTPUTS/'paper'}")

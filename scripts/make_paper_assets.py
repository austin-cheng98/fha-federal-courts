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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from fha import config, doctrine, feii, econometrics as ec
from fha.classify import build_clean_corpus
from fha.extract import extract_corpus

FIG = config.OUTPUTS / "paper" / "figures"
TAB = config.OUTPUTS / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
for stale in ("f1_corpus_time.png", "f3_representativeness.png",
              "f4_circuit_strictness.png", "f8_winrate.png", "f9_feii.png",
              "f8_partial_regression.png", "f9_component_exclusions.png",
              "f10_diagnostic.png", "f11_housing_feasibility.png"):
    (FIG / stale).unlink(missing_ok=True)
(TAB / "t4_regression.csv").unlink(missing_ok=True)

rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.45,
    "grid.linewidth": 0.55,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.7,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
# The reference paper uses a restrained blue/orange palette with gray
# scaffolding. Keep these colors stable across the main comparison figures.
BLUE, ORANGE, MIDGRAY, INK = "#8ebad9", "#ffbe86", "#b7b7b7", "#222222"
NAVY, RUST, TEAL, GOLD = BLUE, ORANGE, "#9fc6d9", "#e7b47e"
INDICATOR_LABELS = {
    "disparate_treatment": "Proof: disparate\ntreatment",
    "disparate_impact": "Proof: disparate\nimpact",
    "reasonable_accommodation": "Duty: reasonable\naccommodation",
    "refusal_rent_sell": "Conduct: refusal /\nsteering",
    "zoning_exclusionary": "Conduct: zoning /\nland use",
}
K = {}   # key numbers for the prose


def savefig(name):
    plt.savefig(FIG / name); plt.close()
    print("  fig:", name)


# Fig 0 — corrected measurement pipeline. The housing branch ends in a
# feasibility diagnostic; the real frozen merge does not support TWFE.
fig, ax = plt.subplots(figsize=(5.8, 6.51))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
def pbox(x, y, w, h, label, fc, ec=BLUE, fs=7.2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                           facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs,
            color=INK, linespacing=1.15)
def arrow(x1, y1, x2, y2, color=TEAL):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=1.0, color=color))
# Leave a generous top margin because the manuscript preserves a portrait
# image slot from the earlier pipeline figure.
ax.text(0.45, 9.25, "FHA signal measurement pipeline", fontsize=11, weight="bold", color=INK)
ax.text(0.45, 8.92, "district-court corpus → auditable indicators → feasibility check", fontsize=7.2, color="0.35")
pbox(0.65, 8.00, 3.8, 0.55, "CourtListener district opinions\nNOS-443 snapshot", "#e7edf3")
pbox(5.95, 8.00, 3.45, 0.55, "ACS 2012 / 2017 / 2022\nnon-overlapping vintages", "#efe8f3", ec="#75508b")
arrow(2.55, 7.96, 2.55, 7.48, NAVY); arrow(7.675, 7.96, 7.675, 5.66, "#75508b")
pbox(1.15, 6.72, 4.7, 0.62, "Canonical corpus\n757 clusters · 751 full text\n417 substantive · district only", "#dfe8f0", fs=6.5)
arrow(3.5, 6.68, 3.5, 6.25, NAVY)
pbox(1.15, 5.46, 4.7, 0.62, "Structured record\nproof · duty · conduct\nframeworks · remedies · outcome cues", "#dff0ed", ec=TEAL, fs=6.5)
arrow(3.5, 5.42, 3.5, 4.98, TEAL)
pbox(1.15, 4.20, 4.7, 0.62, "Text regimes\nTF-IDF → LSA → KMeans (k=3)\ndescriptive posture / disposition", "#e8f1ee", ec=TEAL, fs=6.5)
arrow(3.5, 4.16, 3.5, 3.72, GOLD)
pbox(1.15, 2.94, 4.7, 0.62, "FEII\ncircuit-year descriptive index\nvolume + cues + remedies", "#f7f0dc", ec=GOLD, fs=6.5)
arrow(3.5, 2.90, 3.5, 2.42, "#6d7b84")
pbox(0.65, 1.56, 3.8, 0.62, "Validation diagnostic\n93-case overlap · 30 double-coded\nchoice-based; unweighted", "#edf0f2", ec="#6d7b84", fs=6.2)
arrow(7.675, 5.62, 7.675, 5.16, "#75508b")
pbox(5.95, 4.40, 3.45, 0.62, "Housing merge\n36 input rows · 8 matched cells\nall matched cells are 2022", "#f3e9f0", ec="#75508b", fs=6.2)
arrow(7.675, 4.36, 7.675, 3.90, "#75508b")
pbox(5.95, 3.14, 3.45, 0.62, "Feasibility result\none year\nno real-data TWFE coefficient", "#f8e8e3", ec=RUST, fs=6.2)
ax.text(5.95, 2.02, "Evidence trail retained: rule text,\nmissingness, denominators, and seeds.", fontsize=6.9, color="0.35")
savefig("f0_pipeline.png")


# load + extract
recs = [json.loads(l) for l in (config.PROCESSED / "paper_corpus.jsonl").open()
        if l.strip()]
recs = [r for r in recs if r.get("text")]
clean, _ = build_clean_corpus(recs, use_ml=False, require_nos443=True)
feat = extract_corpus(clean)
feat = feat[feat.circuit.notna() & feat.year.notna()].reset_index(drop=True)
texts = [r.get("text", "") for r in clean if r.get("circuit") and r.get("year")][:len(feat)]
emb = doctrine.embed(texts)
labels, cinfo = doctrine.cluster_regimes(emb, feat.legal_signal_score.to_numpy())
feat["regime"] = labels
pop = [json.loads(l) for l in (config.RAW / "bulk_fha_cases.jsonl").open() if l.strip()]

K["n_text"] = len(recs); K["n_substantive"] = len(feat); K["n_population"] = len(pop)
CLAIMS = [c for c in feat if c.startswith("claim_") and not c.endswith("_spans")]
K["claim_shares"] = {c.replace("claim_", ""): round(float(feat[c].mean()), 3) for c in CLAIMS}
cue = feat.outcome_cue.dropna()
K["outcome_cue_rate"] = round(float(cue.mean()), 3); K["n_outcome_cues"] = int(len(cue))
K["regimes"] = cinfo["sizes"]; K["regime_legal_signal"] = cinfo["regime_legal_signal"]
print(f"corpus n={len(recs)} text, {len(feat)} substantive, pop={len(pop)}")

# Figure 3 — textual indicators separated by analytic layer.
layer_order = ["claim_disparate_treatment", "claim_reasonable_accommodation",
               "claim_disparate_impact", "claim_refusal_rent_sell",
               "claim_zoning_exclusionary"]
order = [c for c in layer_order if c in feat]
vals = [feat[c].mean() for c in order]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
colors = [TEAL if c.startswith("claim_disparate") or c.endswith("accommodation") else GOLD
          for c in order]
labels = [INDICATOR_LABELS[c.replace("claim_", "")] for c in order]
ax.barh(labels, vals, color=colors, edgecolor=INK, linewidth=0.3)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
ax.set_xlabel("share of rule-positive opinions"); ax.set_xlim(0, max(vals) + 0.1)
ax.set_title("(b) FHA textual indicators by analytic layer")
savefig("f2_claims.png")

# Figure 1 — canonical population versus recovered full text by circuit.
def shares(rows, key):
    from collections import Counter
    c = Counter(key(r) for r in rows); t = sum(c.values())
    return {k: v / t for k, v in c.items()}
# Paired-dot display used by the manuscript so the coverage comparison is not
# another bar chart.
order = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC"]
pc = shares(pop, lambda r: str(r.get("circuit")))
tc = shares(recs, lambda r: str(r.get("circuit")))
fig, ax = plt.subplots(figsize=(7.2, 3.35))
y = np.arange(len(order))
pv = np.array([pc.get(k, 0) for k in order])
tv = np.array([tc.get(k, 0) for k in order])
for yi, pval, tval in zip(y, pv, tv):
    ax.plot([pval, tval], [yi, yi], color=MIDGRAY, lw=1.1, zorder=1)
ax.scatter(pv, y, s=38, color=BLUE, edgecolor=INK, linewidth=0.35,
           label=f"population (n={len(pop)})", zorder=3)
ax.scatter(tv, y, s=38, color=ORANGE, edgecolor=INK, linewidth=0.35,
           label=f"full text (n={len(recs)})", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("share of opinion clusters"); ax.set_ylabel("circuit")
ax.grid(axis="x", color="#b0b0b0", alpha=0.45, linewidth=0.55)
ax.legend(frameon=True, facecolor="white", edgecolor="#777777",
          loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
savefig("f1_corpus_match.png")

# Figure 4 — textual regimes in embedding space (2D) + legal-signal breadth.
from sklearn.decomposition import PCA, TruncatedSVD
xy = PCA(n_components=2, random_state=0).fit_transform(emb)
cmap = {"strict": RUST, "moderate": GOLD, "weak": TEAL}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.5), gridspec_kw={"width_ratios": [1.4, 1]})
for reg in ["weak", "moderate", "strict"]:
    m = feat.regime == reg
    a1.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.6, color=cmap[reg],
               label=f"{reg} (n={int(m.sum())})")
a1.set_xlabel("doctrinal embedding dim 1"); a1.set_ylabel("dim 2")
a1.legend(fontsize=8, frameon=False)
rs = cinfo["regime_legal_signal"]
rx = np.arange(len(rs))
a2.scatter(rx, list(rs.values()), s=62, color=[cmap[k] for k in rs],
           edgecolor=INK, linewidth=0.35, zorder=3)
a2.axhline(0, color=MIDGRAY, lw=0.7)
for xi, (k, v) in enumerate(rs.items()):
    a2.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
a2.set_xticks(rx); a2.set_xticklabels(list(rs.keys()))
a2.set_ylim(0, max(rs.values()) + 0.08)
a2.set_ylabel("mean legal-signal breadth")
savefig("f5_regimes.png")

# Fig 5b — vocabulary associated with the three regimes. The original
# vocabulary display used three panels of horizontal bars; replace those bars
# with ranked loading dots while retaining the same TF-IDF/LSA/KMeans logic.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                      max_df=0.9, stop_words="english", max_features=50000)
X = vec.fit_transform(texts)
k_svd = min(100, X.shape[1] - 1, X.shape[0] - 1)
svd = TruncatedSVD(n_components=k_svd, random_state=0)
lsa = normalize(svd.fit_transform(X))
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(lsa)
legal_by_cluster = {
    c: float(feat.loc[km.labels_ == c, "legal_signal_score"].mean())
    for c in range(3)
}
ordered_clusters = sorted(legal_by_cluster, key=legal_by_cluster.get)
regime_for_cluster = {c: ["weak", "moderate", "strict"][i]
                      for i, c in enumerate(ordered_clusters)}
loadings = km.cluster_centers_ @ svd.components_
vocab = np.array(vec.get_feature_names_out())
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), sharex=False)
for ax, cluster in zip(axes, ordered_clusters):
    regime = regime_for_cluster[cluster]
    top = np.argsort(loadings[cluster])[::-1][:8]
    terms = vocab[top][::-1]
    vals = loadings[cluster][top][::-1]
    yy = np.arange(len(terms))
    ax.scatter(vals, yy, s=30, color=cmap[regime], edgecolor=INK,
               linewidth=0.3, zorder=3)
    ax.axvline(0, color=MIDGRAY, lw=0.7)
    ax.set_yticks(yy); ax.set_yticklabels(terms, fontsize=8)
    ax.set_title(f"{regime} regime", color=cmap[regime], weight="bold")
    ax.set_xlabel("cluster loading")
    ax.grid(axis="x", color="#b0b0b0", alpha=0.35, linewidth=0.5)
fig.suptitle("Top distinguishing terms (TF-IDF/LSA loading)", y=1.02)
fig.tight_layout()
savefig("f5_terms.png")

# Fig 6 — doctrinal map heatmap (circuit x decade strictness)
feat["decade"] = (feat.year // 10 * 10).astype(int)
piv = feat.pivot_table(index="circuit", columns="decade", values="legal_signal_score", aggfunc="mean")
order_c = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC"]
piv = piv.reindex([c for c in order_c if c in piv.index])
fig, ax = plt.subplots(figsize=(6.8, 4.0))
im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=0, vmax=0.7)
ax.set_yticks(range(len(piv.index)), ["C" + str(c) for c in piv.index])
ax.set_xticks(range(len(piv.columns)), [f"{c}s" for c in piv.columns])
ax.set_title("(h) Legal-signal breadth map (circuit × decade)")
fig.colorbar(im, ax=ax, label="mean legal-signal breadth", shrink=0.8)
savefig("f6_doctrine_map.png")

# Fig 7 — temporal: disparate-impact recognition over time (Inclusive Communities)
tt = feat[feat.year >= 1995].copy()
tt["period"] = np.where(tt.year < 2015, "pre-2015", "2015+")
roll = tt.groupby("year")["claim_disparate_impact"].mean().rolling(3, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(roll.index, roll.values, color=BLUE, lw=1.8)
ax.axvline(2015, color=ORANGE, ls="--", lw=1.3)
pre = tt[tt.year < 2015]["claim_disparate_impact"].mean()
post = tt[tt.year >= 2015]["claim_disparate_impact"].mean()
ax.axhline(pre, xmax=0.66, color=ORANGE, ls=":", lw=1)
ax.axhline(post, xmin=0.66, color=BLUE, ls=":", lw=1)
# Keep the mean labels in the margin with a light backing patch. This avoids
# placing text on top of the time-series path, especially in the sparse years.
label_box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
ax.text(0.02, 0.96, f"pre-2015 mean {pre:.0%}", transform=ax.transAxes,
        color=ORANGE, fontsize=8, ha="left", va="top", bbox=label_box)
ax.text(0.98, 0.96, f"2015+ mean {post:.0%}", transform=ax.transAxes,
        color=BLUE, fontsize=8, ha="right", va="top", bbox=label_box)
ax.set_ylabel("disparate-impact share (3-yr avg)"); ax.set_xlabel("year")
ax.set_title("(i) Disparate-impact claims around Inclusive Communities")
ax.set_ylim(0, min(1.0, max(float(roll.max()), float(pre), float(post)) + 0.12))
savefig("f7_temporal.png")
K["impact_pre2015"] = round(float(pre), 3); K["impact_post2015"] = round(float(post), 3)

# FEII is used for the descriptive housing-link figure below; no unused
# circuit bar chart is emitted.
pf = feii.aggregate(feat, unit="circuit")

# Econometrics + tables
hp = pd.read_csv(config.EXTERNAL / "housing_panel.csv")
panel = ec.build_panel(pf, hp)
econ = ec.twfe(panel, y="dissimilarity_index", x="FEII")
if "note" not in econ:
    wb = ec.wild_cluster_bootstrap(panel, y="dissimilarity_index", x="FEII", B=999)
    econ["wild_p"] = wb["wild_p"]

# Fig 8 — descriptive regression plot. The eight matched cells are all from
# 2022, so this is a visual association and not a fixed-effects estimate.
matched = panel.dropna(subset=["FEII", "dissimilarity_index"]).copy()
fig, ax = plt.subplots(figsize=(6.8, 3.3))
ax.scatter(matched["FEII"], matched["dissimilarity_index"], s=28,
           color=BLUE, edgecolor=INK, linewidth=0.35, alpha=0.9,
           label=f"matched cells (n={len(matched)})")
if len(matched) >= 2 and matched["FEII"].nunique() > 1:
    slope, intercept = np.polyfit(matched["FEII"], matched["dissimilarity_index"], 1)
    xx = np.linspace(float(matched["FEII"].min()), float(matched["FEII"].max()), 100)
    ax.plot(xx, intercept + slope * xx, color=ORANGE, lw=1.7,
            label="descriptive linear fit")
ax.set_xlabel("FEII (descriptive circuit-year index)")
ax.set_ylabel("Black–White dissimilarity index")
ax.legend(frameon=True, facecolor="white", edgecolor="#777777")
ax.grid(color="#b0b0b0", alpha=0.45, linewidth=0.55)
savefig("f8_partial_regression.png")

# Fig 10 — observed district-year volume diagnostic from the canonical
# corpus. Zero cells are not imputed; the denominator is observed cells.
dcell = (feat.dropna(subset=["court_id", "year"])
         .groupby(["court_id", "year"]).size())
fig, ax = plt.subplots(figsize=(6.8, 3.1))
freq = dcell.value_counts().sort_index()
ax.plot(freq.index.to_numpy(), freq.values, color=TEAL, lw=1.5,
        marker="o", markersize=4.5, markeredgecolor=INK,
        markeredgewidth=0.3)
ax.axvline(float(dcell.mean()), color=RUST, ls="--", lw=1.2,
           label=f"mean observed cell = {dcell.mean():.2f}")
ax.set_xlabel("opinion clusters per observed district-year")
ax.set_ylabel("number of observed cells")
ax.set_title("Observed district-year opinion volume")
ax.legend(frameon=False, fontsize=8)
savefig("f10_district_cells.png")

# Fig 10 — diagnostic validation metrics. Values are the reported, unweighted
# metrics from the 93-case enriched overlap, not a population estimate.
metrics = pd.DataFrame({
    "construct": ["treatment", "impact", "accommodation", "zoning", "refusal / steering"],
    "precision": [0.91, 0.41, 0.83, 0.80, 0.73],
    "recall": [0.72, 0.92, 0.94, 0.80, 0.61],
    "F1": [0.80, 0.56, 0.88, 0.80, 0.67],
})
fig, ax = plt.subplots(figsize=(7.0, 3.5))
x = np.arange(len(metrics))
w = 0.24
for j, col, color in [(0, "precision", BLUE), (1, "recall", ORANGE), (2, "F1", MIDGRAY)]:
    ax.bar(x + (j - 1) * w, metrics[col], w, label=col, color=color,
           edgecolor=INK, linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(metrics.construct, rotation=18, ha="right")
ax.set_ylabel("diagnostic score"); ax.set_ylim(0, 1.05)
ax.legend(frameon=True, facecolor="white", edgecolor="#777777", ncol=3)
ax.grid(axis="y", color="#b0b0b0", alpha=0.45, linewidth=0.55)
savefig("f10_diagnostic.png")

# Fig 11 — housing-link feasibility by ACS vintage. The full input has 12
# circuit rows per vintage; the substantive legal merge matches eight cells,
# all in 2022.
vintages = [2012, 2017, 2022]
input_counts = hp.groupby("year").size().reindex(vintages, fill_value=0).to_numpy()
matched_counts = panel.groupby("year").size().reindex(vintages, fill_value=0).to_numpy()
fig, ax = plt.subplots(figsize=(6.8, 3.2))
x = np.arange(len(vintages))
ax.plot(x, input_counts, color=BLUE, lw=1.6, marker="o", markersize=6,
        markeredgecolor=INK, markeredgewidth=0.35, label="ACS input cells")
ax.plot(x, matched_counts, color=ORANGE, lw=1.6, marker="s", markersize=6,
        markeredgecolor=INK, markeredgewidth=0.35, label="matched substantive cells")
for xi, val in zip(x, input_counts):
    ax.text(xi, val + 0.35, str(int(val)), ha="center", va="bottom", fontsize=8)
for xi, val in zip(x, matched_counts):
    ax.text(xi, val + 0.35, str(int(val)), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in vintages])
ax.set_xlabel("ACS five-year vintage"); ax.set_ylabel("circuit cells")
ax.set_ylim(0, max(input_counts) + 3)
ax.legend(frameon=True, facecolor="white", edgecolor="#777777",
          loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2)
ax.grid(axis="y", color="#b0b0b0", alpha=0.45, linewidth=0.55)
savefig("f11_housing_feasibility.png")
K["twfe"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in econ.items()
             if k in ("coef", "se", "t", "p", "wild_p", "n", "n_clusters", "note")}
K["panel_cells"] = int(len(panel))
K["panel_clusters"] = int(panel.unit.nunique())
K["outcome_mean"] = round(float(panel["dissimilarity_index"].mean()), 4)
K["outcome_sd"] = round(float(panel["dissimilarity_index"].std(ddof=1)), 4)
K["outcome_min"] = round(float(panel["dissimilarity_index"].min()), 4)
K["outcome_max"] = round(float(panel["dissimilarity_index"].max()), 4)
K["feii_mean"] = round(float(panel["FEII"].mean()), 4)
K["feii_sd"] = round(float(panel["FEII"].std(ddof=1)), 4)
K["feii_min"] = round(float(panel["FEII"].min()), 4)
K["feii_max"] = round(float(panel["FEII"].max()), 4)
if "note" not in econ:
    K["twfe_ci95_t"] = [round(float(econ["ci_low"]), 4),
                         round(float(econ["ci_high"]), 4)]

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
    ["District-court opinions in corpus", int((feat.court_level == "district").sum())],
    ["Circuits represented", feat.circuit.nunique()],
    ["ACS vintages", "2012, 2017, 2022 (non-overlapping)"],
    ["Year span", f"{int(feat.year.min())}-{int(feat.year.max())}"],
    ["Margin of error (50/50 proportion, 95%, FPC)", f"±{moe:.1%}"],
    ["Total-variation distance, circuit (sample vs pop)", f"{tvd_circ:.3f}"],
    ["Total-variation distance, decade (sample vs pop)", f"{tvd_dec:.3f}"],
], columns=["quantity", "value"]).to_csv(TAB / "t1_corpus.csv", index=False)

# Table 2 — circuit comparison
g2 = (feat.groupby("circuit").agg(
        n=("cluster_id", "count"), legal_signal=("doctrinal_strictness", "mean"),
        proof_treatment=("proof_treatment", "mean"), proof_impact=("proof_impact", "mean"),
        duty_accommodation=("duty_accommodation", "mean"),
        n_outcome_cues=("outcome_cue", lambda s: int(s.notna().sum())),
        outcome_cue_rate=("outcome_cue", lambda s: s.dropna().mean())).reset_index())
g2 = g2.merge(pf.groupby("unit")["FEII"].mean().rename("FEII").reset_index().rename(columns={"unit": "circuit"}),
              on="circuit", how="left")
g2 = g2.sort_values("legal_signal", ascending=False).round(3)
g2.to_csv(TAB / "t2_circuits.csv", index=False)

# Table 3 — regime characteristics
reg_tab = feat.groupby("regime").agg(
    n=("cluster_id", "count"), legal_signal=("doctrinal_strictness", "mean"),
    proof_impact=("proof_impact", "mean"),
    hud_framework=("reason_hud_burden_shifting", "mean"),
    mcdonnell=("reason_mcdonnell_douglas", "mean"),
    n_outcome_cues=("outcome_cue", lambda s: int(s.notna().sum())),
    outcome_cue_rate=("outcome_cue", lambda s: s.dropna().mean())).round(3)
reg_tab.to_csv(TAB / "t3_regimes.csv")

# Feasibility record; the real-data regression is intentionally not estimated.
pd.DataFrame([{
    "housing_rows": len(hp),
    "matched_substantive_cells": len(panel),
    "matched_years": int(panel.year.nunique()),
    "matched_circuits": int(panel.unit.nunique()),
    "acs_vintages": "2012, 2017, 2022",
    "estimate_reported": False,
    "reason": K["twfe"].get("note", "")
}]).to_csv(TAB / "housing_feasibility.csv", index=False)

(config.OUTPUTS / "paper" / "key_numbers.json").write_text(json.dumps(K, indent=2, default=str))
print("\nKEY NUMBERS:"); print(json.dumps(K, indent=1, default=str))
print(f"\nAll assets in {config.OUTPUTS/'paper'}")

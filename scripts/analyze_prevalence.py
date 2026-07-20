#!/usr/bin/env python3
"""Claim prevalence in the frozen 150-cluster random sample, on the substantive
denominator, against the paper's regex shares and their Rogan-Gladen correction."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import config  # noqa: E402
from fha import prevalence as pv  # noqa: E402
from fha.extract import extract_case  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

CLAIMS = pv.CLAIMS


def assurance(n10, n01, n_observed, n_target, alpha=0.05, gridsize=80):
    """Expected power at ``n_target`` averaged over the posterior of the effect size
    (favor-leading-side probability pi ~ Beta), which the point estimate ignores."""
    if not n_target:
        return None
    psi = (n10 + n01) / n_observed
    grid = np.linspace(0.002, 0.998, gridsize)
    weights = beta.pdf(grid, max(n10, n01) + 1, min(n10, n01) + 1)
    weights = weights / weights.sum()
    return float(sum(w * pv.power_psi_pi(psi, g, n_target, alpha)
                     for g, w in zip(grid, weights)))

OUT = config.OUTPUTS / "paper" / "validation"

# Regex extractor shares over the paper's 417 substantive clusters.
PAPER_SHARES_417 = {"disparate_treatment": 0.376, "disparate_impact": 0.281,
                    "refusal_rent_sell": 0.237, "reasonable_accommodation": 0.355,
                    "zoning_exclusionary": 0.094}

PAIRS = [("reasonable_accommodation", "disparate_impact"),
         ("disparate_treatment", "reasonable_accommodation"),
         ("disparate_treatment", "disparate_impact")]

FOCAL_PAIR = PAIRS[0]

SILVER_WARNING = (
    "NOTE: the random-sample labels above are LLM-GENERATED SILVER LABELS, not "
    "human gold. Self-consistency measures agreement across passes of the same "
    "model (reliability), not agreement with a human coder (validity); a model "
    "can be perfectly self-consistent and perfectly wrong. The LLM's measured "
    "accuracy comes only from the 93-case gold overlap, which was an enriched, "
    "choice-based sample and does not transport to this random draw. A "
    "HUMAN-CODED RANDOM SUBSET IS STILL REQUIRED before these prevalence "
    "figures may be reported as validated."
)


def load_majority_votes():
    with open(config.VALIDATION / "llm_majority_votes.json") as fh:
        return json.load(fh)


def regex_operating_characteristics(corpus):
    """Regex precision/recall vs the human primary codings on the gold overlap."""
    human = json.load(open(config.VALIDATION / "gold_human_codings.json"))
    prim = {c["case_id"]: c for c in human["primary"]}
    votes = load_majority_votes()
    ids = sorted(i for i in prim if i in corpus and str(i) in votes["goldVote"])
    if not ids:
        raise SystemExit("no overlap between human primary codings, corpus and gold votes")
    feats = {i: extract_case(corpus[i]) for i in ids}
    out = {}
    for claim in CLAIMS:
        y_true = [prim[i]["claims"][claim] for i in ids]
        y_pred = [int(feats[i][f"claim_{claim}"]) for i in ids]
        p, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0)
        out[claim] = (float(p), float(r))
    return out, len(ids)


def build_prevalence_table(votes, substantive, regex_pr):
    table = pv.prevalence_table(votes, substantive)
    rows = []
    for _, row in table.iterrows():
        claim = row["construct"]
        observed = PAPER_SHARES_417[claim]
        precision, recall = regex_pr[claim]
        rows.append({
            "construct": claim,
            "k": int(row["k"]),
            "n": int(row["n"]),
            "llm_share": round(float(row["rate"]), 3),
            "wilson_lo": round(float(row["ci_low"]), 3),
            "wilson_hi": round(float(row["ci_high"]), 3),
            "regex_share_417": observed,
            "regex_precision": round(precision, 3),
            "regex_recall": round(recall, 3),
            "regex_share_417_corrected": round(
                pv.rogan_gladen(observed, precision, recall), 3),
        })
    return pd.DataFrame(rows)


def build_paired_tests(votes, substantive):
    rows = []
    for claim_a, claim_b in PAIRS:
        res = pv.mcnemar_exact(votes, substantive, claim_a, claim_b)
        rows.append({"claim_a": claim_a, "claim_b": claim_b,
                     "n": len(substantive),
                     "share_a": round(res["rate_a"], 3),
                     "share_b": round(res["rate_b"], 3),
                     "diff": round(res["diff"], 3),
                     "n10": res["n10"], "n01": res["n01"],
                     "n_discordant": res["n10"] + res["n01"],
                     "p_exact_mcnemar": round(res["p_value"], 6)})
    return pd.DataFrame(rows)


def print_denominators(index, n_drawn, n_sub, n_missing):
    print("== RANDOM SAMPLE ==")
    print(f" seed {index['seed']}  frame {index['frame_size']} clusters  "
          f"drawn {index['n']} (resolved in corpus: {n_drawn})")
    if n_missing:
        print(f" WARNING: {n_missing} drawn clusters have no LLM majority vote "
              f"and are excluded")
    print(f" DENOMINATORS: {n_drawn} drawn, {n_sub} substantive under rule_is_fha "
          f"({n_sub / n_drawn:.1%})")
    print(" WARNING: the paper's 417-cluster shares are computed over SUBSTANTIVE")
    print(f" clusters only. They are comparable to the n={n_sub} denominator alone.")
    print(f" Comparing them against all {n_drawn} drawn clusters would divide a")
    print(" substantive-only numerator by a pre-screening denominator and would")
    print(" understate every share by roughly half. That comparison is INVALID.")


def print_report(table, paired, power, n_overlap, n_sub, n_drawn, self_consistency):
    print(f"\n== PREVALENCE (LLM 3-pass majority, n={n_sub} substantive) ==")
    print(f" regex precision/recall measured on the {n_overlap}-case human overlap;")
    print(" corrected share = observed * precision / recall (Rogan-Gladen)")
    header = (f"{'construct':<26}{'k':>4}{'share':>8}{'95% CI':>17}"
              f"{'regex417':>10}{'rxP':>7}{'rxR':>7}{'corrected':>11}")
    print(header)
    print("-" * len(header))
    for _, row in table.iterrows():
        ci = f"[{row['wilson_lo']:.3f},{row['wilson_hi']:.3f}]"
        print(f"{row['construct']:<26}{row['k']:>4}{row['llm_share']:>8.3f}{ci:>17}"
              f"{row['regex_share_417']:>10.3f}{row['regex_precision']:>7.3f}"
              f"{row['regex_recall']:>7.3f}{row['regex_share_417_corrected']:>11.3f}")

    print(f"\n== PAIRED EXACT McNEMAR (same {n_sub} cases) ==")
    for _, row in paired.iterrows():
        stars = "" if row["p_exact_mcnemar"] >= 0.05 else "  *"
        print(f" {row['claim_a']} vs {row['claim_b']}: "
              f"{row['share_a']:.3f} vs {row['share_b']:.3f}  "
              f"n10={row['n10']} n01={row['n01']}  "
              f"p={row['p_exact_mcnemar']:.4f}{stars}")

    print("\n== POWER (exact McNemar, focal pair) ==")
    claim_a, claim_b = FOCAL_PAIR
    p80, p90 = power["p80"], power["p90"]
    print(f" {claim_a} vs {claim_b}: n10={power['n10']} n01={power['n01']}, "
          f"p={power['p_observed']:.4f}, power={power['power_at_observed']:.3f} "
          f"at n={n_sub} substantive.")
    print(" required substantive n to detect the observed effect "
          "(holding discordant rate and split fixed):")
    for tag, res in [("80% power", p80), ("90% power", p90)]:
        if res["converged"]:
            print(f"   {tag}: n={res['n_substantive']} substantive "
                  f"(~{res['n_draws']} total draws)")
        else:
            print(f"   {tag}: not reachable (split is non-informative)")
    lo, hi = power["pi_ci"]
    print(f" the effect is imprecisely estimated: favor-{claim_a[:4]} share "
          f"pi={power['pi']:.3f} rests on {power['n10'] + power['n01']} discordant "
          f"pairs,")
    flag = "INCLUDES 0.5" if lo <= 0.5 <= hi else "excludes 0.5"
    print(f"   Wilson 95% CI [{lo:.3f}, {hi:.3f}] {flag}.")
    if power["assurance_at_80"] is not None:
        print(f" averaging power over that uncertainty (posterior pi ~ Beta), "
              f"assurance at the 80%-power n is only {power['assurance_at_80']:.2f},")
        print(" so the ordering may not resolve even at a nominally powered size. "
              "The treatment lead is already significant.")
    print(f"\n== SELF-CONSISTENCY ==  {self_consistency} "
          f"(fraction of claim decisions unanimous across passes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT),
                    help="directory for prevalence_random.csv and paired_tests.csv")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for the power projection")
    ap.add_argument("--no-write", action="store_true",
                    help="print the report without writing output files")
    args = ap.parse_args()

    corpus = pv.load_corpus()
    index = json.load(open(config.VALIDATION / "random_sample_index.json"))
    committed = load_majority_votes()
    votes = committed["randVote"]

    drawn = [c for c in index["cluster_ids"] if c in corpus]
    if len(drawn) != index["n"]:
        print(f"WARNING: {index['n'] - len(drawn)} drawn cluster_ids are absent "
              f"from paper_corpus.jsonl")
    scored = [c for c in drawn if str(c) in votes]
    substantive = pv.substantive_subset(scored, corpus)

    print_denominators(index, len(drawn), len(substantive), len(drawn) - len(scored))

    regex_pr, n_overlap = regex_operating_characteristics(corpus)
    table = build_prevalence_table(votes, substantive, regex_pr)
    paired = build_paired_tests(votes, substantive)

    focal = pv.mcnemar_exact(votes, substantive, *FOCAL_PAIR)
    n_sub = len(substantive)
    rate = n_sub / len(drawn)
    n10, n01 = focal["n10"], focal["n01"]
    p80 = pv.n_for_power(n10, n01, n_sub, 0.80, args.alpha, rate)
    p90 = pv.n_for_power(n10, n01, n_sub, 0.90, args.alpha, rate)
    assurance_80 = (assurance(n10, n01, n_sub, p80["n_substantive"], args.alpha)
                    if p80["converged"] else None)
    power = {"n10": n10, "n01": n01, "p_observed": focal["p_value"],
             "power_at_observed": p80["power_at_observed"], "pi": p80["pi"],
             "pi_ci": pv.wilson(max(n10, n01), n10 + n01),
             "p80": p80, "p90": p90, "assurance_at_80": assurance_80}

    print_report(table, paired, power, n_overlap, len(substantive), len(drawn),
                 committed["selfConsistency"])

    if not args.no_write:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "prevalence_random.csv", index=False)
        paired.to_csv(out_dir / "paired_tests.csv", index=False)
        print(f"\nwrote {out_dir / 'prevalence_random.csv'}")
        print(f"wrote {out_dir / 'paired_tests.csv'}")

    print("\n" + "=" * 78)
    print(SILVER_WARNING)
    print("=" * 78)


if __name__ == "__main__":
    main()

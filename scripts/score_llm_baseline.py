#!/usr/bin/env python3
"""Head-to-head of the LLM 3-pass baseline vs the deterministic regex extractor
on the 93-case human-coded overlap."""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import config  # noqa: E402
from fha.extract import extract_case  # noqa: E402
from fha.llm_baseline import majority_vote, self_consistency  # noqa: E402

CLAIMS = ["disparate_treatment", "disparate_impact", "refusal_rent_sell",
          "reasonable_accommodation", "zoning_exclusionary"]
FW = {"hud_three_step": "hud", "mcdonnell_douglas": "mcdonnell",
      "none_explicit": "none"}

OUT = config.OUTPUTS / "paper" / "validation"

SAMPLE_WARNING = (
    "WARNING: the 93-case overlap is an ENRICHED, CHOICE-BASED sample "
    "(human coding was allocated to cases likely to carry FHA claims). "
    "The rates below are DIAGNOSTIC comparisons between two extractors on "
    "that sample, NOT corpus-wide error rates, and they do not transport to "
    "the 751-opinion corpus without reweighting."
)


def load_corpus():
    corpus = {}
    with open(config.PROCESSED / "paper_corpus.jsonl") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                corpus[rec["cluster_id"]] = rec
    return corpus


def group_passes(labels):
    grouped = collections.defaultdict(list)
    for rec in labels:
        grouped[(rec["set"], rec["cluster_id"])].append(rec)
    return grouped


def build_votes(grouped):
    votes = {"gold": {}, "rand": {}}
    for (which, cluster_id), records in grouped.items():
        votes[which][cluster_id] = majority_vote(records)
    return votes


def check_votes_match(votes, committed):
    """Compare recomputed votes against the committed file; return mismatches."""
    mismatches = []
    for which, key in (("gold", "goldVote"), ("rand", "randVote")):
        mine = votes[which]
        theirs = committed[key]
        if set(map(str, mine)) != set(theirs):
            mismatches.append(f"{which}: case-id set differs "
                              f"({len(mine)} recomputed vs {len(theirs)} committed)")
            continue
        for cluster_id, vote in sorted(mine.items()):
            ref = theirs[str(cluster_id)]
            for field in CLAIMS + ["framework", "n"]:
                if vote[field] != ref[field]:
                    mismatches.append(
                        f"{which}/{cluster_id}/{field}: "
                        f"recomputed {vote[field]!r} != committed {ref[field]!r}")
    return mismatches


def score_binary(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return (round(float(p), 3), round(float(r), 3), round(float(f), 3),
            round(float(cohen_kappa_score(y_true, y_pred)), 3))


def claim_table(ids, human, regex_label, llm_label):
    rows = []
    regex_stack = []
    llm_stack = []
    human_stack = []
    for claim in CLAIMS:
        y_true = np.array([human[i]["claims"][claim] for i in ids])
        y_regex = np.array([regex_label(i, claim) for i in ids])
        y_llm = np.array([llm_label(i, claim) for i in ids])
        human_stack.append(y_true)
        regex_stack.append(y_regex)
        llm_stack.append(y_llm)
        rows.append(_row(f"claim_{claim}", y_true, y_regex, y_llm))
    y_true = np.array(human_stack).T.ravel()
    y_regex = np.array(regex_stack).T.ravel()
    y_llm = np.array(llm_stack).T.ravel()
    rows.append(_row("claims_micro", y_true, y_regex, y_llm))
    return rows


def _row(name, y_true, y_regex, y_llm):
    rp, rr, rf, rk = score_binary(y_true, y_regex)
    lp, lr, lf, lk = score_binary(y_true, y_llm)
    return {"variable": name, "n": int(len(y_true)),
            "prevalence_human": round(float(y_true.mean()), 3),
            "regex_precision": rp, "regex_recall": rr,
            "regex_f1": rf, "regex_kappa": rk,
            "llm_precision": lp, "llm_recall": lr,
            "llm_f1": lf, "llm_kappa": lk,
            "delta_f1": round(lf - rf, 3)}


def framework_accuracy(ids, human, predict):
    """Human 'both' counts as correct when the extractor names either framework."""
    correct = 0
    for i in ids:
        gold = human[i]["framework"]
        pred = predict(i)
        correct += (pred == gold) or (gold == "both" and pred in ("hud", "mcdonnell"))
    return round(correct / len(ids), 3)


def print_report(res, rows):
    print(SAMPLE_WARNING)
    print()
    print(f"model {res['model']}  passes {res['n_passes']}  "
          f"overlap cases {res['n_cases']} (corpus n human primary)")
    print(f"majority votes recomputed and matched committed file: "
          f"{res['votes_match_committed']}")
    print()
    print("== CLAIMS: regex vs LLM 3-pass majority (human primary = reference) ==")
    header = (f"{'variable':<32}{'prev':>7}"
              f"{'rxP':>7}{'rxR':>7}{'rxF1':>7}{'rxK':>7}"
              f"{'llmP':>8}{'llmR':>7}{'llmF1':>7}{'llmK':>7}{'dF1':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['variable']:<32}{row['prevalence_human']:>7.3f}"
              f"{row['regex_precision']:>7.3f}{row['regex_recall']:>7.3f}"
              f"{row['regex_f1']:>7.3f}{row['regex_kappa']:>7.3f}"
              f"{row['llm_precision']:>8.3f}{row['llm_recall']:>7.3f}"
              f"{row['llm_f1']:>7.3f}{row['llm_kappa']:>7.3f}"
              f"{row['delta_f1']:>+8.3f}")
    print()
    print(f"== FRAMEWORK ==  regex accuracy {res['framework_accuracy_regex']}  "
          f"LLM accuracy {res['framework_accuracy_llm']}  "
          f"delta {res['framework_accuracy_delta']:+.3f}")
    sc = res["self_consistency"]
    print(f"== SELF-CONSISTENCY ==  {sc['fraction']} of {sc['n_decisions']} "
          f"claim decisions unanimous across passes "
          f"({sc['n_cases']} cases, gold+random)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT),
                    help="directory for llm_vs_regex.csv and llm_baseline_summary.json")
    ap.add_argument("--no-write", action="store_true",
                    help="print the report without writing output files")
    args = ap.parse_args()

    human = json.load(open(config.VALIDATION / "gold_human_codings.json"))
    prim = {c["case_id"]: c for c in human["primary"]}
    raw = json.load(open(config.VALIDATION / "llm_labels_3pass.json"))
    committed = json.load(open(config.VALIDATION / "llm_majority_votes.json"))
    corpus = load_corpus()

    grouped = group_passes(raw["labels"])
    votes = build_votes(grouped)
    mismatches = check_votes_match(votes, committed)
    if mismatches:
        raise SystemExit("recomputed majority votes disagree with "
                         "llm_majority_votes.json:\n  " + "\n  ".join(mismatches[:20]))
    gold_votes = votes["gold"]

    ids = sorted(i for i in prim if i in corpus and i in gold_votes)
    if not ids:
        raise SystemExit("no overlap between human primary codings, corpus and LLM gold votes")

    regex_rows = {i: extract_case(corpus[i]) for i in ids}
    sc_fraction = self_consistency(raw["labels"])
    sc_total = len(grouped) * len(CLAIMS)

    rows = claim_table(
        ids, prim,
        lambda i, claim: int(regex_rows[i][f"claim_{claim}"]),
        lambda i, claim: int(gold_votes[i][claim]))

    acc_regex = framework_accuracy(
        ids, prim, lambda i: FW.get(str(regex_rows[i]["burden_framework"]), "none"))
    acc_llm = framework_accuracy(ids, prim, lambda i: gold_votes[i]["framework"])

    res = {
        "model": raw["model"],
        "n_passes": raw["n_passes"],
        "n_cases": len(ids),
        "votes_match_committed": True,
        "sample_warning": SAMPLE_WARNING,
        "claims": rows,
        "framework_accuracy_regex": acc_regex,
        "framework_accuracy_llm": acc_llm,
        "framework_accuracy_delta": round(acc_llm - acc_regex, 3),
        "self_consistency": {
            "fraction": round(sc_fraction, 4),
            "n_cases": len(grouped),
            "n_decisions": sc_total,
            "committed": committed["selfConsistency"],
        },
    }

    print_report(res, rows)

    if not args.no_write:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "llm_vs_regex.csv", index=False)
        json.dump(res, (out_dir / "llm_baseline_summary.json").open("w"), indent=1)
        print(f"\nwrote {out_dir / 'llm_vs_regex.csv'}")
        print(f"wrote {out_dir / 'llm_baseline_summary.json'}")


if __name__ == "__main__":
    main()

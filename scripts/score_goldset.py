#!/usr/bin/env python3
"""
Score the 93-case enriched/choice-based canonical-corpus overlap (machine vs
hand coding) and inter-annotator reliability from the 30-case second pass.
Reads data/validation/, writes
outputs/paper/validation/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import config  # noqa: E402
from fha.extract import extract_case  # noqa: E402
from fha.classify import build_clean_corpus  # noqa: E402

S = str(Path(__file__).resolve().parents[1] / "data" / "validation")
OUT = config.OUTPUTS / "paper" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

CLAIMS = ["disparate_treatment", "disparate_impact", "refusal_rent_sell",
          "reasonable_accommodation", "zoning_exclusionary"]
FW = {"hud_three_step": "hud", "mcdonnell_douglas": "mcdonnell",
      "none_explicit": "none"}


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 3), round((c + h) / d, 3)


def main():
    human = json.load(open(f"{S}/gold_human_codings.json"))
    prim = {c["case_id"]: c for c in human["primary"]}
    sec = {c["case_id"]: c for c in human["second"]}
    # Regenerate machine labels from the current extractor so validation never
    # silently scores a stale CSV after a rule or lexicon change.
    corpus = {r["cluster_id"]: r for r in
              (json.loads(l) for l in open(config.PROCESSED / "paper_corpus.jsonl")
               if l.strip())}
    ml_rows = []
    for case_id in sorted(set(prim) | set(sec)):
        if case_id not in corpus:
            continue
        row = extract_case(corpus[case_id])
        ml_rows.append({"cluster_id": case_id, **{
            k: row.get(k) for k in [
                "year", "circuit", "court_level",
                "claim_disparate_treatment", "claim_disparate_impact",
                "claim_zoning_exclusionary", "claim_refusal_rent_sell",
                "claim_reasonable_accommodation", "burden_framework",
                "outcome_cue", "doctrinal_strictness"]}})
    ml = pd.DataFrame(ml_rows).set_index("cluster_id")
    ml.to_csv(f"{S}/gold_machine_labels.csv")
    ids = [i for i in ml.index if i in prim]
    print(f"scored cases: {len(ids)} (machine ∩ human primary)")

    res = {"n_cases": len(ids)}

    # ---- claims: machine vs human (human = reference) ----
    rows = []
    for c in CLAIMS:
        y_true = np.array([prim[i]["claims"][c] for i in ids])
        y_pred = np.array([int(ml.loc[i, f"claim_{c}"]) for i in ids])
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0)
        k = cohen_kappa_score(y_true, y_pred)
        rows.append({"variable": f"claim_{c}", "prevalence_human": round(y_true.mean(), 3),
                     "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(f, 3), "kappa": round(k, 3)})
    # micro over all claim decisions
    yt = np.array([[prim[i]["claims"][c] for c in CLAIMS] for i in ids]).ravel()
    yp = np.array([[int(ml.loc[i, f"claim_{c}"]) for c in CLAIMS] for i in ids]).ravel()
    mp, mr, mf, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    rows.append({"variable": "claims_micro", "prevalence_human": round(yt.mean(), 3),
                 "precision": round(mp, 3), "recall": round(mr, 3),
                 "f1": round(mf, 3), "kappa": round(cohen_kappa_score(yt, yp), 3)})
    res["claims"] = rows

    # ---- framework ----
    def m_fw(i):
        return FW.get(str(ml.loc[i, "burden_framework"]), "none")
    correct = 0
    for i in ids:
        h = prim[i]["framework"]
        m = m_fw(i)
        ok = (m == h) or (h == "both" and m in ("hud", "mcdonnell"))
        correct += ok
    res["framework_accuracy"] = round(correct / len(ids), 3)
    # kappa on the 3 machine-expressible categories (drop human "both")
    hh = [prim[i]["framework"] for i in ids if prim[i]["framework"] != "both"]
    mm = [m_fw(i) for i in ids if prim[i]["framework"] != "both"]
    res["framework_kappa"] = round(cohen_kappa_score(hh, mm), 3)

    # ---- winner / disposition ----
    def h_win(i):
        return {"P": 1, "D": 0}.get(prim[i]["winner"], None)  # mixed/unclear -> None

    def m_win(i):
        v = ml.loc[i, "outcome_cue"]
        return None if pd.isna(v) else int(v)

    # holding detection: does machine flag a decisive holding where a human sees one
    hd = np.array([h_win(i) is not None for i in ids])
    md = np.array([m_win(i) is not None for i in ids])
    p, r, f, _ = precision_recall_fscore_support(hd, md, average="binary", zero_division=0)
    res["holding_detection"] = {"precision": round(p, 3), "recall": round(r, 3),
                                "f1": round(f, 3), "n_human_holdings": int(hd.sum()),
                                "n_machine_holdings": int(md.sum())}

    # among the machine's holdings: human distribution + win-rate check
    m_hold = [i for i in ids if m_win(i) is not None]
    hd_dist = pd.Series([prim[i]["winner"] for i in m_hold]).value_counts().to_dict()
    both_dec = [i for i in m_hold if h_win(i) is not None]
    acc = np.mean([m_win(i) == h_win(i) for i in both_dec]) if both_dec else np.nan
    kap = cohen_kappa_score([m_win(i) for i in both_dec],
                            [h_win(i) for i in both_dec]) if both_dec else np.nan
    m_wr = np.mean([m_win(i) for i in m_hold])
    h_wr = np.mean([h_win(i) for i in both_dec]) if both_dec else np.nan
    res["winner_on_machine_holdings"] = {
        "n_machine_holdings": len(m_hold),
        "human_label_distribution": hd_dist,
        "n_both_decisive": len(both_dec),
        "binary_accuracy": round(float(acc), 3),
        "binary_kappa": round(float(kap), 3),
        "machine_win_rate": round(float(m_wr), 3),
        "human_win_rate_same_cases": round(float(h_wr), 3),
    }

    # Win rate and interval from the current canonical full-text corpus.
    full = [json.loads(l) for l in
            open(config.PROCESSED / "paper_corpus.jsonl") if l.strip()]
    full_clean, _ = build_clean_corpus(full, use_ml=False, require_nos443=True)
    full_feat = pd.DataFrame(extract_case(r) for r in full_clean)
    full_feat = full_feat[full_feat.circuit.notna() & full_feat.year.notna()]
    full_pw = full_feat["outcome_cue"].dropna()
    full_rate = float(full_pw.mean()) if len(full_pw) else float("nan")
    res["corpus_win_rate"] = {"rate": round(full_rate, 3), "n": int(len(full_pw)),
                              "wilson95": wilson(full_rate, int(len(full_pw)))}

    # ---- inter-annotator reliability (second pass vs primary) ----
    sp = [i for i in sec if i in prim]
    # claims (pooled over 5 labels)
    a = np.array([[prim[i]["claims"][c] for c in CLAIMS] for i in sp]).ravel()
    b = np.array([[sec[i]["claims"][c] for c in CLAIMS] for i in sp]).ravel()
    # framework (drop nothing; 4-way)
    fa = [prim[i]["framework"] for i in sp]
    fb = [sec[i]["framework"] for i in sp]
    # winner 4-way and decisive-binary
    wa = [prim[i]["winner"] for i in sp]
    wb = [sec[i]["winner"] for i in sp]
    dec = [i for i in sp if h_win(i) is not None and
           {"P": 1, "D": 0}.get(sec[i]["winner"]) is not None]
    res["inter_annotator"] = {
        "n_double_coded": len(sp),
        "claims_agreement": round(float((a == b).mean()), 3),
        "claims_kappa": round(cohen_kappa_score(a, b), 3),
        "framework_agreement": round(float(np.mean([x == y for x, y in zip(fa, fb)])), 3),
        "framework_kappa": round(cohen_kappa_score(fa, fb), 3),
        "winner_agreement_4way": round(float(np.mean([x == y for x, y in zip(wa, wb)])), 3),
        "winner_kappa_4way": round(cohen_kappa_score(wa, wb), 3),
        "winner_binary_agreement": round(float(np.mean(
            [{"P": 1, "D": 0}[prim[i]["winner"]] == {"P": 1, "D": 0}[sec[i]["winner"]]
             for i in dec])), 3) if dec else None,
        "n_winner_binary": len(dec),
    }

    json.dump(res, (OUT / "goldset_metrics.json").open("w"), indent=1)
    pd.DataFrame(rows).to_csv(OUT / "goldset_precision_recall.csv", index=False)

    # console summary
    print("\n== CLAIMS (machine vs hand; hand = reference) ==")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n== FRAMEWORK ==  accuracy {res['framework_accuracy']}  kappa {res['framework_kappa']}")
    print("\n== WINNER ==")
    print(" holding detection:", res["holding_detection"])
    print(" on machine's holdings:", res["winner_on_machine_holdings"])
    print(f" corpus win rate {res['corpus_win_rate']['rate']:.3f} "
          f"(n={res['corpus_win_rate']['n']}) 95% CI "
          f"{res['corpus_win_rate']['wilson95']}")
    print("\n== INTER-ANNOTATOR (2nd pass, n=%d) ==" % res["inter_annotator"]["n_double_coded"])
    ia = res["inter_annotator"]
    print(f" claims agree {ia['claims_agreement']} k={ia['claims_kappa']} | "
          f"framework agree {ia['framework_agreement']} k={ia['framework_kappa']} | "
          f"winner4 agree {ia['winner_agreement_4way']} k={ia['winner_kappa_4way']} | "
          f"winner-binary agree {ia['winner_binary_agreement']} (n={ia['n_winner_binary']})")


if __name__ == "__main__":
    main()

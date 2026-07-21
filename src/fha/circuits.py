"""Cross-circuit doctrinal divergence -- the 'circuit split' -- over the substantive
corpus. Tests whether circuits differ in which FHA theories their opinions assert."""
import json
import math

import pandas as pd
from scipy.stats import chi2_contingency

from . import config
from .classify import rule_is_fha
from .extract import extract_case

CLAIMS = ["disparate_treatment", "disparate_impact", "refusal_rent_sell",
          "reasonable_accommodation", "zoning_exclusionary"]


def substantive_rows(path=None):
    """One extracted feature row per substantive cluster with a known circuit."""
    path = path or config.PROCESSED / "paper_corpus.jsonl"
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rule_is_fha(rec) and str(rec.get("circuit")) not in ("None", ""):
            feat = extract_case(rec)
            feat["circuit"] = str(rec["circuit"])
            rows.append(feat)
    return rows


def _circuits(rows, min_n):
    counts = {}
    for r in rows:
        counts[r["circuit"]] = counts.get(r["circuit"], 0) + 1
    keep = [c for c, n in counts.items() if n >= min_n]
    return sorted(keep, key=lambda c: int(c) if c.isdigit() else 99)


def circuit_prevalence(rows=None, min_n=5):
    """Per-circuit prevalence of each claim construct (circuits with >= min_n clusters)."""
    rows = substantive_rows() if rows is None else rows
    out = []
    for c in _circuits(rows, min_n):
        rc = [r for r in rows if r["circuit"] == c]
        row = {"circuit": c, "n": len(rc)}
        for cl in CLAIMS:
            row[cl] = round(sum(r["claim_" + cl] for r in rc) / len(rc), 3)
        out.append(row)
    return pd.DataFrame(out)


def circuit_split(rows=None, min_n=5):
    """Chi-square test of circuit x claim independence, per construct. A significant
    result is a measured cross-circuit doctrinal split. Cramer's V is the effect size;
    with a 2-column table it is sqrt(chi2 / n)."""
    rows = substantive_rows() if rows is None else rows
    circs = _circuits(rows, min_n)
    res = {}
    for cl in CLAIMS:
        table = [[sum(r["claim_" + cl] for r in rows if r["circuit"] == c),
                  sum(1 - r["claim_" + cl] for r in rows if r["circuit"] == c)]
                 for c in circs]
        chi, p, dof, _ = chi2_contingency(table)
        n = sum(sum(t) for t in table)
        res[cl] = {"chi2": round(float(chi), 1), "dof": int(dof),
                   "p": round(float(p), 4), "cramers_v": round(math.sqrt(chi / n), 2),
                   "n": int(n)}
    return res

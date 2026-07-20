"""Random-sample prevalence estimation and measurement-error correction."""

import json
import math

import numpy as np
import pandas as pd

from . import config
from .classify import rule_is_fha

CLAIMS = ["disparate_treatment", "disparate_impact", "refusal_rent_sell",
          "reasonable_accommodation", "zoning_exclusionary"]


def load_corpus(path=None):
    path = path or config.PROCESSED / "paper_corpus.jsonl"
    corpus = {}
    for line in open(path):
        if line.strip():
            record = json.loads(line)
            corpus[record["cluster_id"]] = record
    return corpus


def _lookup(mapping, cluster_id):
    if cluster_id in mapping:
        return mapping[cluster_id]
    return mapping.get(str(cluster_id))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, (c - h) / d)), float(min(1.0, (c + h) / d)))


def rogan_gladen(observed, precision, recall):
    if recall == 0:
        return float("nan")
    return float(min(1.0, max(0.0, observed * precision / recall)))


def substantive_subset(cluster_ids, corpus):
    if not isinstance(corpus, dict):
        corpus = {r["cluster_id"]: r for r in corpus}
    kept = []
    for cluster_id in cluster_ids:
        record = _lookup(corpus, cluster_id)
        if record is not None and rule_is_fha(record):
            kept.append(cluster_id)
    return kept


def prevalence_table(votes, ids):
    scored = [i for i in ids if _lookup(votes, i) is not None]
    n = len(scored)
    rows = []
    for construct in CLAIMS:
        k = sum(int(_lookup(votes, i)[construct]) for i in scored)
        lo, hi = wilson(k, n)
        rows.append({"construct": construct, "k": k, "n": n,
                     "rate": k / n if n else float("nan"),
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _exact_p(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(n10, n01) + 1))
    return float(min(1.0, 2 * tail / 2 ** n))


def mcnemar_exact(votes, ids, construct_a, construct_b):
    scored = [i for i in ids if _lookup(votes, i) is not None]
    a = [int(_lookup(votes, i)[construct_a]) for i in scored]
    b = [int(_lookup(votes, i)[construct_b]) for i in scored]
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = len(scored)
    rate_a = sum(a) / n if n else float("nan")
    rate_b = sum(b) / n if n else float("nan")
    return {"n10": n10, "n01": n01, "rate_a": rate_a, "rate_b": rate_b,
            "diff": rate_a - rate_b, "p_value": _exact_p(n10, n01)}


def required_n(n10, n01, alpha=0.05, max_mult=10, n_substantive=None,
               substantive_rate=None):
    grid = np.arange(1.0, max_mult + 1e-9, 0.01)
    multiplier = None
    p_at_multiplier = None
    for m in grid:
        p = _exact_p(int(round(n10 * m)), int(round(n01 * m)))
        if p < alpha:
            multiplier = float(m)
            p_at_multiplier = p
            break
    out = {"n10": n10, "n01": n01, "alpha": alpha,
           "p_observed": _exact_p(n10, n01),
           "multiplier": multiplier, "p_at_multiplier": p_at_multiplier,
           "converged": multiplier is not None, "max_mult": max_mult}
    if multiplier is None:
        out["n_substantive_required"] = None
        out["n_draw_required"] = None
        return out
    n_sub = (int(math.ceil(n_substantive * multiplier))
             if n_substantive is not None else None)
    out["n_substantive_required"] = n_sub
    out["n_draw_required"] = (int(math.ceil(n_sub / substantive_rate))
                              if n_sub is not None and substantive_rate
                              else None)
    return out

"""Random-sample prevalence estimation and measurement-error correction."""

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import binom

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


def precision_recall_correction(observed, precision, recall):
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


def _reject_thresholds(n_discordant, alpha=0.05):
    """Largest minority discordant count that still rejects the two-sided exact
    McNemar test at ``alpha``, computed for every value in ``n_discordant`` at once.
    -1 marks discordant totals too small for any split to reject. Uses the binomial
    CDF (not 2**n integer sums) so it is stable for large samples."""
    d = np.asarray(n_discordant, dtype=int)
    cand = binom.ppf(alpha / 2.0, d, 0.5)
    cand = np.where(np.isnan(cand), -1, cand).astype(int)
    for _ in range(2):  # ppf can be off by one; step down until 2*cdf < alpha holds
        too_high = (cand >= 0) & (2 * binom.cdf(cand, d, 0.5) >= alpha)
        cand = np.where(too_high, cand - 1, cand)
    for _ in range(2):  # then step up while the next count still rejects
        can_up = 2 * binom.cdf(np.clip(cand + 1, 0, d), d, 0.5) < alpha
        cand = np.where(can_up, cand + 1, cand)
    return np.maximum(cand, -1)


def power_psi_pi(psi, pi, n_target, alpha=0.05):
    """Exact unconditional power of the two-sided exact McNemar test at substantive
    sample size ``n_target``. Each case is a discordant pair with probability ``psi``
    and, when discordant, favors the leading side with probability ``pi``. This
    integrates over the random number of discordant pairs, so the point where it first
    reaches ~0.5 is roughly the old break-even "just clears alpha" estimate."""
    if n_target <= 0 or psi <= 0:
        return 0.0
    d = np.arange(n_target + 1)
    p_disc = binom.pmf(d, n_target, psi)
    thr = _reject_thresholds(d, alpha)
    valid = thr >= 0
    lower = binom.cdf(np.where(valid, thr, 0), d, pi)
    upper = 1.0 - binom.cdf(np.where(valid, d - thr - 1, d), d, pi)
    reject = np.where(valid, lower + upper, 0.0)
    return float((p_disc * reject).sum())


def mcnemar_power(n10, n01, n_observed, n_target, alpha=0.05):
    """Power to detect the observed paired effect at substantive sample size
    ``n_target``, holding the observed discordant rate and split fixed."""
    d = n10 + n01
    if d == 0 or n_observed <= 0:
        return 0.0
    return power_psi_pi(d / n_observed, max(n10, n01) / d, n_target, alpha)


def n_for_power(n10, n01, n_observed, target_power=0.80, alpha=0.05,
                substantive_rate=None, max_n=6000):
    """Smallest substantive sample size reaching ``target_power`` for the observed
    effect. ``converged`` is False when the split is non-informative (pi<=0.5) or the
    search ceiling cannot reach the target."""
    d = n10 + n01
    pi = max(n10, n01) / d if d else 0.0
    out = {"n10": n10, "n01": n01, "target_power": target_power, "alpha": alpha,
           "pi": pi, "n_observed": n_observed,
           "power_at_observed": mcnemar_power(n10, n01, n_observed, n_observed, alpha)}
    if pi <= 0.5 or mcnemar_power(n10, n01, n_observed, max_n, alpha) < target_power:
        out.update(converged=False, n_substantive=None, n_draws=None)
        return out
    lo, hi = 1, max_n
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_power(n10, n01, n_observed, mid, alpha) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    out.update(converged=True, n_substantive=lo,
               n_draws=(int(math.ceil(lo / substantive_rate))
                        if substantive_rate else None))
    return out

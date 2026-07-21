#!/usr/bin/env python3
"""Scoring helpers for the frozen LLM extraction baseline.

These functions recompute votes and diagnostics from committed artifacts; they do not call
an LLM.
"""
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CLAIMS = ["disparate_treatment", "disparate_impact", "refusal_rent_sell",
          "reasonable_accommodation", "zoning_exclusionary"]


def _key(rec):
    return (rec.get("set"), rec["cluster_id"])


def group_passes(labels):
    groups = defaultdict(list)
    for rec in labels:
        groups[_key(rec)].append(rec)
    return dict(groups)


def majority_vote(records):
    out = {}
    n = len(records)
    for claim in CLAIMS:
        out[claim] = int(sum(r[claim] for r in records) * 2 > n)
    frameworks = [r["framework"] for r in records]
    counts = {f: frameworks.count(f) for f in frameworks}
    best = max(counts.values())
    out["framework"] = sorted(f for f, count in counts.items() if count == best)[0]
    out["n"] = n
    return out


def self_consistency(labels):
    groups = group_passes(labels)
    unanimous = 0
    total = 0
    for records in groups.values():
        for claim in CLAIMS:
            total += 1
            unanimous += len({r[claim] for r in records}) == 1
    if total == 0:
        return float("nan")
    return unanimous / total


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def precision_recall_correction(observed, precision, recall):
    if recall == 0:
        return float("nan")
    return float(min(1.0, max(0.0, observed * precision / recall)))


def mcnemar_exact(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))

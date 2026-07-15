#!/usr/bin/env python3
"""Normalize the frozen CourtListener snapshot to an exact NOS-443 corpus.

The first snapshot used a permissive substring/name filter. This script keeps
the original candidate rows auditable while rewriting the released population
and text corpus to the official NOS-443 code/description only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

sys_path = Path(__file__).resolve().parents[1] / "src"
import sys
sys.path.insert(0, str(sys_path))

from fha.reference import canonical_nos_code  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/bulk_fha_cases.jsonl")
    ap.add_argument("--text", default="data/processed/paper_corpus.jsonl")
    ap.add_argument("--excluded", default="data/validation/excluded_non_nos443.jsonl")
    args = ap.parse_args()

    raw_path, text_path = Path(args.raw), Path(args.text)
    raw = load(raw_path)
    text = load(text_path)
    kept_raw, excluded = [], []
    for row in raw:
        code = canonical_nos_code(row.get("nature_of_suit"))
        if code:
            row = {**row, "nature_of_suit_raw": row.get("nature_of_suit"),
                   "nature_of_suit": code, "nature_of_suit_code": code}
            kept_raw.append(row)
        else:
            excluded.append({"cluster_id": row.get("cluster_id"),
                             "docket_id": row.get("docket_id"),
                             "court_id": row.get("court_id"),
                             "nature_of_suit": row.get("nature_of_suit"),
                             "case_name": row.get("case_name"),
                             "exclusion_reason": "not_exact_nos443"})
    kept_ids = {r.get("cluster_id") for r in kept_raw}
    kept_text = []
    for row in text:
        if row.get("cluster_id") in kept_ids:
            row = {**row, "nature_of_suit_raw": row.get("nature_of_suit"),
                   "nature_of_suit": "443", "nature_of_suit_code": "443"}
            kept_text.append(row)
    write(raw_path, kept_raw)
    write(text_path, kept_text)
    write(Path(args.excluded), excluded)
    print(json.dumps({"candidate_rows": len(raw), "canonical_nos443": len(kept_raw),
                      "full_text_rows": len(kept_text), "excluded": len(excluded)},
                     indent=2))


if __name__ == "__main__":
    main()

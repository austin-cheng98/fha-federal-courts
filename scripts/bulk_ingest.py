#!/usr/bin/env python3
"""
Build the federal FHA case list by streaming the CourtListener bulk
dockets and opinion-clusters (NOS 443 or named 'fair housing'). Writes
data/raw/bulk_fha_cases.jsonl (metadata only; text is filled separately).

  python scripts/bulk_ingest.py [--max-cases N]
"""
import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import config                                   # noqa: E402
from fha.bulkdata import stream_bulk_rows                # noqa: E402
from fha.reference import court_to_circuit, court_level  # noqa: E402

# authoritative CSV column indices (from the bulk files' own header rows)
DOCK = {"id": 0, "date_filed": 14, "case_name": 18, "nature_of_suit": 25, "court_id": 42}
CLUS = {"id": 0, "judges": 3, "date_filed": 4, "case_name": 8, "nature_of_suit": 17,
        "citation_count": 27, "precedential_status": 28, "docket_id": 33}

def _on_retry(pos, tries, err):
    print(f"  [stream resumed @ {pos/1e9:.2f} GB after reset, retry {tries}]",
          flush=True)


def _stream_rows(table: str):
    # resilient (auto-resume on connection reset) + correct CSV dialect
    return stream_bulk_rows(table, on_retry=_on_retry)


def _is_housing(nos: str, name: str) -> bool:
    nos, name = (nos or "").lower(), (name or "").lower()
    return ("443" in nos or "housing" in nos or "fair housing" in name)


def build(max_cases: int | None = None, max_rows: int | None = None,
          max_cluster_rows: int | None = None) -> dict:
    # Pass 1: federal housing dockets -> {docket_id: meta}
    housing: dict[str, dict] = {}
    scanned = 0
    for row in _stream_rows("dockets"):
        scanned += 1
        if scanned % 1_000_000 == 0:
            print(f"  dockets scanned: {scanned:,}  housing kept: {len(housing):,}",
                  flush=True)
        if len(row) <= DOCK["court_id"]:
            continue
        court = (row[DOCK["court_id"]] or "").strip().lower()
        if court_level(court) is None:          # federal-only
            continue
        nos, name = row[DOCK["nature_of_suit"]], row[DOCK["case_name"]]
        if not _is_housing(nos, name):
            continue
        housing[row[DOCK["id"]]] = {
            "court_id": court, "circuit": court_to_circuit(court),
            "court_level": court_level(court), "nature_of_suit": nos,
            "docket_case_name": name, "docket_date_filed": row[DOCK["date_filed"]],
        }
        if max_rows and scanned >= max_rows:
            break
    print(f"Pass 1 done: {len(housing):,} federal housing dockets "
          f"(scanned {scanned:,})", flush=True)

    # Pass 2: clusters whose docket is a federal housing docket -> case records
    out_path = config.RAW / "bulk_fha_cases.jsonl"
    n = 0
    cl_scanned = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in _stream_rows("opinion-clusters"):
            cl_scanned += 1
            if max_cluster_rows and cl_scanned > max_cluster_rows:
                break
            if cl_scanned % 1_000_000 == 0:
                print(f"  clusters scanned: {cl_scanned:,}  matched: {n:,}", flush=True)
            if len(row) <= CLUS["docket_id"]:
                continue
            d = housing.get(row[CLUS["docket_id"]])
            if d is None:
                continue
            date_filed = row[CLUS["date_filed"]] or d["docket_date_filed"]
            year = int(date_filed[:4]) if date_filed[:4].isdigit() else None
            rec = {
                "cluster_id": int(row[CLUS["id"]]),
                "case_name": row[CLUS["case_name"]] or d["docket_case_name"],
                "court_id": d["court_id"], "circuit": d["circuit"],
                "court_level": d["court_level"], "date_filed": date_filed,
                "year": year, "docket_id": row[CLUS["docket_id"]],
                "nature_of_suit": d["nature_of_suit"],
                "judges": row[CLUS["judges"]],
                "precedential_status": row[CLUS["precedential_status"]],
                "citations": [], "opinion_ids": [],
                "text": "", "text_len": 0, "text_source": "bulk_metadata",
                "source": "courtlistener_bulk",
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if max_cases and n >= max_cases:
                break
    print(f"Pass 2 done: wrote {n:,} federal FHA cases -> {out_path}", flush=True)
    return {"n_dockets_housing": len(housing), "n_cases": n, "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-cases", type=int, default=None,
                    help="stop after N matched cases (quick validation)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="stop scanning dockets after N rows (quick validation)")
    ap.add_argument("--max-cluster-rows", type=int, default=None,
                    help="stop scanning clusters after N rows (quick validation)")
    args = ap.parse_args()
    build(max_cases=args.max_cases, max_rows=args.max_rows,
          max_cluster_rows=args.max_cluster_rows)


if __name__ == "__main__":
    main()

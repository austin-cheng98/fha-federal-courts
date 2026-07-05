#!/usr/bin/env python3
"""
Fetch opinion text by streaming the bulk opinions dump (no API, no rate
limit), keeping only cluster_ids from bulk_ingest.py. Writes
data/raw/fha_text_corpus.jsonl.

  python scripts/fetch_bulk_text_from_dump.py [--max-rows N]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import config                       # noqa: E402
from fha.bulkdata import stream_bulk_rows, read_local_bulk_rows  # noqa: E402

OP = {"id": 0, "type": 6, "plain_text": 11, "html_with_citations": 18,
      "html": 12, "xml_harvard": 16, "cluster_id": 21}
_TAG = re.compile(r"<[^>]+>")


def _text(row) -> str:
    for idx in (OP["plain_text"], OP["html_with_citations"], OP["html"], OP["xml_harvard"]):
        v = row[idx] if len(row) > idx else ""
        if v and v.strip():
            return v.strip() if idx == OP["plain_text"] else _TAG.sub(" ", v).strip()
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--local", default=None,
                    help="scan a locally-downloaded opinions .csv.bz2 instead of "
                         "streaming from S3 (reliable; pairs with the disk download)")
    args = ap.parse_args()

    bulk_path = config.RAW / "bulk_fha_cases.jsonl"
    cases = {json.loads(l)["cluster_id"]: json.loads(l)
             for l in bulk_path.open(encoding="utf-8") if l.strip()}
    out = config.RAW / "fha_text_corpus.jsonl"

    # --- resume: keep clusters already filled with text, re-scan only the rest ---
    have: dict[int, dict] = {}
    if out.exists():
        for l in out.open(encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("text") and r["cluster_id"] in cases:
                have[r["cluster_id"]] = r
    # rewrite the file deduped (clean base to append onto)
    with out.open("w", encoding="utf-8") as fh:
        for r in have.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    want = set(cases) - set(have)
    print(f"target clusters: {len(cases)} | already have text: {len(have)} | "
          f"to find: {len(want)}", flush=True)
    if not want:
        print("all clusters already have text; nothing to scan.", flush=True)
        return

    scanned = 0

    def _on_retry(pos, tries, err):
        print(f"  [opinions stream resumed @ {pos/1e9:.1f} GB, retry {tries}]",
              flush=True)

    # Append each found opinion AS DISCOVERED so a kill preserves progress and a
    # re-run resumes (skipping what's already filled). Stop early once all found.
    rows = (read_local_bulk_rows(args.local) if args.local
            else stream_bulk_rows("opinions", on_retry=_on_retry))
    fh = out.open("a", encoding="utf-8")
    try:
        for row in rows:
            scanned += 1
            if scanned % 5_000_000 == 0:
                print(f"  opinions scanned: {scanned:,} | remaining: {len(want)}",
                      flush=True)
            if args.max_rows and scanned >= args.max_rows:
                break
            if len(row) <= OP["cluster_id"]:
                continue
            try:
                cid = int(row[OP["cluster_id"]])
            except ValueError:
                continue
            if cid not in want:
                continue
            t = _text(row)
            if not t:
                continue
            rec = {**cases[cid], "text": t, "text_len": len(t),
                   "text_source": "bulk_full"}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            want.discard(cid)
            if not want:                         # found them all
                print("  all target clusters found; stopping scan early.", flush=True)
                break
    finally:
        fh.close()

    have_now = sum(1 for l in out.open(encoding="utf-8") if l.strip())
    note = "" if want else " (complete)"
    print(f"\ntext corpus now {have_now}/{len(cases)} clusters "
          f"(scanned {scanned:,} opinions){note} -> {out}", flush=True)
    if want:
        print(f"{len(want)} clusters still unfilled; re-run to resume "
              f"(or they have no text in the dump).", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Harvest a real FHA corpus from the CourtListener API to JSONL
(rate-limited, resumable), then run the pipeline with --source existing.

  python scripts/collect_live.py --max-pages 60 --out data/raw/live_corpus.jsonl
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fha.courtlistener import CourtListenerClient  # noqa: E402
from fha import config  # noqa: E402

# Each query targets a distinct FHA theory; the union maximizes recall.
DEFAULT_QUERIES = [
    '"fair housing act"',
    '"42 U.S.C. 3604" OR "42 U.S.C. 3605"',
    '"disparate impact" housing discrimination',
    '"reasonable accommodation" "fair housing"',
    '"exclusionary zoning" OR "fair housing" land use',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.RAW / "live_corpus.jsonl"))
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--filed-after", default="1990-01-01")
    ap.add_argument("--filed-before", default="2024-12-31")
    ap.add_argument("--no-text", action="store_true",
                    help="metadata only (skip per-opinion text fetches)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent text-fetch threads. KEEP AT 1: CourtListener "
                         "throttles the opinions endpoint, and concurrency only "
                         "triggers more 429s (and drops text). Sequential + the "
                         "capped back-off self-paces to the sustainable rate.")
    args = ap.parse_args()

    import json
    import requests

    client = CourtListenerClient()
    print(f"CourtListener token: {'set' if client.authenticated else 'ANONYMOUS'}",
          flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- resume: skip cluster_ids already on disk, append rather than truncate ---
    skip_ids: set[int] = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                skip_ids.add(int(json.loads(line)["cluster_id"]))
            except (ValueError, KeyError):
                continue
        print(f"resume: {len(skip_ids)} cases already harvested; skipping them",
              flush=True)
    t0 = time.time()

    def on_progress(done, total):
        if done == 0:
            print(f"  phase 1: {total} NEW federal cases queued for text fetch",
                  flush=True)
        elif done % 20 == 0 or done == total:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {done}/{total} fetched ({rate:.2f}/s, "
                  f"throttle_hits={client.throttle_hits})", flush=True)

    n = 0
    throttled = False
    with out.open("a", encoding="utf-8") as fh:
        try:
            for rec in client.harvest_parallel(
                    DEFAULT_QUERIES, filed_after=args.filed_after,
                    filed_before=args.filed_before, max_pages=args.max_pages,
                    workers=args.workers, fetch_text=not args.no_text,
                    skip_ids=skip_ids, on_progress=on_progress):
                fh.write(rec.to_json() + "\n")
                fh.flush()
                n += 1
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) == 429:
                throttled = True
            else:
                raise
    total = len(skip_ids) + n
    print(f"\nadded {n} new cases ({total} total) in {time.time()-t0:.0f}s -> {out}",
          flush=True)
    if throttled:
        print("STOPPED EARLY: CourtListener rate limit hit. Progress saved. "
              "Re-run this exact command later (e.g. in ~1h) to resume from where "
              "it left off -- already-harvested cases are skipped automatically.",
              flush=True)


if __name__ == "__main__":
    main()

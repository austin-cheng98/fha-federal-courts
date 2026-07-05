#!/usr/bin/env python3
"""
Fetch opinion text for the bulk case list via the API (resumable,
quota-graceful). Writes data/raw/fha_text_corpus.jsonl.

  python scripts/fetch_bulk_text.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import config                          # noqa: E402
from fha.courtlistener import CourtListenerClient  # noqa: E402

BULK = config.RAW / "bulk_fha_cases.jsonl"
OUT = config.RAW / "fha_text_corpus.jsonl"


def _load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="max cases this run")
    ap.add_argument("--bulk", default=str(BULK))
    args = ap.parse_args()

    cases = _load_jsonl(Path(args.bulk))
    if not cases:
        print(f"no bulk case list at {args.bulk}; run scripts/bulk_ingest.py first")
        return
    done = {r["cluster_id"] for r in _load_jsonl(OUT) if r.get("text")}
    todo = [c for c in cases if c["cluster_id"] not in done]
    print(f"bulk cases: {len(cases)} | already have text: {len(done)} | "
          f"to fetch: {len(todo)}", flush=True)

    client = CourtListenerClient()
    if not client.authenticated:
        print("WARNING: no token -> only snippets/401s. Put it in config/.cl_token.")

    n = 0
    throttled = False
    with OUT.open("a", encoding="utf-8") as fh:
        for c in todo:
            cid = c["cluster_id"]
            try:
                cluster = client.get_cluster(cid)
                ids = []
                for u in cluster.get("sub_opinions", []) or []:
                    try:
                        ids.append(int(u.rstrip("/").split("/")[-1]))
                    except (ValueError, AttributeError):
                        pass
                texts = []
                for oid in ids:
                    texts.append(client._opinion_text(client.get_opinion(oid)))
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) == 429:
                    throttled = True
                    break
                continue            # 404 etc: skip this case
            text = "\n\n".join(t for t in texts if t)
            rec = {**c, "opinion_ids": ids, "text": text, "text_len": len(text),
                   "text_source": "full" if text else "none",
                   "judges": c.get("judges") or cluster.get("judges", "")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if args.limit and n >= args.limit:
                break

    total = len(done) + n
    print(f"\nfetched text for {n} new cases ({total}/{len(cases)} total) -> {OUT}",
          flush=True)
    if throttled:
        print("STOPPED: rate limit (429). Progress saved; re-run later to resume.",
              flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the FHA pipeline. --source synthetic|live|existing; add --real-housing
for live Census/HMDA.

  python scripts/run_pipeline.py --source synthetic
  python scripts/run_pipeline.py --source live --max-pages 40 --real-housing
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fha.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "live", "existing"],
                    default="synthetic")
    ap.add_argument("--query", default='"fair housing act"')
    ap.add_argument("--max-pages", type=int, default=20,
                    help="CourtListener search pages (20 results/page)")
    ap.add_argument("--filed-after", default="1990-01-01")
    ap.add_argument("--filed-before", default="2024-12-31")
    ap.add_argument("--real-housing", action="store_true",
                    help="pull Census/HMDA live instead of the synthetic panel")
    ap.add_argument("--backend", choices=["tfidf", "legalbert"], default="tfidf")
    args = ap.parse_args()

    live_kwargs = None
    if args.source == "live":
        live_kwargs = {"query": args.query, "max_pages": args.max_pages,
                       "filed_after": args.filed_after,
                       "filed_before": args.filed_before, "fetch_text": True}

    rep = run(source=args.source, live_kwargs=live_kwargs,
              real_housing=args.real_housing, embedding_backend=args.backend)

    # console digest
    print("\n================  FHA PIPELINE  ================")
    print(f"source={rep['source']}  cases={rep['step3_n_cases']}  "
          f"FEII cells={rep['step5_feii_cells']}")
    t = rep["step7_twfe"]
    print(f"\n[Step 7] TWFE  outcome ~ FEII  (circuit+year FE, cluster SE)")
    print(f"         coef={t['coef']:+.4f}  se={t['se']:.4f}  "
          f"p={t['p']:.4f}  n={t['n']}")
    d = rep["step7_did_2x2"]
    print(f"[Step 7] DiD 2x2 treated:post  coef={d['coef']:+.4f}  p={d['p']:.4f}")
    print(f"\nOutputs in {rep['outputs_dir']}/  (tables, figures, SUMMARY.md)")


if __name__ == "__main__":
    main()

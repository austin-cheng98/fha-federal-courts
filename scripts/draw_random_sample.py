#!/usr/bin/env python3
"""Regenerate the frozen 150-cluster random sample and verify the committed index."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import config  # noqa: E402
from fha.classify import rule_is_fha  # noqa: E402

SEED = 20260720
N_SAMPLE = 150
NOTE = ("uniform draw from the 658 paper_corpus clusters not in the "
        "120-case human gold set")
INDEX_PATH = config.VALIDATION / "random_sample_index.json"


def load_corpus():
    records = [json.loads(line) for line in
               open(config.PROCESSED / "paper_corpus.jsonl") if line.strip()]
    return {r["cluster_id"]: r for r in records}


def build_frame(corpus_ids, gold_ids):
    return sorted(set(corpus_ids) - set(gold_ids))


def draw_sample(frame, seed=SEED, n=N_SAMPLE):
    if n > len(frame):
        raise ValueError(f"cannot draw {n} from a frame of {len(frame)}")
    rng = random.Random()
    rng.seed(seed)
    return sorted(rng.sample(frame, n))


def compare(drawn, frame_size, committed):
    diffs = []
    if committed.get("seed") != SEED:
        diffs.append(f"seed: committed {committed.get('seed')} != expected {SEED}")
    if committed.get("n") != N_SAMPLE:
        diffs.append(f"n: committed {committed.get('n')} != expected {N_SAMPLE}")
    if committed.get("frame_size") != frame_size:
        diffs.append(f"frame_size: committed {committed.get('frame_size')} "
                     f"!= recomputed {frame_size}")
    committed_ids = list(committed.get("cluster_ids", []))
    if sorted(committed_ids) != committed_ids:
        diffs.append("cluster_ids: committed list is not sorted ascending")
    if committed_ids != drawn:
        missing = sorted(set(drawn) - set(committed_ids))
        extra = sorted(set(committed_ids) - set(drawn))
        diffs.append(f"cluster_ids: {len(committed_ids)} committed vs "
                     f"{len(drawn)} redrawn; {len(missing)} redrawn ids absent "
                     f"from the index, {len(extra)} committed ids not redrawn")
        if missing:
            diffs.append(f"  missing from index (first 10): {missing[:10]}")
        if extra:
            diffs.append(f"  extra in index (first 10): {extra[:10]}")
    return diffs


def write_index(drawn, frame_size):
    payload = {"seed": SEED, "frame_size": frame_size, "n": N_SAMPLE,
               "cluster_ids": drawn, "note": NOTE}
    json.dump(payload, INDEX_PATH.open("w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="overwrite the committed index instead of verifying it")
    args = ap.parse_args()

    corpus = load_corpus()
    human = json.load(open(config.VALIDATION / "gold_human_codings.json"))
    gold_ids = [c["case_id"] for c in human["primary"]]

    frame = build_frame(corpus, gold_ids)
    drawn = draw_sample(frame)

    print(f"corpus clusters:        {len(corpus)}")
    print(f"human primary gold set: {len(gold_ids)} "
          f"({len(set(gold_ids) & set(corpus))} present in corpus)")
    print(f"sampling frame:         {len(frame)}")
    print(f"drawn (seed={SEED}):    {len(drawn)}")

    substantive = [cid for cid in drawn if rule_is_fha(corpus[cid])]
    share = len(substantive) / len(drawn)
    print(f"rule_is_fha substantive: {len(substantive)} ({share:.1%})")

    if args.write:
        write_index(drawn, len(frame))
        print(f"\nWROTE {INDEX_PATH}")
        return 0

    if not INDEX_PATH.exists():
        print(f"\nFAIL: {INDEX_PATH} does not exist. "
              f"Re-run with --write to create it.")
        return 1

    committed = json.load(open(INDEX_PATH))
    diffs = compare(drawn, len(frame), committed)
    if diffs:
        print(f"\nFAIL: redrawn sample does not match {INDEX_PATH}")
        for d in diffs:
            print(f"  {d}")
        print("\nThe committed sample is not reproducible from "
              f"seed={SEED}, n={N_SAMPLE} over the current frame. "
              "Re-run with --write only if the frame legitimately changed.")
        return 1

    print(f"\nOK: {INDEX_PATH} matches the redrawn sample exactly "
          f"({len(drawn)} cluster_ids, frame {len(frame)}, seed {SEED}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

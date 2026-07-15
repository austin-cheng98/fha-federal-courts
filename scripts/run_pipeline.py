#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fha.pipeline import run


parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=("paper", "synthetic"), default="paper")
args = parser.parse_args()
print(json.dumps(run(args.source), indent=2, default=str))

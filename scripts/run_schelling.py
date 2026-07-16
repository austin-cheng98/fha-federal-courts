#!/usr/bin/env python3
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fha.schelling import run_scenarios


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "feii_panel.csv"
OUTPUT = ROOT / "outputs" / "schelling_scenarios.json"


def main():
    values = defaultdict(list)
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["unit"]].append(float(row["FEII"]))
    feii = {unit: sum(items) / len(items) for unit, items in values.items()}
    scenarios = run_scenarios(feii)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(scenarios, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "circuits": scenarios["feii_reference"]["circuits"],
        "phase_cells": len(scenarios["phase_sweep"]),
    }, indent=2))


if __name__ == "__main__":
    main()

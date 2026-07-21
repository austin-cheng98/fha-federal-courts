#!/usr/bin/env python3
"""Per-circuit doctrinal prevalence and the cross-circuit split test (chi-square)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import config  # noqa: E402
from fha import circuits  # noqa: E402


def main():
    prevalence = circuits.circuit_prevalence()
    split = circuits.circuit_split()

    print("== per-circuit claim prevalence (substantive clusters, n >= 5) ==")
    print(prevalence.to_string(index=False))
    print("\n== circuit split: chi-square of circuit x claim independence ==")
    for construct, s in split.items():
        star = " *" if s["p"] < 0.05 else ""
        print(f" {construct:26s} chi2={s['chi2']:6.1f} df={s['dof']} "
              f"p={s['p']:.4f} Cramer's V={s['cramers_v']:.2f} (n={s['n']}){star}")
    print("\nPer-circuit samples are small (tens of clusters); the split is a measured"
          "\ndivergence in doctrinal mix, not a ranking of individual circuits.")

    out = config.OUTPUTS / "paper" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(out / "circuit_prevalence.csv", index=False)
    print(f"\nwrote {out / 'circuit_prevalence.csv'}")


if __name__ == "__main__":
    main()

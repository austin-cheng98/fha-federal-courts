import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fha import circuits


def test_circuit_split_reproduces_paper_numbers():
    split = circuits.circuit_split()
    di = split["disparate_impact"]
    assert di["p"] < 0.001
    assert abs(di["chi2"] - 30.5) < 0.5
    assert abs(di["cramers_v"] - 0.27) < 0.02
    dt = split["disparate_treatment"]
    assert dt["p"] < 0.01
    assert abs(dt["chi2"] - 27.6) < 0.5


def test_circuit_prevalence_shows_divergence():
    prev = circuits.circuit_prevalence()
    assert 8 <= len(prev) <= 13          # ~11 numbered circuits with enough clusters
    di = prev["disparate_impact"]
    assert di.max() - di.min() > 0.30    # 0.08 (10th) to 0.54 (7th)
    assert di.min() < 0.15 and di.max() > 0.45

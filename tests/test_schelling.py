import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fha.schelling import feii_reference, run_phase_sweep, simulate


def test_feii_reference_records_scale_without_circuit_mapping():
    reference = feii_reference({"1": -0.2, "2": 0.4})
    assert reference["circuits"] == 2
    assert reference["feii_min"] == -0.2
    assert reference["feii_max"] == 0.4
    assert reference["access_scenarios"] == [0.15, 0.325, 0.50, 0.675, 0.85]


def test_seeded_schelling_run_is_bounded():
    outcome = simulate(0.30, 0.50, size=8, max_steps=20, seed=7)
    assert 0.0 <= outcome["segregation"] <= 1.0
    assert 0 <= outcome["cross_group_admissions"] <= outcome["cross_group_attempts"]
    assert 1 <= outcome["steps"] <= 20


def test_phase_sweep_crosses_preference_and_access():
    rows = run_phase_sweep(
        tolerances=(0.20,), access_levels=(0.15, 0.85), replications=2, seed=1
    )
    assert len(rows) == 2
    assert {row["access"] for row in rows} == {0.15, 0.85}

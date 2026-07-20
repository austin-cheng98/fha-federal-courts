from __future__ import annotations

import numpy as np


def feii_reference(feii_by_circuit):
    """Record the observed legal-signal span without assigning outcomes to circuits."""
    values = np.asarray(list(feii_by_circuit.values()), dtype=float)
    return {
        "circuits": int(values.size),
        "feii_min": float(values.min()),
        "feii_max": float(values.max()),
        "access_scenarios": [0.15, 0.325, 0.50, 0.675, 0.85],
    }


def _same_share(grid, row, col, kind=None):
    kind = grid[row, col] if kind is None else kind
    r0, r1 = max(0, row - 1), min(grid.shape[0], row + 2)
    c0, c1 = max(0, col - 1), min(grid.shape[1], col + 2)
    occupied = 0
    same_kind = 0
    for neighbor_row in range(r0, r1):
        for neighbor_col in range(c0, c1):
            neighbor = grid[neighbor_row, neighbor_col]
            if neighbor >= 0:
                occupied += 1
                same_kind += int(neighbor == kind)
    if occupied == 0:
        return 0.5
    return same_kind / occupied


def _segregation(grid):
    shares = [_same_share(grid, row, col)
              for row, col in zip(*np.where(grid >= 0))]
    return float(np.mean(shares))


def simulate(
    tolerance,
    access,
    barrier=0.80,
    turnover=0.04,
    size=20,
    vacancy_rate=0.10,
    max_steps=250,
    max_moves_per_step=16,
    seed=0,
):
    rng = np.random.default_rng(seed)
    total = size * size
    vacancies = int(round(total * vacancy_rate))
    agents = total - vacancies
    kinds = np.concatenate((
        np.zeros(agents // 2, dtype=int),
        np.ones(agents - agents // 2, dtype=int),
        -np.ones(vacancies, dtype=int),
    ))
    rng.shuffle(kinds)
    grid = kinds.reshape((size, size))
    vacancies = [tuple(position) for position in np.argwhere(grid < 0)]
    cross_group_attempts = 0
    cross_group_admissions = 0
    moves = 0

    for step in range(max_steps):
        positions = np.argwhere(grid >= 0)
        sample_size = min(max_moves_per_step * 3, len(positions))
        sampled = positions[rng.choice(len(positions), size=sample_size, replace=False)]
        active = []
        for row, col in sampled:
            if _same_share(grid, int(row), int(col)) < tolerance or rng.random() < turnover:
                active.append((int(row), int(col)))
                if len(active) == max_moves_per_step:
                    break
        for row, col in active:
            kind = grid[row, col]
            grid[row, col] = -1
            vacancies.append((row, col))
            candidates = list(vacancies)
            rng.shuffle(candidates)
            moved = False
            for new_row, new_col in candidates:
                if (new_row, new_col) == (row, col):
                    continue
                grid[new_row, new_col] = kind
                share = _same_share(grid, new_row, new_col, kind)
                grid[new_row, new_col] = -1
                if share < tolerance:
                    continue
                if share < 0.5:
                    cross_group_attempts += 1
                    if rng.random() < barrier * (1.0 - access):
                        continue
                    cross_group_admissions += 1
                grid[new_row, new_col] = kind
                moved = True
                moves += 1
                vacancies.remove((new_row, new_col))
                break
            if not moved:
                grid[row, col] = kind
                vacancies.remove((row, col))

    return {
        "tolerance": float(tolerance),
        "access": float(access),
        "segregation": _segregation(grid),
        "cross_group_attempts": int(cross_group_attempts),
        "cross_group_admissions": int(cross_group_admissions),
        "moves": int(moves),
        "steps": int(step + 1),
    }


def run_phase_sweep(
    tolerances=(0.20, 0.30, 0.45),
    access_levels=(0.15, 0.325, 0.50, 0.675, 0.85),
    replications=40,
    seed=20260715,
):
    rows = []
    for tolerance_index, tolerance in enumerate(tolerances):
        for access_index, access in enumerate(access_levels):
            outcomes = [
                simulate(
                    tolerance=tolerance,
                    access=access,
                    seed=seed + 10000 * tolerance_index + 1000 * access_index + rep,
                )
                for rep in range(replications)
            ]
            rows.append({
                "tolerance": float(tolerance),
                "access": float(access),
                "segregation_mean": float(np.mean([item["segregation"] for item in outcomes])),
                "segregation_sd": float(np.std([item["segregation"] for item in outcomes], ddof=1)),
                "cross_group_admission_mean": float(np.mean([
                    item["cross_group_admissions"] / max(item["cross_group_attempts"], 1)
                    for item in outcomes
                ])),
            })
    return rows


def run_scenarios(feii_by_circuit, replications=40, seed=20260715):
    return {
        "meta": {
            "barrier": 0.80,
            "turnover": 0.04,
            "grid_size": 20,
            "vacancy_rate": 0.10,
            "max_steps": 250,
            "replications": replications,
            "seed": seed,
        },
        "feii_reference": feii_reference(feii_by_circuit),
        "phase_sweep": run_phase_sweep(replications=replications, seed=seed),
    }

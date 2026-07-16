from __future__ import annotations

import numpy as np


def normalize_feii(values):
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if hi == lo:
        return np.full(values.shape, 0.5)
    return (values - lo) / (hi - lo)


def tolerance_from_feii(values, low=0.28, high=0.48):
    scaled = normalize_feii(values)
    return high - (high - low) * scaled


def _same_share(grid, row, col):
    kind = grid[row, col]
    if kind < 0:
        return np.nan
    r0, r1 = max(0, row - 1), min(grid.shape[0], row + 2)
    c0, c1 = max(0, col - 1), min(grid.shape[1], col + 2)
    neighborhood = grid[r0:r1, c0:c1]
    occupied = neighborhood[neighborhood >= 0]
    if occupied.size <= 1:
        return 1.0
    return float(np.mean(occupied == kind))


def _unhappy(grid, tolerance):
    positions = []
    for row, col in zip(*np.where(grid >= 0)):
        if _same_share(grid, row, col) < tolerance:
            positions.append((int(row), int(col)))
    return positions


def _segregation(grid):
    shares = [_same_share(grid, row, col)
              for row, col in zip(*np.where(grid >= 0))]
    return float(np.nanmean(shares))


def simulate(tolerance, size=40, vacancy_rate=0.10, max_steps=500, seed=0):
    rng = np.random.default_rng(seed)
    total = size * size
    vacancies = int(round(total * vacancy_rate))
    agents = total - vacancies
    kinds = np.concatenate((np.zeros(agents // 2, dtype=int),
                            np.ones(agents - agents // 2, dtype=int),
                            -np.ones(vacancies, dtype=int)))
    rng.shuffle(kinds)
    grid = kinds.reshape((size, size))

    for step in range(max_steps):
        unhappy = _unhappy(grid, tolerance)
        if not unhappy:
            break
        rng.shuffle(unhappy)
        moved = False
        vacant = list(zip(*np.where(grid < 0)))
        rng.shuffle(vacant)
        for row, col in unhappy:
            kind = grid[row, col]
            for new_row, new_col in vacant:
                grid[new_row, new_col] = kind
                acceptable = _same_share(grid, new_row, new_col) >= tolerance
                grid[new_row, new_col] = -1
                if acceptable:
                    grid[new_row, new_col] = kind
                    grid[row, col] = -1
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break

    remaining_unhappy = _unhappy(grid, tolerance)
    return {
        "tolerance": float(tolerance),
        "segregation": _segregation(grid),
        "unhappy_share": len(remaining_unhappy) / max(agents, 1),
        "steps": step + 1,
    }


def run_scenarios(feii_by_circuit, replications=40, seed=20260715,
                  size=20, max_steps=250):
    circuits = list(feii_by_circuit)
    feii = np.asarray([feii_by_circuit[circuit] for circuit in circuits], dtype=float)
    tolerances = tolerance_from_feii(feii)
    rows = []
    for circuit, value, tolerance in zip(circuits, feii, tolerances):
        outcomes = [simulate(tolerance, size=size, max_steps=max_steps, seed=seed + i)
                     for i in range(replications)]
        rows.append({
            "circuit": circuit,
            "FEII": float(value),
            "tolerance": float(tolerance),
            "segregation_mean": float(np.mean([x["segregation"] for x in outcomes])),
            "segregation_sd": float(np.std([x["segregation"] for x in outcomes], ddof=1)),
            "unhappy_share_mean": float(np.mean([x["unhappy_share"] for x in outcomes])),
        })
    return rows

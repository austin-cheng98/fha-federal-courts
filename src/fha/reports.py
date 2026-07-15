from . import config


def write_summary(cases, feii_cells, housing_rows, twfe):
    lines = [
        "# Reproducibility summary",
        "",
        f"- cases: {cases}",
        f"- FEII cells: {feii_cells}",
        f"- housing rows: {housing_rows}",
    ]
    if twfe.get("note"):
        lines.append(f"- real-data inference: not estimated ({twfe['note']})")
    else:
        lines.append("- estimator check: completed")
    path = config.OUTPUTS / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n")
    return path

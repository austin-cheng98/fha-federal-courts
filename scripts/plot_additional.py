#!/usr/bin/env python3
"""Render supplementary figures from frozen inputs and stipulated scenarios."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


FEII_INPUT = ROOT / "data" / "processed" / "feii_panel.csv"
HOUSING_INPUT = ROOT / "data" / "external" / "housing_panel.csv"
SCENARIO_INPUT = ROOT / "outputs" / "schelling_scenarios.json"
HEATMAP_OUTPUT = ROOT / "outputs" / "feii_component_heatmap.png"
SENSITIVITY_OUTPUT = ROOT / "outputs" / "schelling_sensitivity.png"
HOUSING_OUTPUT = ROOT / "outputs" / "housing_feasibility_matrix.png"


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def circuit_means():
    values = defaultdict(lambda: defaultdict(list))
    with FEII_INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["unit_type"] != "circuit":
                continue
            circuit = str(int(float(row["unit"])))
            for key in ("FEII", "z_opinion_volume", "z_outcome_cue_rate", "z_remedy_cue_intensity"):
                values[circuit][key].append(float(row[key]))
    circuits = sorted(values, key=int)
    return circuits, {
        circuit: {key: float(np.mean(values[circuit][key])) for key in values[circuit]}
        for circuit in circuits
    }


def text_size(draw, text, typeface):
    box = draw.textbbox((0, 0), text, font=typeface)
    return box[2] - box[0], box[3] - box[1]


def draw_rotated_label(image, text, xy, typeface, fill):
    width, height = text_size(ImageDraw.Draw(image), text, typeface)
    layer = Image.new("RGBA", (width + 20, height + 20), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((10, 10), text, font=typeface, fill=fill)
    layer = layer.rotate(90, expand=True)
    image.paste(layer, xy, layer)


def render_component_heatmap():
    circuits, values = circuit_means()
    width, height = 1700, 1000
    left, right, top, bottom = 240, 180, 150, 125
    plot_width = width - left - right
    plot_height = height - top - bottom
    columns = [
        ("opinion volume", "z_opinion_volume"),
        ("outcome-cue rate", "z_outcome_cue_rate"),
        ("remedy-cue intensity", "z_remedy_cue_intensity"),
    ]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    navy, rust, gray = "#243B53", "#B84A32", "#536471"
    title_font, axis_font, tick_font, cell_font = font(34, True), font(25), font(20), font(20)
    draw.text((left, 30), "Circuit-level FEII components", fill=navy, font=title_font)
    draw.text((left, 78), "Circuit means show the composite is not a single legal dimension", fill=gray, font=font(19))

    cell_width = plot_width // len(columns)
    row_height = plot_height // len(circuits)

    def cell_color(value):
        value = max(-2.0, min(2.0, value)) / 2.0
        if value < 0:
            weight = value + 1.0
            start, end = (36, 59, 83), (238, 242, 245)
        else:
            weight = value
            start, end = (238, 242, 245), (184, 74, 50)
        return tuple(int(start[i] + (end[i] - start[i]) * weight) for i in range(3))

    for column_index, (label, key) in enumerate(columns):
        x0 = left + column_index * cell_width
        draw.text((x0 + cell_width // 2 - text_size(draw, label, axis_font)[0] // 2, top - 48), label, fill=navy, font=axis_font)
        for row_index, circuit in enumerate(circuits):
            y0 = top + row_index * row_height
            value = values[circuit][key]
            draw.rectangle((x0, y0, x0 + cell_width, y0 + row_height), fill=cell_color(value), outline="white", width=2)
            label = f"{value:+.2f}"
            text_width, text_height = text_size(draw, label, cell_font)
            fill = "white" if abs(value) > 1.05 else navy
            draw.text((x0 + (cell_width - text_width) / 2, y0 + (row_height - text_height) / 2 - 2), label, fill=fill, font=cell_font)

    for row_index, circuit in enumerate(circuits):
        y = top + row_index * row_height + row_height // 2
        label = f"C{circuit}"
        text_width, text_height = text_size(draw, label, axis_font)
        draw.text((left - 18 - text_width, y - text_height / 2), label, fill=navy, font=axis_font)

    legend_x, legend_y, legend_height = width - right + 35, top, plot_height
    for index in range(legend_height):
        value = 2.0 - 4.0 * index / max(legend_height - 1, 1)
        draw.line((legend_x, legend_y + index, legend_x + 28, legend_y + index), fill=cell_color(value), width=1)
    draw.rectangle((legend_x, legend_y, legend_x + 28, legend_y + legend_height), outline=gray, width=1)
    for value in (2, 1, 0, -1, -2):
        y = legend_y + (2 - value) / 4 * legend_height
        draw.text((legend_x + 40, y - 12), f"{value:+d}", fill=gray, font=tick_font)
    draw_rotated_label(image, "standardized component", (width - 55, top + 210), axis_font, gray)
    HEATMAP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(HEATMAP_OUTPUT, "PNG", optimize=True)


def render_sensitivity():
    scenarios = json.loads(SCENARIO_INPUT.read_text(encoding="utf-8"))
    rows = scenarios["phase_sweep"]
    tolerances = sorted({row["tolerance"] for row in rows})
    access_levels = sorted({row["access"] for row in rows})
    lookup = {(row["tolerance"], row["access"]): row for row in rows}
    values = [row["segregation_mean"] for row in rows]

    width, height = 1700, 1000
    left, right, top, bottom = 190, 130, 140, 155
    x0, x1 = min(access_levels) - 0.05, max(access_levels) + 0.05
    y0, y1 = min(values) - 0.03, max(values) + 0.03
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    navy, grid, gray = "#243B53", "#D9E1E8", "#536471"
    colors = ["#2F8F83", "#C9A227", "#B84A32"]
    title_font, axis_font, tick_font, legend_font = font(34, True), font(25), font(20), font(20)

    def px(value):
        return int(left + (value - x0) / (x1 - x0) * (width - left - right))

    def py(value):
        return int(height - bottom - (value - y0) / (y1 - y0) * (height - top - bottom))

    draw.text((left, 30), "Access effects vary by sorting preference", fill=navy, font=title_font)
    draw.text((left, 78), "The access channel matters only under preference rules that permit cross-group moves", fill=gray, font=font(19))
    for tick in access_levels:
        x = px(tick)
        draw.line((x, top, x, height - bottom), fill=grid, width=2)
        label = f"{tick:.2f}"
        text_width, _ = text_size(draw, label, tick_font)
        draw.text((x - text_width / 2, height - bottom + 18), label, fill=gray, font=tick_font)
    for tick in np.linspace(round(y0, 2), round(y1, 2), 5):
        y = py(float(tick))
        draw.line((left, y, width - right, y), fill=grid, width=2)
        label = f"{tick:.2f}"
        text_width, text_height = text_size(draw, label, tick_font)
        draw.text((left - 18 - text_width, y - text_height / 2), label, fill=gray, font=tick_font)
    draw.line((left, top, left, height - bottom), fill=navy, width=4)
    draw.line((left, height - bottom, width - right, height - bottom), fill=navy, width=4)

    legend_x, legend_y = width - 465, top + 24
    for tolerance, color in zip(tolerances, colors):
        points = [
            (px(access), py(lookup[(tolerance, access)]["segregation_mean"]))
            for access in access_levels
        ]
        for first, second in zip(points, points[1:]):
            draw.line((*first, *second), fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="white", width=2)
        draw.line((legend_x, legend_y + 12, legend_x + 45, legend_y + 12), fill=color, width=5)
        draw.text((legend_x + 60, legend_y), f"preference threshold {tolerance:.2f}", fill=navy, font=legend_font)
        legend_y += 38

    xlabel, ylabel = "cross-group access scenario", "simulated same-type neighbor share"
    text_width, _ = text_size(draw, xlabel, axis_font)
    draw.text(((left + width - right - text_width) / 2, height - 70), xlabel, fill=navy, font=axis_font)
    draw_rotated_label(image, ylabel, (24, top + 150), axis_font, navy)
    draw.text((left, height - 112), "40 replications per cell; preference threshold held fixed within each line", fill=gray, font=font(18))
    SENSITIVITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(SENSITIVITY_OUTPUT, "PNG", optimize=True)


def render_housing_feasibility():
    with HOUSING_INPUT.open(newline="", encoding="utf-8") as handle:
        housing = list(csv.DictReader(handle))
    with FEII_INPUT.open(newline="", encoding="utf-8") as handle:
        legal = [row for row in csv.DictReader(handle) if row["unit_type"] == "circuit"]
    housing_cells = {(row["circuit"], int(row["year"])) for row in housing}
    legal_cells = {(str(int(float(row["unit"]))), int(row["year"])) for row in legal}
    matches = housing_cells & legal_cells
    circuits = [str(value) for value in range(1, 12)] + ["DC"]
    years = sorted({int(row["year"]) for row in housing})

    width, height = 1600, 1000
    left, right, top, bottom = 290, 180, 190, 185
    plot_width, plot_height = width - left - right, height - top - bottom
    cell_width, cell_height = plot_width / len(years), plot_height / len(circuits)
    navy, teal, pale, gray = "#243B53", "#2F8F83", "#E7EDF1", "#536471"
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, tick_font = font(34, True), font(25), font(21)

    draw.text((left, 35), "Legal–housing merge feasibility", fill=navy, font=title_font)
    draw.text((left, 82), f"{len(matches)} matched circuit-year cells, all in {', '.join(str(year) for year in sorted({year for _, year in matches}))}; no within-circuit panel", fill=gray, font=font(19))
    for column_index, year in enumerate(years):
        x0 = left + column_index * cell_width
        label = str(year)
        text_width, _ = text_size(draw, label, axis_font)
        draw.text((x0 + cell_width / 2 - text_width / 2, top - 48), label, fill=navy, font=axis_font)
        for row_index, circuit in enumerate(circuits):
            y0 = top + row_index * cell_height
            cell = (circuit, year)
            fill = teal if cell in matches else pale if cell in housing_cells else "white"
            draw.rectangle((x0, y0, x0 + cell_width, y0 + cell_height), fill=fill, outline="white", width=3)
            if cell in matches:
                draw.text((x0 + cell_width / 2 - 9, y0 + cell_height / 2 - 13), "X", fill="white", font=font(24, True))
    for row_index, circuit in enumerate(circuits):
        y = top + row_index * cell_height + cell_height / 2
        label = "D.C." if circuit == "DC" else f"Cir. {circuit}"
        text_width, text_height = text_size(draw, label, tick_font)
        draw.text((left - 22 - text_width, y - text_height / 2), label, fill=navy, font=tick_font)

    legend_y = height - 122
    draw.rectangle((left, legend_y, left + 28, legend_y + 28), fill=pale, outline="white")
    draw.text((left + 42, legend_y + 2), "housing input only", fill=gray, font=tick_font)
    draw.rectangle((left + 300, legend_y, left + 328, legend_y + 28), fill=teal, outline="white")
    draw.text((left + 342, legend_y + 2), "matched legal and housing cell", fill=gray, font=tick_font)
    draw.text((left, height - 62), "The matrix documents data availability; it does not estimate a legal–segregation association.", fill=gray, font=font(19))
    HOUSING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(HOUSING_OUTPUT, "PNG", optimize=True)


if __name__ == "__main__":
    render_component_heatmap()
    render_sensitivity()
    render_housing_feasibility()
    print(HEATMAP_OUTPUT)
    print(SENSITIVITY_OUTPUT)
    print(HOUSING_OUTPUT)

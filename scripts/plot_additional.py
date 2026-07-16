#!/usr/bin/env python3
"""Render the two additional paper figures from the frozen inputs."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fha.schelling import simulate, tolerance_from_feii


FEII_INPUT = ROOT / "data" / "processed" / "feii_panel.csv"
HEATMAP_OUTPUT = ROOT / "outputs" / "feii_component_heatmap.png"
SENSITIVITY_OUTPUT = ROOT / "outputs" / "schelling_sensitivity.png"


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
    navy, rust, grid, gray = "#243B53", "#B84A32", "#D9E1E8", "#536471"
    title_font, axis_font, tick_font, cell_font = font(34, True), font(25), font(20), font(20)
    draw.text((left, 30), "Circuit-level FEII components", fill=navy, font=title_font)
    draw.text((left, 78), "Circuit means show the composite is not a single legal dimension", fill=gray, font=font(19))

    cell_width = plot_width // len(columns)
    row_height = plot_height // len(circuits)

    def color(value):
        value = max(-2.0, min(2.0, value)) / 2.0
        if value < 0:
            t = value + 1.0
            a, b = (36, 59, 83), (238, 242, 245)
        else:
            t = value
            a, b = (238, 242, 245), (184, 74, 50)
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    for j, (label, key) in enumerate(columns):
        x0 = left + j * cell_width
        draw.text((x0 + cell_width // 2 - text_size(draw, label, axis_font)[0] // 2, top - 48), label, fill=navy, font=axis_font)
        for i, circuit in enumerate(circuits):
            y0 = top + i * row_height
            value = values[circuit][key]
            draw.rectangle((x0, y0, x0 + cell_width, y0 + row_height), fill=color(value), outline="white", width=2)
            text = f"{value:+.2f}"
            tw, th = text_size(draw, text, cell_font)
            fill = "white" if abs(value) > 1.05 else navy
            draw.text((x0 + (cell_width - tw) / 2, y0 + (row_height - th) / 2 - 2), text, fill=fill, font=cell_font)

    for i, circuit in enumerate(circuits):
        y = top + i * row_height + row_height // 2
        label = f"C{circuit}"
        tw, th = text_size(draw, label, axis_font)
        draw.text((left - 18 - tw, y - th / 2), label, fill=navy, font=axis_font)

    # Diverging legend.
    legend_x = width - right + 35
    legend_y = top
    legend_h = plot_height
    for k in range(legend_h):
        value = 2.0 - 4.0 * k / max(legend_h - 1, 1)
        draw.line((legend_x, legend_y + k, legend_x + 28, legend_y + k), fill=color(value), width=1)
    draw.rectangle((legend_x, legend_y, legend_x + 28, legend_y + legend_h), outline=gray, width=1)
    for value in (2, 1, 0, -1, -2):
        y = legend_y + (2 - value) / 4 * legend_h
        draw.text((legend_x + 40, y - 12), f"{value:+d}", fill=gray, font=tick_font)
    draw_rotated_label(image, "standardized component", (width - 55, top + 210), axis_font, gray)
    HEATMAP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(HEATMAP_OUTPUT, "PNG", optimize=True)


def render_sensitivity():
    circuits, values = circuit_means()
    x = np.array([values[circuit]["FEII"] for circuit in circuits], dtype=float)
    order = np.argsort(x)
    x = x[order]
    endpoint_sets = [(0.22, 0.42), (0.28, 0.48), (0.34, 0.54)]
    colors = ["#2F8F83", "#B84A32", "#C9A227"]
    outcomes = []
    for low, high in endpoint_sets:
        tolerance = tolerance_from_feii(x, low=low, high=high)
        means = []
        for value in tolerance:
            reps = [simulate(float(value), size=20, vacancy_rate=0.10, max_steps=250, seed=20260715 + i)["segregation"] for i in range(40)]
            means.append(float(np.mean(reps)))
        outcomes.append(np.array(means))

    width, height = 1700, 1000
    left, right, top, bottom = 190, 110, 140, 155
    x0, x1 = float(x.min() - 0.05), float(x.max() + 0.05)
    y0, y1 = 0.57, 0.90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    navy, grid, gray = "#243B53", "#D9E1E8", "#536471"
    title_font, axis_font, tick_font, legend_font = font(34, True), font(25), font(20), font(20)

    def px(value):
        return int(left + (value - x0) / (x1 - x0) * (width - left - right))

    def py(value):
        return int(height - bottom - (value - y0) / (y1 - y0) * (height - top - bottom))

    draw.text((left, 30), "Schelling coupling sensitivity", fill=navy, font=title_font)
    draw.text((left, 78), "Changing tolerance endpoints preserves the illustrative FEII ordering", fill=gray, font=font(19))
    for tick in np.arange(-0.3, 0.41, 0.1):
        xpix = px(float(tick))
        draw.line((xpix, top, xpix, height - bottom), fill=grid, width=2)
        label = f"{tick:.1f}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((xpix - tw / 2, height - bottom + 18), label, fill=gray, font=tick_font)
    for tick in np.arange(0.60, 0.901, 0.05):
        ypix = py(float(tick))
        draw.line((left, ypix, width - right, ypix), fill=grid, width=2)
        label = f"{tick:.2f}"
        tw, th = text_size(draw, label, tick_font)
        draw.text((left - 18 - tw, ypix - th / 2), label, fill=gray, font=tick_font)
    draw.line((left, top, left, height - bottom), fill=navy, width=4)
    draw.line((left, height - bottom, width - right, height - bottom), fill=navy, width=4)

    for (low, high), color, means in zip(endpoint_sets, colors, outcomes):
        points = [(px(float(xv)), py(float(yv))) for xv, yv in zip(x, means)]
        for first, second in zip(points, points[1:]):
            draw.line((*first, *second), fill=color, width=5)
        for xpix, ypix in points:
            draw.ellipse((xpix - 9, ypix - 9, xpix + 9, ypix + 9), fill=color, outline="white", width=2)

    legend_x, legend_y = width - 540, top + 20
    for (low, high), color in zip(endpoint_sets, colors):
        draw.line((legend_x, legend_y + 12, legend_x + 45, legend_y + 12), fill=color, width=5)
        draw.ellipse((legend_x + 14, legend_y + 3, legend_x + 32, legend_y + 21), fill=color, outline="white", width=2)
        draw.text((legend_x + 60, legend_y), f"tolerance range [{low:.2f}, {high:.2f}]", fill=navy, font=legend_font)
        legend_y += 38
    draw.text((width - 540, legend_y + 12), "40 replications per circuit; grid and update rule fixed", fill=gray, font=font(18))
    xlabel = "circuit-mean FEII"
    ylabel = "simulated same-type neighbor share"
    tw, th = text_size(draw, xlabel, axis_font)
    draw.text(((left + width - right - tw) / 2, height - 70), xlabel, fill=navy, font=axis_font)
    draw_rotated_label(image, ylabel, (24, top + 150), axis_font, navy)
    SENSITIVITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(SENSITIVITY_OUTPUT, "PNG", optimize=True)


if __name__ == "__main__":
    render_component_heatmap()
    render_sensitivity()
    print(HEATMAP_OUTPUT)
    print(SENSITIVITY_OUTPUT)

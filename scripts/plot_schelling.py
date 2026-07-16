#!/usr/bin/env python3
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "schelling_scenarios.json"
OUTPUT = ROOT / "outputs" / "schelling_coupling.png"


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def color(value, lower, upper):
    position = (value - lower) / max(upper - lower, 1e-12)
    position = min(max(position, 0.0), 1.0)
    if position < 0.5:
        start, end, weight = (238, 243, 246), (47, 143, 131), position * 2
    else:
        start, end, weight = (47, 143, 131), (184, 74, 50), (position - 0.5) * 2
    return tuple(round(start[i] + (end[i] - start[i]) * weight) for i in range(3))


def main():
    scenarios = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = scenarios["phase_sweep"]
    tolerances = sorted({row["tolerance"] for row in rows}, reverse=True)
    access_levels = sorted({row["access"] for row in rows})
    lookup = {(row["tolerance"], row["access"]): row for row in rows}
    values = [row["segregation_mean"] for row in rows]

    width, height = 1600, 1000
    left, right, top, bottom = 315, 170, 190, 185
    plot_width, plot_height = width - left - right, height - top - bottom
    cell_width = plot_width / len(access_levels)
    cell_height = plot_height / len(tolerances)
    navy, gray = "#243B53", "#536471"
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, tick_font, cell_font = font(34, True), font(25), font(21), font(28, True)

    draw.text((left, 35), "Schelling mechanism map", fill=navy, font=title_font)
    draw.text(
        (left, 82),
        "Sorting preferences and cross-group access vary independently; FEII provides context, not a circuit mapping",
        fill=gray,
        font=font(19),
    )

    for row_index, tolerance in enumerate(tolerances):
        y0 = top + row_index * cell_height
        label = f"{tolerance:.2f}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 35 - (box[2] - box[0]), y0 + cell_height / 2 - (box[3] - box[1]) / 2), label, fill=navy, font=tick_font)
        for column_index, access in enumerate(access_levels):
            x0 = left + column_index * cell_width
            row = lookup[(tolerance, access)]
            value = row["segregation_mean"]
            fill = color(value, min(values), max(values))
            draw.rectangle((x0, y0, x0 + cell_width, y0 + cell_height), fill=fill, outline="white", width=3)
            label = f"{value:.2f}"
            box = draw.textbbox((0, 0), label, font=cell_font)
            text_fill = "white" if value > 0.76 else navy
            draw.text((x0 + (cell_width - (box[2] - box[0])) / 2, y0 + (cell_height - (box[3] - box[1])) / 2), label, fill=text_fill, font=cell_font)

    for column_index, access in enumerate(access_levels):
        x0 = left + column_index * cell_width
        label = f"{access:.2f}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x0 + cell_width / 2 - (box[2] - box[0]) / 2, top + plot_height + 22), label, fill=navy, font=tick_font)

    ylabel = "minimum same-type share required for a move"
    draw.text((left, top - 44), ylabel, fill=navy, font=axis_font)
    xlabel = "cross-group access scenario"
    box = draw.textbbox((0, 0), xlabel, font=axis_font)
    draw.text((left + (plot_width - (box[2] - box[0])) / 2, height - 90), xlabel, fill=navy, font=axis_font)
    draw.text((left, height - 48), "Cells report mean same-type neighbor share across 40 replications.", fill=gray, font=font(19))
    draw.text((left, height - 22), "Access is a stipulated gatekeeping channel, not an observed housing outcome.", fill=gray, font=font(19))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, "PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()

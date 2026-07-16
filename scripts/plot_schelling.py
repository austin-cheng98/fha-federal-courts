#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
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


def mix(color_a, color_b, weight):
    a = tuple(int(color_a[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i:i + 2], 16) for i in (1, 3, 5))
    return tuple(round(a[i] * (1 - weight) + b[i] * weight) for i in range(3))


def main():
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: float(row["FEII"]))
    feii = np.array([float(row["FEII"]) for row in rows])
    low, high = float(feii.min()), float(feii.max())

    width, height = 1600, 1000
    left, right, top, bottom = 300, 260, 145, 160
    x0, x1 = 0.58, 0.88
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    navy, rust, grid, gray = "#243B53", "#B84A32", "#D9E1E8", "#536471"
    title_font = font(34, True)
    axis_font, tick_font, label_font = font(26), font(22), font(22, True)
    note_font = font(19)

    def px(value):
        return int(left + (value - x0) / (x1 - x0) * (width - left - right))

    draw.text((left, 24), "Illustrative Schelling outcomes by circuit", fill=navy, font=title_font)
    draw.text((left, 68), "Dots are 40-replication means; whiskers are simulation intervals; rows are ordered by circuit-mean FEII", fill=gray, font=note_font)
    for tick in np.arange(0.60, 0.881, 0.05):
        xpix = px(float(tick))
        draw.line((xpix, top, xpix, height - bottom), fill=grid, width=2)
        label = f"{tick:.2f}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((xpix - (box[2] - box[0]) / 2, height - bottom + 18), label, fill=gray, font=tick_font)

    draw.line((left, top, left, height - bottom), fill=navy, width=4)
    draw.line((left, height - bottom, width - right, height - bottom), fill=navy, width=4)
    row_height = (height - top - bottom) / len(rows)
    for idx, row in enumerate(rows):
        ypix = int(top + row_height * (idx + 0.5))
        draw.line((left, ypix, width - right, ypix), fill=grid, width=1)
        feii_value = float(row["FEII"])
        segregation = float(row["segregation_mean"])
        sd = float(row["segregation_sd"])
        ci = 1.96 * sd / np.sqrt(40)
        color = mix(navy, rust, (feii_value - low) / max(high - low, 1e-12))
        draw.text((left - 80, ypix - 14), f"C{row['circuit']}", fill=navy, font=label_font)
        draw.line((left, ypix, px(segregation), ypix), fill=color, width=6)
        lo = max(x0, segregation - ci)
        hi = min(x1, segregation + ci)
        draw.line((px(lo), ypix, px(hi), ypix), fill=gray, width=3)
        draw.line((px(lo), ypix - 10, px(lo), ypix + 10), fill=gray, width=3)
        draw.line((px(hi), ypix - 10, px(hi), ypix + 10), fill=gray, width=3)
        draw.ellipse((px(segregation) - 12, ypix - 12, px(segregation) + 12, ypix + 12), fill=color, outline="white", width=3)
        draw.text((width - right + 35, ypix - 26), f"FEII {feii_value:+.3f}", fill=gray, font=note_font)
        draw.text((width - right + 35, ypix + 2), f"t(c) {float(row['tolerance']):.3f}", fill=gray, font=note_font)

    xlabel = "simulated same-type neighbor share"
    box = draw.textbbox((0, 0), xlabel, font=axis_font)
    draw.text(((left + width - right - (box[2] - box[0])) / 2, height - 72), xlabel, fill=navy, font=axis_font)
    draw.text((width - right + 35, top - 40), "coupling inputs", fill=navy, font=font(21, True))
    draw.text((left, height - 112), "lower FEII", fill=navy, font=note_font)
    draw.text((width - right - 105, height - 112), "higher FEII", fill=rust, font=note_font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, "PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()

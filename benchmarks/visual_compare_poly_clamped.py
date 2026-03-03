#!/usr/bin/env python3
"""
Visual comparison for Experiment F (Poly baseline vs Poly clamped).

Generates:
- A midpoint swatch grid
- Gradient strips across t in [0,1]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import mixbox

    HAS_MIXBOX = True
except ImportError:
    HAS_MIXBOX = False


GRAY_CHANNEL_THRESHOLD = 3
COLOR_PAIRS = [
    ("Blue + Yellow", (0, 33, 133), (252, 211, 0)),
    ("Red + Blue", (255, 39, 2), (0, 33, 133)),
    ("Red + Yellow", (255, 39, 2), (252, 211, 0)),
    ("Magenta + Yellow", (128, 2, 46), (252, 211, 0)),
    ("Blue + White", (0, 33, 133), (255, 255, 255)),
    ("Red + White", (255, 39, 2), (255, 255, 255)),
    ("Green + Magenta", (0, 60, 50), (128, 2, 46)),
]


def load_font(size: int):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_model(cache_dir: Path):
    model_path = cache_dir / "poly_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python scripts/train_poly_model.py"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def baseline_predict(model, c1, c2, t: float):
    x = np.array([[c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], t]], dtype=float)
    pred = np.clip(model.predict(x)[0], 0.0, 255.0)
    return tuple(np.clip(pred, 0, 255).astype(int))


def clamped_from_baseline(c1, c2, t: float, pred):
    c1v = np.array(c1, dtype=float)
    c2v = np.array(c2, dtype=float)
    tt = float(np.clip(t, 0.0, 1.0))
    if tt <= 0.0:
        return tuple(c1)
    if tt >= 1.0:
        return tuple(c2)

    base = (1.0 - tt) * c1v + tt * c2v
    is_gray1 = (np.max(c1v) - np.min(c1v)) <= GRAY_CHANNEL_THRESHOLD
    is_gray2 = (np.max(c2v) - np.min(c2v)) <= GRAY_CHANNEL_THRESHOLD
    if is_gray1 and is_gray2:
        return tuple(np.clip(base, 0, 255).astype(int))

    predv = np.array(pred, dtype=float)
    strength = 4.0 * tt * (1.0 - tt)
    out = base + strength * (predv - base)
    return tuple(np.clip(out, 0, 255).astype(int))


def rgb_lerp(c1, c2, t):
    return tuple(int((1.0 - t) * c1[i] + t * c2[i]) for i in range(3))


def make_swatch_grid(model, output_path: Path, swatch_size: int = 120):
    headers = ["Color 1", "Color 2", "RGB Lerp", "Poly Baseline", "Poly Clamped"]
    if HAS_MIXBOX:
        headers.append("Mixbox")

    margin = 10
    header_h = 44
    label_h = 28
    row_h = swatch_size + label_h + margin
    width = margin + len(headers) * (swatch_size + margin)
    height = header_h + margin + len(COLOR_PAIRS) * row_h + margin

    canvas = Image.new("RGB", (width, height), (241, 243, 246))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    header_font = load_font(16)
    label_font = load_font(14)

    title = "Experiment F: Poly Baseline vs Poly Clamped (t = 0.5)"
    tw = draw.textbbox((0, 0), title, font=title_font)[2]
    draw.text(((width - tw) // 2, 8), title, fill=(20, 20, 20), font=title_font)

    y_head = 40
    for i, h in enumerate(headers):
        x = margin + i * (swatch_size + margin)
        hw = draw.textbbox((0, 0), h, font=header_font)[2]
        draw.text((x + (swatch_size - hw) // 2, y_head), h, fill=(40, 40, 40), font=header_font)

    y = header_h + margin
    for name, c1, c2 in COLOR_PAIRS:
        cols = [c1, c2]
        rgb = rgb_lerp(c1, c2, 0.5)
        baseline = baseline_predict(model, c1, c2, 0.5)
        clamped = clamped_from_baseline(c1, c2, 0.5, baseline)
        cols.extend([rgb, baseline, clamped])
        if HAS_MIXBOX:
            cols.append(tuple(int(v) for v in mixbox.lerp(c1, c2, 0.5)))

        for i, c in enumerate(cols):
            x = margin + i * (swatch_size + margin)
            draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=c)

        text_y = y + swatch_size + 4
        draw.text((margin, text_y), name, fill=(20, 20, 20), font=label_font)
        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def draw_gradient_strip(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, colors):
    steps = len(colors)
    for i, c in enumerate(colors):
        x0 = x + (i * width) // steps
        x1 = x + ((i + 1) * width) // steps
        draw.rectangle([x0, y, x1, y + height], fill=c)


def make_gradient_grid(model, output_path: Path, strip_w: int = 640, strip_h: int = 26, steps: int = 25):
    methods = ["RGB Lerp", "Poly Baseline", "Poly Clamped"]
    if HAS_MIXBOX:
        methods.append("Mixbox")

    margin = 16
    row_gap = 8
    method_gap = 5
    pair_block_h = (strip_h + method_gap) * len(methods) + 26
    width = strip_w + 260
    height = 52 + len(COLOR_PAIRS) * (pair_block_h + row_gap) + margin

    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    label_font = load_font(14)
    method_font = load_font(13)

    title = "Experiment F Gradients: Baseline vs Clamped"
    tw = draw.textbbox((0, 0), title, font=title_font)[2]
    draw.text(((width - tw) // 2, 10), title, fill=(18, 18, 18), font=title_font)

    x_label = margin
    x_strip = 210
    y = 52
    t_values = np.linspace(0.0, 1.0, steps)

    for name, c1, c2 in COLOR_PAIRS:
        draw.text((x_label, y), name, fill=(20, 20, 20), font=label_font)
        y += 20

        series = {
            "RGB Lerp": [rgb_lerp(c1, c2, float(t)) for t in t_values],
            "Poly Baseline": [baseline_predict(model, c1, c2, float(t)) for t in t_values],
        }
        series["Poly Clamped"] = [
            clamped_from_baseline(c1, c2, float(t), series["Poly Baseline"][i]) for i, t in enumerate(t_values)
        ]
        if HAS_MIXBOX:
            series["Mixbox"] = [tuple(int(v) for v in mixbox.lerp(c1, c2, float(t))) for t in t_values]

        for m in methods:
            draw.text((x_label, y + 5), m, fill=(60, 60, 60), font=method_font)
            draw_gradient_strip(draw, x_strip, y, strip_w, strip_h, series[m])
            y += strip_h + method_gap

        y += row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Poly baseline vs clamped visuals")
    parser.add_argument("--cache-dir", default="models", help="Directory containing poly_model.pkl")
    parser.add_argument("--grid-out", default="benchmarks/poly_clamped_comparison.png")
    parser.add_argument("--grad-out", default="benchmarks/poly_clamped_gradients.png")
    parser.add_argument("--swatch-size", type=int, default=120)
    parser.add_argument("--gradient-steps", type=int, default=25)
    args = parser.parse_args()

    model = load_model(Path(args.cache_dir))
    grid = make_swatch_grid(model, Path(args.grid_out), swatch_size=args.swatch_size)
    grad = make_gradient_grid(model, Path(args.grad_out), steps=args.gradient_steps)

    print(f"Saved: {grid}")
    print(f"Saved: {grad}")


if __name__ == "__main__":
    main()

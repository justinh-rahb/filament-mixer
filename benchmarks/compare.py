#!/usr/bin/env python3
"""
Benchmark: FilamentMixer vs RGB Lerp vs Mixbox

Generates hard numbers and visual proof that pigment-based mixing
produces better results than naive RGB interpolation.

Optionally compares against pymixbox (pip install pymixbox) if installed.

Usage:
    pip install -e ".[viz]"
    pip install pymixbox          # optional, for head-to-head comparison
    python benchmarks/compare.py
"""

import numpy as np
import sys
import time
from pathlib import Path

from filament_mixer import FilamentMixer, CMYW_PALETTE

# Try to import PolyMixer
try:
    from filament_mixer import PolyMixer
    poly_mixer = PolyMixer.from_cache("lut_poly")
    HAS_POLY = True
except (ImportError, FileNotFoundError):
    HAS_POLY = False
    poly_mixer = None

# Try to import Mixbox for head-to-head comparison
try:
    import mixbox

    HAS_MIXBOX = True
except ImportError:
    HAS_MIXBOX = False

# Try to import matplotlib for visual output
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Delta-E (CIE76) — perceptual color distance
# ---------------------------------------------------------------------------


def srgb_to_linear(c: float) -> float:
    """Inverse sRGB companding."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def rgb_to_xyz(r: int, g: int, b: int):
    """Convert sRGB [0-255] to CIE XYZ."""
    rl = srgb_to_linear(r / 255.0)
    gl = srgb_to_linear(g / 255.0)
    bl = srgb_to_linear(b / 255.0)

    # sRGB -> XYZ (D65)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    return X, Y, Z


def xyz_to_lab(X: float, Y: float, Z: float):
    """Convert CIE XYZ to CIELAB (D65 reference white)."""
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t):
        if t > 0.008856:
            return t ** (1 / 3)
        return 7.787 * t + 16 / 116

    L = 116 * f(Y / Yn) - 16
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))
    return L, a, b


def delta_e(rgb1, rgb2):
    """CIE76 Delta-E between two sRGB colors (0-255 each)."""
    L1, a1, b1 = xyz_to_lab(*rgb_to_xyz(*rgb1))
    L2, a2, b2 = xyz_to_lab(*rgb_to_xyz(*rgb2))
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def rgb_lerp(c1, c2, t):
    """Naive RGB linear interpolation."""
    return tuple(int((1 - t) * c1[i] + t * c2[i]) for i in range(3))


def hue_of(r, g, b):
    """Return hue in degrees [0, 360)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta < 1e-8:
        return 0.0
    if cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    return h % 360


def saturation_of(r, g, b):
    """Return HSL saturation in [0, 1]."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta < 1e-8:
        return 0.0
    l = (cmax + cmin) / 2
    if l < 0.5:
        return delta / (cmax + cmin)
    return delta / (2 - cmax - cmin)


# ---------------------------------------------------------------------------
# Test cases: the same pairs Mixbox uses in their docs
# ---------------------------------------------------------------------------

MIXBOX_COLOR_PAIRS = [
    ("Blue + Yellow", (0, 33, 133), (252, 211, 0)),
    ("Red + Blue", (255, 39, 2), (0, 33, 133)),
    ("Red + Yellow", (255, 39, 2), (252, 211, 0)),
    ("Magenta + Yellow", (128, 2, 46), (252, 211, 0)),
    ("Blue + White", (0, 33, 133), (255, 255, 255)),
    ("Red + White", (255, 39, 2), (255, 255, 255)),
    ("Green + Magenta", (0, 60, 50), (128, 2, 46)),
]

# Expected perceptual midpoint hues (approximate, for sanity checks)
EXPECTED_HUES = {
    "Blue + Yellow": (60, 180),  # should be green-ish (hue 60-180)
    "Red + Yellow": (15, 60),  # should be orange-ish
    "Red + Blue": (240, 330),  # should be purple-ish
}


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def benchmark_mixing():
    """Run the head-to-head comparison."""
    mixer = FilamentMixer(CMYW_PALETTE)

    print("=" * 80)
    header = "  BENCHMARK: FilamentMixer"
    if HAS_POLY:
        header += " + PolyMixer"
    header += " vs RGB Lerp"
    if HAS_MIXBOX:
        header += " vs Mixbox"
    print(header)
    print("=" * 80)

    if not HAS_MIXBOX:
        print("\n  (Install pymixbox for head-to-head comparison: pip install pymixbox)")
    if not HAS_POLY:
        print("\n  (Train polynomial model: python scripts/train_poly_model.py)")

    results = []

    for name, c1, c2 in MIXBOX_COLOR_PAIRS:
        t = 0.5

        rgb_result = rgb_lerp(c1, c2, t)
        fm_result = mixer.lerp(*c1, *c2, t)

        poly_result = None
        if HAS_POLY:
            poly_result = poly_mixer.lerp(*c1, *c2, t)

        mixbox_result = None
        if HAS_MIXBOX:
            mixbox_result = mixbox.lerp(c1, c2, t)

        results.append(
            {
                "name": name,
                "c1": c1,
                "c2": c2,
                "rgb": rgb_result,
                "filament_mixer": fm_result,
                "poly": poly_result,
                "mixbox": mixbox_result,
            }
        )

    # Print results table
    print()
    cols = ["RGB Lerp", "FM"]
    if HAS_POLY:
        cols.append("Poly")
    if HAS_MIXBOX:
        cols.append("Mixbox")
    header_line = f"  {'Pair':<22}"
    for col in cols:
        header_line += f" {col:>14}"
    print(header_line)
    print("  " + "-" * (22 + 15 * len(cols)))

    for r in results:
        line = f"  {r['name']:<22} {str(r['rgb']):>14} {str(r['filament_mixer']):>14}"
        if HAS_POLY:
            line += f" {str(r['poly']):>14}"
        if HAS_MIXBOX:
            line += f" {str(r['mixbox']):>14}"
        print(line)

    return results


def benchmark_saturation(results):
    """Compare saturation retention — the core quality metric."""
    print("\n" + "=" * 80)
    print("  SATURATION RETENTION (higher = more vivid, less muddy)")
    print("=" * 80)

    cols = ["RGB Lerp", "FM"]
    if HAS_POLY:
        cols.append("Poly")
    if HAS_MIXBOX:
        cols.append("Mixbox")
    cols.append("FM vs RGB")
    header_line = f"\n  {'Pair':<22}"
    for col in cols:
        header_line += f" {col:>10}"
    print(header_line)
    print("  " + "-" * (22 + 11 * len(cols)))

    fm_wins = 0
    for r in results:
        sat_rgb = saturation_of(*r["rgb"])
        sat_fm = saturation_of(*r["filament_mixer"])

        improvement = ((sat_fm - sat_rgb) / max(sat_rgb, 0.001)) * 100

        line = f"  {r['name']:<22} {sat_rgb:>10.3f} {sat_fm:>10.3f}"
        if HAS_POLY and r["poly"]:
            sat_poly = saturation_of(*r["poly"])
            line += f" {sat_poly:>10.3f}"
        if HAS_MIXBOX and r["mixbox"]:
            sat_mx = saturation_of(*r["mixbox"])
            line += f" {sat_mx:>10.3f}"
        line += f"  {improvement:>+9.1f}%"
        print(line)

        if sat_fm > sat_rgb:
            fm_wins += 1

    print(f"\n  FilamentMixer more saturated than RGB lerp: {fm_wins}/{len(results)} pairs")


def benchmark_hue_accuracy(results):
    """Check that midpoint hues make physical sense."""
    print("\n" + "=" * 80)
    print("  HUE ACCURACY (does blue+yellow actually make green?)")
    print("=" * 80)

    cols = ["RGB hue", "FM hue"]
    if HAS_POLY:
        cols.append("Poly")
    if HAS_MIXBOX:
        cols.append("Mixbox")
    cols.append("Expected")
    header_line = f"\n  {'Pair':<22}"
    for col in cols:
        header_line += f" {col:>10}"
    print(header_line)
    print("  " + "-" * (22 + 11 * len(cols)))

    for r in results:
        hue_rgb = hue_of(*r["rgb"])
        hue_fm = hue_of(*r["filament_mixer"])

        expected = EXPECTED_HUES.get(r["name"], None)
        exp_str = f"{expected[0]}-{expected[1]}°" if expected else "—"

        line = f"  {r['name']:<22} {hue_rgb:>9.1f}° {hue_fm:>9.1f}°"
        if HAS_POLY and r["poly"]:
            hue_poly = hue_of(*r["poly"])
            line += f" {hue_poly:>9.1f}°"
        if HAS_MIXBOX and r["mixbox"]:
            hue_mx = hue_of(*r["mixbox"])
            line += f" {hue_mx:>9.1f}°"
        line += f"  {exp_str:>16}"
        print(line)

        # Flag the canonical test
        if r["name"] == "Blue + Yellow" and expected:
            lo, hi = expected
            rgb_in = lo <= hue_rgb <= hi
            fm_in = lo <= hue_fm <= hi
            if fm_in and not rgb_in:
                print(f"  {'':>22} ^ RGB is NOT green, FM IS green")
            elif fm_in and rgb_in:
                print(f"  {'':>22} ^ Both produce green hues")


def benchmark_delta_e_vs_mixbox(results):
    """If pymixbox is installed, measure how close we are to it."""
    if not HAS_MIXBOX:
        return

    print("\n" + "=" * 80)
    print("  DELTA-E vs MIXBOX (lower = closer to Mixbox quality)")
    print("=" * 80)

    cols = ["dE(FM)"]
    if HAS_POLY:
        cols.append("dE(Poly)")
    cols.append("dE(RGB)")
    cols.append("Winner")
    header_line = f"\n  {'Pair':<22}"
    for col in cols:
        header_line += f" {col:>10}"
    print(header_line)
    print("  " + "-" * (22 + 11 * len(cols)))

    fm_closer = 0
    poly_closer = 0
    fm_deltas = []
    poly_deltas = []
    for r in results:
        if r["mixbox"] is None:
            continue
        de_fm = delta_e(r["filament_mixer"], r["mixbox"])
        de_rgb = delta_e(r["rgb"], r["mixbox"])
        fm_deltas.append(de_fm)

        de_poly = None
        if HAS_POLY and r["poly"]:
            de_poly = delta_e(r["poly"], r["mixbox"])
            poly_deltas.append(de_poly)

        # Determine winner
        candidates = {"FM": de_fm, "RGB": de_rgb}
        if de_poly is not None:
            candidates["Poly"] = de_poly
        winner = min(candidates, key=candidates.get)

        if de_fm < de_rgb:
            fm_closer += 1
        if de_poly is not None and de_poly < de_rgb:
            poly_closer += 1

        line = f"  {r['name']:<22} {de_fm:>10.2f}"
        if de_poly is not None:
            line += f" {de_poly:>10.2f}"
        line += f" {de_rgb:>10.2f}  {winner:>10}"
        print(line)

    print()
    print(f"  FM closer to Mixbox than RGB: {fm_closer}/{len(results)} pairs")
    print(f"  Mean Delta-E (FM vs Mixbox): {np.mean(fm_deltas):.2f}")
    if poly_deltas:
        print(f"  Poly closer to Mixbox than RGB: {poly_closer}/{len(results)} pairs")
        print(f"  Mean Delta-E (Poly vs Mixbox): {np.mean(poly_deltas):.2f}")
    print(f"  (< 2.0 = imperceptible, < 5.0 = minor, < 10.0 = noticeable)")


def benchmark_roundtrip():
    """Test encode/decode roundtrip accuracy across random colors."""
    print("\n" + "=" * 74)
    print("  ROUNDTRIP ACCURACY (RGB -> latent -> RGB)")
    print("=" * 74)

    mixer = FilamentMixer(CMYW_PALETTE)
    rng = np.random.default_rng(42)

    n_samples = 50
    errors = []
    worst = (0, None, None)

    for _ in range(n_samples):
        original = tuple(rng.integers(0, 256, size=3).tolist())
        latent = mixer.rgb_to_latent(*original)
        reconstructed = mixer.latent_to_rgb(latent)
        de = delta_e(original, reconstructed)
        errors.append(de)
        if de > worst[0]:
            worst = (de, original, reconstructed)

    errors = np.array(errors)
    print(f"\n  {n_samples} random RGB colors encoded -> decoded:")
    print(f"  Mean Delta-E:   {errors.mean():.2f}")
    print(f"  Median Delta-E: {np.median(errors):.2f}")
    print(f"  Max Delta-E:    {errors.max():.2f}  (RGB{worst[1]} -> RGB{worst[2]})")
    print(f"  < 2.0 (imperceptible): {np.sum(errors < 2.0)}/{n_samples}")
    print(f"  < 5.0 (minor):         {np.sum(errors < 5.0)}/{n_samples}")


def benchmark_speed():
    """Measure timing for key operations."""
    print("\n" + "=" * 80)
    print("  SPEED")
    print("=" * 80)

    mixer = FilamentMixer(CMYW_PALETTE)
    c1, c2 = (0, 33, 133), (252, 211, 0)

    # Warm up
    mixer.lerp(*c1, *c2, 0.5)

    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        mixer.lerp(*c1, *c2, 0.5)
    t_lerp = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(n):
        mixer.get_filament_ratios(128, 200, 80)
    t_ratios = (time.perf_counter() - t0) / n

    print(f"\n  FM lerp():               {t_lerp*1000:>8.2f} ms")
    print(f"  FM get_filament_ratios(): {t_ratios*1000:>7.2f} ms")

    if HAS_POLY:
        n_poly = 1000
        poly_mixer.lerp(*c1, *c2, 0.5)  # warm up
        t0 = time.perf_counter()
        for _ in range(n_poly):
            poly_mixer.lerp(*c1, *c2, 0.5)
        t_poly = (time.perf_counter() - t0) / n_poly
        print(f"  PolyMixer.lerp():        {t_poly*1000:>8.4f} ms  ({t_lerp/t_poly:.0f}x faster than FM)")

    if HAS_MIXBOX:
        t0 = time.perf_counter()
        for _ in range(n):
            mixbox.lerp(c1, c2, 0.5)
        t_mb = (time.perf_counter() - t0) / n
        print(f"  mixbox.lerp():           {t_mb*1000:>8.4f} ms")


def generate_visual(results):
    """Generate comparison gradient image."""
    if not HAS_MPL:
        print("\n  (Install matplotlib for visual output: pip install matplotlib)")
        return

    print("\n  Generating visual comparison...")

    mixer = FilamentMixer(CMYW_PALETTE)
    steps = 100

    n_pairs = len(results)
    n_methods = 3 if HAS_MIXBOX else 2
    method_labels = ["RGB Lerp", "FilamentMixer"]
    if HAS_MIXBOX:
        method_labels.append("Mixbox")

    fig, axes = plt.subplots(
        n_pairs, n_methods, figsize=(4 * n_methods, 1.6 * n_pairs)
    )
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for row, r in enumerate(results):
        c1, c2 = r["c1"], r["c2"]

        for col, method in enumerate(method_labels):
            gradient = np.zeros((1, steps, 3))
            for i in range(steps):
                t = i / (steps - 1)
                if method == "RGB Lerp":
                    rgb = rgb_lerp(c1, c2, t)
                elif method == "FilamentMixer":
                    rgb = mixer.lerp(*c1, *c2, t)
                elif method == "Mixbox":
                    rgb = mixbox.lerp(c1, c2, t)
                else:
                    rgb = (0, 0, 0)
                gradient[0, i] = np.array(rgb) / 255.0

            ax = axes[row, col]
            ax.imshow(gradient, aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(method, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(r["name"], fontsize=9, rotation=0, labelpad=100, va="center")

    plt.suptitle(
        "Pigment Mixing Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    out_path = Path(__file__).parent / "comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    results = benchmark_mixing()
    benchmark_saturation(results)
    benchmark_hue_accuracy(results)
    benchmark_delta_e_vs_mixbox(results)
    benchmark_roundtrip()
    benchmark_speed()
    generate_visual(results)

    print("\n" + "=" * 74)
    print("  DONE")
    print("=" * 74)

    if not HAS_MIXBOX:
        print("\n  Tip: pip install pymixbox  — to add Mixbox head-to-head numbers\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark: FilamentMixer vs FastLUTMixer vs RGB Lerp vs Mixbox

Extended version that includes LUT-based mixing.

Usage:
    # First generate a LUT if you haven't:
    python scripts/generate_lut.py --resolution 64

    # Then run this benchmark:
    python benchmarks/compare_with_lut.py
    python benchmarks/compare_with_lut.py --lut-resolution 256
"""

import argparse
import numpy as np
import sys
import time
from pathlib import Path

from filament_mixer import FilamentMixer, CMYW_PALETTE

# Try to import LUT mixer
try:
    from filament_mixer import FastLUTMixer
    HAS_LUT = True
except ImportError:
    HAS_LUT = False
    print("Warning: LUT support not available. Install with: pip install -e '.[lut]'")

# Try to import PolyMixer
try:
    from filament_mixer import PolyMixer
    _poly_mixer = PolyMixer.from_cache("lut_poly")
    HAS_POLY = True
except (ImportError, FileNotFoundError):
    HAS_POLY = False
    _poly_mixer = None

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


def benchmark_mixing(lut_resolution=64):
    """Run the head-to-head comparison."""
    mixer = FilamentMixer(CMYW_PALETTE)
    
    # Try to load LUT mixer
    lut_mixer = None
    if HAS_LUT:
        try:
            lut_mixer = FastLUTMixer.from_cache("lut_cache", resolution=lut_resolution)
            print(f"  (Using {lut_resolution}³ LUT from lut_cache/)")
        except FileNotFoundError:
            print(f"  (No {lut_resolution}³ LUT found - run: python scripts/generate_lut.py --resolution {lut_resolution})")
            HAS_LUT_LOADED = False
        else:
            HAS_LUT_LOADED = True
    else:
        HAS_LUT_LOADED = False

    print("=" * 90)
    header = "  BENCHMARK: FilamentMixer"
    if HAS_LUT_LOADED:
        header += " + FastLUT"
    if HAS_POLY:
        header += " + PolyMixer"
    header += " vs RGB Lerp"
    if HAS_MIXBOX:
        header += " vs Mixbox"
    print(header)
    print("=" * 90)

    if not HAS_MIXBOX:
        print("\n  (Install pymixbox for comparison: pip install pymixbox)")
    if not HAS_POLY:
        print("\n  (Train polynomial model: python scripts/train_poly_model.py)")

    results = []

    for name, c1, c2 in MIXBOX_COLOR_PAIRS:
        t = 0.5

        rgb_result = rgb_lerp(c1, c2, t)
        fm_result = mixer.lerp(*c1, *c2, t)
        
        lut_result = None
        if HAS_LUT_LOADED:
            lut_result = lut_mixer.lerp(*c1, *c2, t)

        poly_result = None
        if HAS_POLY:
            poly_result = _poly_mixer.lerp(*c1, *c2, t)

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
                "lut": lut_result,
                "poly": poly_result,
                "mixbox": mixbox_result,
            }
        )

    # Print results table
    print()
    cols = ["RGB", "FM"]
    if HAS_LUT_LOADED:
        cols.append("LUT")
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
        if HAS_LUT_LOADED:
            line += f" {str(r['lut']):>14}"
        if HAS_POLY:
            line += f" {str(r['poly']):>14}"
        if HAS_MIXBOX:
            line += f" {str(r['mixbox']):>14}"
        print(line)

    return results, HAS_LUT_LOADED


def benchmark_delta_e(results, has_lut):
    """Compare all methods against Mixbox."""
    if not HAS_MIXBOX:
        return

    print("\n" + "=" * 90)
    print("  DELTA-E vs MIXBOX (lower = closer to Mixbox quality)")
    print("=" * 90)

    cols = ["dE(FM)"]
    if has_lut:
        cols.append("dE(LUT)")
    if HAS_POLY:
        cols.append("dE(Poly)")
    cols.append("dE(RGB)")
    cols.append("Winner")
    header_line = f"\n  {'Pair':<22}"
    for col in cols:
        header_line += f" {col:>10}"
    print(header_line)
    print("  " + "-" * (22 + 11 * len(cols)))

    fm_deltas = []
    lut_deltas = []
    poly_deltas = []
    wins = {"FM": 0, "LUT": 0, "Poly": 0, "RGB": 0}

    for r in results:
        if not r["mixbox"]:
            continue

        de_fm = delta_e(r["filament_mixer"], r["mixbox"])
        de_rgb = delta_e(r["rgb"], r["mixbox"])
        fm_deltas.append(de_fm)

        candidates = {"FM": de_fm, "RGB": de_rgb}

        de_lut = None
        if has_lut and r["lut"]:
            de_lut = delta_e(r["lut"], r["mixbox"])
            lut_deltas.append(de_lut)
            candidates["LUT"] = de_lut

        de_poly = None
        if HAS_POLY and r["poly"]:
            de_poly = delta_e(r["poly"], r["mixbox"])
            poly_deltas.append(de_poly)
            candidates["Poly"] = de_poly

        winner = min(candidates, key=candidates.get)
        wins[winner] = wins.get(winner, 0) + 1

        line = f"  {r['name']:<22} {de_fm:>10.2f}"
        if de_lut is not None:
            line += f" {de_lut:>10.2f}"
        if de_poly is not None:
            line += f" {de_poly:>10.2f}"
        line += f" {de_rgb:>10.2f}  {winner:>10}"
        print(line)

    print()
    for method, count in wins.items():
        if count > 0:
            print(f"  {method} wins: {count}/{len(results)} pairs")
    print(f"  Mean Delta-E (FM):   {np.mean(fm_deltas):.2f}")
    if lut_deltas:
        print(f"  Mean Delta-E (LUT):  {np.mean(lut_deltas):.2f}")
    if poly_deltas:
        print(f"  Mean Delta-E (Poly): {np.mean(poly_deltas):.2f}")
    print("  (< 2.0 = imperceptible, < 5.0 = minor, < 10.0 = noticeable)")


def benchmark_speed(has_lut, lut_resolution=64):
    """Measure runtime performance."""
    print("\n" + "=" * 90)
    print("  SPEED")
    print("=" * 90)
    print()

    mixer = FilamentMixer(CMYW_PALETTE)
    c1, c2 = (0, 33, 133), (252, 211, 0)

    # RGB lerp
    n = 1000
    start = time.time()
    for _ in range(n):
        _ = rgb_lerp(c1, c2, 0.5)
    rgb_time = (time.time() - start) * 1000 / n
    print(f"  RGB lerp():                {rgb_time:.4f} ms")

    # FilamentMixer
    n_fm = 100
    start = time.time()
    for _ in range(n_fm):
        _ = mixer.lerp(*c1, *c2, 0.5)
    fm_time = (time.time() - start) * 1000 / n_fm
    print(f"  FilamentMixer.lerp():      {fm_time:.2f} ms")

    # LUT mixer
    if has_lut:
        try:
            lut_mixer = FastLUTMixer.from_cache("lut_cache", resolution=lut_resolution)
            start = time.time()
            for _ in range(n):
                _ = lut_mixer.lerp(*c1, *c2, 0.5)
            lut_time = (time.time() - start) * 1000 / n
            print(f"  FastLUTMixer.lerp():       {lut_time:.4f} ms  ({fm_time/lut_time:.0f}x faster than FM)")
        except FileNotFoundError:
            pass

    # PolyMixer
    if HAS_POLY:
        _poly_mixer.lerp(*c1, *c2, 0.5)  # warm up
        start = time.time()
        for _ in range(n):
            _ = _poly_mixer.lerp(*c1, *c2, 0.5)
        poly_time = (time.time() - start) * 1000 / n
        print(f"  PolyMixer.lerp():          {poly_time:.4f} ms  ({fm_time/poly_time:.0f}x faster than FM)")

    # Mixbox
    if HAS_MIXBOX:
        start = time.time()
        for _ in range(n):
            _ = mixbox.lerp(c1, c2, 0.5)
        mx_time = (time.time() - start) * 1000 / n
        print(f"  mixbox.lerp():             {mx_time:.4f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark color mixing with LUT support")
    parser.add_argument(
        "--lut-resolution",
        type=int,
        default=64,
        choices=[64, 256],
        help="LUT resolution to use (64 or 256)"
    )
    args = parser.parse_args()
    
    results, has_lut = benchmark_mixing(lut_resolution=args.lut_resolution)
    benchmark_delta_e(results, has_lut)
    benchmark_speed(has_lut, lut_resolution=args.lut_resolution)

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

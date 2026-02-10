#!/usr/bin/env python3
"""
Benchmark: FilamentMixer vs FastLUTMixer vs RGB Lerp vs Mixbox

Extended version that includes LUT-based mixing.

Usage:
    # First generate a LUT if you haven't:
    python scripts/generate_lut.py --resolution 64

    # Then run this benchmark:
    python benchmarks/compare_with_lut.py
"""

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
    
    # Try to load LUT mixer
    lut_mixer = None
    if HAS_LUT:
        try:
            lut_mixer = FastLUTMixer.from_cache("lut_cache", resolution=64)
            print("  (Using 64³ LUT from lut_cache/)")
        except FileNotFoundError:
            print("  (No LUT found - run: python scripts/generate_lut.py --resolution 64)")
            HAS_LUT_LOADED = False
        else:
            HAS_LUT_LOADED = True
    else:
        HAS_LUT_LOADED = False

    print("=" * 80)
    header = "  BENCHMARK: FilamentMixer"
    if HAS_LUT_LOADED:
        header += " + FastLUT"
    header += " vs RGB Lerp"
    if HAS_MIXBOX:
        header += " vs Mixbox"
    print(header)
    print("=" * 80)

    if not HAS_MIXBOX:
        print("\n  (Install pymixbox for comparison: pip install pymixbox)")

    results = []

    for name, c1, c2 in MIXBOX_COLOR_PAIRS:
        t = 0.5

        rgb_result = rgb_lerp(c1, c2, t)
        fm_result = mixer.lerp(*c1, *c2, t)
        
        lut_result = None
        if HAS_LUT_LOADED:
            lut_result = lut_mixer.lerp(*c1, *c2, t)

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
                "mixbox": mixbox_result,
            }
        )

    # Print results table
    print()
    if HAS_LUT_LOADED and HAS_MIXBOX:
        print(f"  {'Pair':<22} {'RGB':>14} {'FM':>14} {'LUT':>14} {'Mixbox':>14}")
        print("  " + "-" * 78)
    elif HAS_LUT_LOADED:
        print(f"  {'Pair':<22} {'RGB Lerp':>14} {'FilamentMix':>14} {'FastLUT':>14}")
        print("  " + "-" * 64)
    elif HAS_MIXBOX:
        print(f"  {'Pair':<22} {'RGB':>14} {'FM':>14} {'Mixbox':>14}")
        print("  " + "-" * 64)
    else:
        print(f"  {'Pair':<22} {'RGB Lerp':>14} {'FilamentMix':>14}")
        print("  " + "-" * 50)

    for r in results:
        line = f"  {r['name']:<22} {str(r['rgb']):>14} {str(r['filament_mixer']):>14}"
        if HAS_LUT_LOADED:
            line += f" {str(r['lut']):>14}"
        if HAS_MIXBOX:
            line += f" {str(r['mixbox']):>14}"
        print(line)

    return results, HAS_LUT_LOADED


def benchmark_delta_e(results, has_lut):
    """Compare all methods against Mixbox."""
    if not HAS_MIXBOX:
        return

    print("\n" + "=" * 80)
    print("  DELTA-E vs MIXBOX (lower = closer to Mixbox quality)")
    print("=" * 80)

    if has_lut:
        print(f"\n  {'Pair':<22} {'dE(FM)':>10} {'dE(LUT)':>10} {'dE(RGB)':>10}  {'Winner':>10}")
        print("  " + "-" * 66)
    else:
        print(f"\n  {'Pair':<22} {'dE(FM)':>10} {'dE(RGB)':>10}  {'Winner':>10}")
        print("  " + "-" * 56)

    fm_wins = 0
    lut_wins = 0
    rgb_wins = 0
    
    fm_deltas = []
    lut_deltas = []

    for r in results:
        if not r["mixbox"]:
            continue

        de_fm = delta_e(r["filament_mixer"], r["mixbox"])
        de_rgb = delta_e(r["rgb"], r["mixbox"])
        fm_deltas.append(de_fm)
        
        if has_lut and r["lut"]:
            de_lut = delta_e(r["lut"], r["mixbox"])
            lut_deltas.append(de_lut)
            
            winner = "FM" if de_fm <= min(de_lut, de_rgb) else ("LUT" if de_lut <= de_rgb else "RGB")
            if de_fm <= min(de_lut, de_rgb):
                fm_wins += 1
            elif de_lut <= de_rgb:
                lut_wins += 1
            else:
                rgb_wins += 1
            
            line = f"  {r['name']:<22} {de_fm:>10.2f} {de_lut:>10.2f} {de_rgb:>10.2f}  {winner:>10}"
        else:
            winner = "FM" if de_fm < de_rgb else "RGB"
            if de_fm < de_rgb:
                fm_wins += 1
            else:
                rgb_wins += 1
            line = f"  {r['name']:<22} {de_fm:>10.2f} {de_rgb:>10.2f}  {winner:>10}"
        
        print(line)

    print()
    if has_lut and lut_deltas:
        print(f"  FM wins:  {fm_wins}/{len(results)} pairs")
        print(f"  LUT wins: {lut_wins}/{len(results)} pairs")
        print(f"  RGB wins: {rgb_wins}/{len(results)} pairs")
        print(f"  Mean Delta-E (FM):  {np.mean(fm_deltas):.2f}")
        print(f"  Mean Delta-E (LUT): {np.mean(lut_deltas):.2f}")
    else:
        print(f"  FM closer to Mixbox than RGB: {fm_wins}/{len(results)} pairs")
        print(f"  Mean Delta-E (FM vs Mixbox): {np.mean(fm_deltas):.2f}")
    print("  (< 2.0 = imperceptible, < 5.0 = minor, < 10.0 = noticeable)")


def benchmark_speed(has_lut):
    """Measure runtime performance."""
    print("\n" + "=" * 80)
    print("  SPEED")
    print("=" * 80)
    print()

    mixer = FilamentMixer(CMYW_PALETTE)
    c1, c2 = (0, 33, 133), (252, 211, 0)

    # RGB lerp
    n = 1000
    start = time.time()
    for _ in range(n):
        _ = rgb_lerp(c1, c2, 0.5)
    rgb_time = (time.time() - start) * 1000 / n
    print(f"  RGB lerp():                {rgb_time:.2f} ms")

    # FilamentMixer
    start = time.time()
    for _ in range(n):
        _ = mixer.lerp(*c1, *c2, 0.5)
    fm_time = (time.time() - start) * 1000 / n
    print(f"  FilamentMixer.lerp():      {fm_time:.2f} ms")

    # LUT mixer
    if has_lut:
        try:
            lut_mixer = FastLUTMixer.from_cache("lut_cache", resolution=64)
            start = time.time()
            for _ in range(n):
                _ = lut_mixer.lerp(*c1, *c2, 0.5)
            lut_time = (time.time() - start) * 1000 / n
            print(f"  FastLUTMixer.lerp():       {lut_time:.2f} ms  ({fm_time/lut_time:.0f}x faster)")
        except FileNotFoundError:
            pass

    # Mixbox
    if HAS_MIXBOX:
        start = time.time()
        for _ in range(n):
            _ = mixbox.lerp(c1, c2, 0.5)
        mx_time = (time.time() - start) * 1000 / n
        print(f"  mixbox.lerp():             {mx_time:.2f} ms")

    if not has_lut:
        print("\n  (FastLUTMixer uses precomputed LUT — instant lookups)")
    print("\n  Note: FilamentMixer and FastLUT solve optimization/lookup at runtime,")
    print("        while Mixbox uses a precomputed LUT")


def main():
    results, has_lut = benchmark_mixing()
    benchmark_delta_e(results, has_lut)
    benchmark_speed(has_lut)

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark: Polynomial Mixer (Experiment A) — Deep Analysis

A focused benchmark for the polynomial regression mixer:
- Large-scale Delta-E against Mixbox (10k random pairs)
- Boundary/edge case stress testing
- Multi-t sweep accuracy
- Speed comparison
- Green problem analysis across all t values

Usage:
    python benchmarks/compare_poly.py
    python benchmarks/compare_poly.py --samples 50000
"""

import argparse
import numpy as np
import time
from pathlib import Path

try:
    from filament_mixer import PolyMixer
    poly_mixer = PolyMixer.from_cache("lut_poly")
except (ImportError, FileNotFoundError) as e:
    print(f"Error: {e}")
    print("Train the polynomial model first: python scripts/train_poly_model.py")
    exit(1)

try:
    import mixbox
except ImportError:
    print("Error: pymixbox required. Install with: pip install pymixbox")
    exit(1)

# Optional: FastLUTMixer for comparison
try:
    from filament_mixer import FastLUTMixer
    lut_mixer = FastLUTMixer.from_cache("lut_cache", resolution=256)
    HAS_LUT = True
except (ImportError, FileNotFoundError):
    HAS_LUT = False
    lut_mixer = None


# ---------------------------------------------------------------------------
# Delta-E (CIE76)
# ---------------------------------------------------------------------------

def srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def rgb_to_xyz(r: int, g: int, b: int):
    rl = srgb_to_linear(r / 255.0)
    gl = srgb_to_linear(g / 255.0)
    bl = srgb_to_linear(b / 255.0)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    return X, Y, Z


def xyz_to_lab(X: float, Y: float, Z: float):
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
    L1, a1, b1 = xyz_to_lab(*rgb_to_xyz(*rgb1))
    L2, a2, b2 = xyz_to_lab(*rgb_to_xyz(*rgb2))
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def rgb_lerp(c1, c2, t):
    return tuple(int((1 - t) * c1[i] + t * c2[i]) for i in range(3))


# ---------------------------------------------------------------------------
# 1. Large-scale Delta-E
# ---------------------------------------------------------------------------


def benchmark_large_scale_delta_e(n_samples=10000):
    """Test accuracy against Mixbox on random color pairs."""
    print("=" * 80)
    print(f"  LARGE-SCALE DELTA-E vs MIXBOX ({n_samples:,} random pairs, t=0.5)")
    print("=" * 80)

    rng = np.random.default_rng(123)

    poly_des = []
    rgb_des = []
    lut_des = []

    for _ in range(n_samples):
        c1 = tuple(rng.integers(0, 256, 3).tolist())
        c2 = tuple(rng.integers(0, 256, 3).tolist())

        ref = mixbox.lerp(c1, c2, 0.5)
        poly_result = poly_mixer.lerp(*c1, *c2, 0.5)
        rgb_result = rgb_lerp(c1, c2, 0.5)

        poly_des.append(delta_e(poly_result, ref))
        rgb_des.append(delta_e(rgb_result, ref))

        if HAS_LUT:
            lut_result = lut_mixer.lerp(*c1, *c2, 0.5)
            lut_des.append(delta_e(lut_result, ref))

    poly_des = np.array(poly_des)
    rgb_des = np.array(rgb_des)

    def print_stats(name, des):
        print(f"\n  {name}:")
        print(f"    Mean:   {des.mean():.2f}")
        print(f"    Median: {np.median(des):.2f}")
        print(f"    Std:    {des.std():.2f}")
        print(f"    Max:    {des.max():.2f}")
        print(f"    < 2.0 (imperceptible): {np.sum(des < 2.0):,}/{len(des):,} ({np.sum(des < 2.0)/len(des)*100:.1f}%)")
        print(f"    < 5.0 (minor):         {np.sum(des < 5.0):,}/{len(des):,} ({np.sum(des < 5.0)/len(des)*100:.1f}%)")
        print(f"    < 10.0 (noticeable):   {np.sum(des < 10.0):,}/{len(des):,} ({np.sum(des < 10.0)/len(des)*100:.1f}%)")

    print_stats("PolyMixer", poly_des)
    print_stats("RGB Lerp", rgb_des)
    if HAS_LUT:
        lut_des = np.array(lut_des)
        print_stats("FastLUT (256³)", lut_des)

    print(f"\n  PolyMixer wins: {np.sum(poly_des < rgb_des):,}/{n_samples:,} vs RGB")
    if HAS_LUT:
        print(f"  PolyMixer wins: {np.sum(poly_des < lut_des):,}/{n_samples:,} vs LUT")


# ---------------------------------------------------------------------------
# 2. Boundary Stress Test
# ---------------------------------------------------------------------------


def benchmark_boundary_stress():
    """Test edge cases: extremes, pure colors, near-black, near-white."""
    print("\n" + "=" * 80)
    print("  BOUNDARY STRESS TEST")
    print("=" * 80)

    test_cases = [
        ("Black + Black", (0, 0, 0), (0, 0, 0)),
        ("White + White", (255, 255, 255), (255, 255, 255)),
        ("Black + White", (0, 0, 0), (255, 255, 255)),
        ("Red + Red", (255, 0, 0), (255, 0, 0)),
        ("Red + Green", (255, 0, 0), (0, 255, 0)),
        ("Red + Blue", (255, 0, 0), (0, 0, 255)),
        ("Green + Blue", (0, 255, 0), (0, 0, 255)),
        ("Blue + Yellow", (0, 33, 133), (252, 211, 0)),
        ("Near-Black pair", (5, 3, 8), (10, 7, 12)),
        ("Near-White pair", (250, 248, 252), (245, 250, 247)),
        ("Saturated extremes", (255, 0, 0), (0, 255, 255)),
    ]

    print(f"\n  {'Case':<24} {'Poly':>14} {'Mixbox':>14} {'dE':>8} {'RGB dE':>8}")
    print("  " + "-" * 70)

    for name, c1, c2 in test_cases:
        ref = mixbox.lerp(c1, c2, 0.5)
        poly_result = poly_mixer.lerp(*c1, *c2, 0.5)
        rgb_result = rgb_lerp(c1, c2, 0.5)

        de_poly = delta_e(poly_result, ref)
        de_rgb = delta_e(rgb_result, ref)

        marker = "✓" if de_poly < 5.0 else "⚠" if de_poly < 10.0 else "✗"
        print(f"  {name:<24} {str(poly_result):>14} {str(ref):>14} {de_poly:>7.2f} {de_rgb:>7.2f}  {marker}")


# ---------------------------------------------------------------------------
# 3. Multi-t Sweep
# ---------------------------------------------------------------------------


def benchmark_multi_t_sweep():
    """Test accuracy across the full t range, not just t=0.5."""
    print("\n" + "=" * 80)
    print("  MULTI-T SWEEP (accuracy across full mixing range)")
    print("=" * 80)

    t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    n_pairs = 2000
    rng = np.random.default_rng(456)

    # Pre-generate pairs
    pairs = [(tuple(rng.integers(0, 256, 3).tolist()), tuple(rng.integers(0, 256, 3).tolist())) for _ in range(n_pairs)]

    print(f"\n  {'t':>6} {'Mean dE':>10} {'Median dE':>10} {'Max dE':>10} {'<2.0':>8} {'<5.0':>8}")
    print("  " + "-" * 56)

    for t in t_values:
        des = []
        for c1, c2 in pairs:
            ref = mixbox.lerp(c1, c2, t)
            pred = poly_mixer.lerp(*c1, *c2, t)
            des.append(delta_e(pred, ref))
        des = np.array(des)

        pct_lt2 = f"{np.sum(des < 2.0)/len(des)*100:.0f}%"
        pct_lt5 = f"{np.sum(des < 5.0)/len(des)*100:.0f}%"
        print(f"  {t:>6.2f} {des.mean():>10.2f} {np.median(des):>10.2f} {des.max():>10.2f} {pct_lt2:>8} {pct_lt5:>8}")


# ---------------------------------------------------------------------------
# 4. Speed Benchmark
# ---------------------------------------------------------------------------


def benchmark_speed():
    """Measure per-mix timing."""
    print("\n" + "=" * 80)
    print("  SPEED BENCHMARK")
    print("=" * 80)

    c1, c2 = (0, 33, 133), (252, 211, 0)
    n = 10000

    # Warm up
    poly_mixer.lerp(*c1, *c2, 0.5)

    # PolyMixer
    t0 = time.perf_counter()
    for _ in range(n):
        poly_mixer.lerp(*c1, *c2, 0.5)
    poly_time = (time.perf_counter() - t0) / n

    # RGB lerp
    t0 = time.perf_counter()
    for _ in range(n):
        rgb_lerp(c1, c2, 0.5)
    rgb_time = (time.perf_counter() - t0) / n

    # Mixbox
    t0 = time.perf_counter()
    for _ in range(n):
        mixbox.lerp(c1, c2, 0.5)
    mx_time = (time.perf_counter() - t0) / n

    print(f"\n  {'Method':<20} {'ms/mix':>10} {'vs Poly':>10}")
    print("  " + "-" * 42)
    print(f"  {'RGB lerp':<20} {rgb_time*1000:>10.4f} {poly_time/rgb_time:>9.1f}x")
    print(f"  {'PolyMixer':<20} {poly_time*1000:>10.4f} {'1.0x':>10}")
    print(f"  {'Mixbox':<20} {mx_time*1000:>10.4f} {poly_time/mx_time:>9.1f}x")

    if HAS_LUT:
        lut_mixer.lerp(*c1, *c2, 0.5)  # warm up
        t0 = time.perf_counter()
        for _ in range(n):
            lut_mixer.lerp(*c1, *c2, 0.5)
        lut_time = (time.perf_counter() - t0) / n
        print(f"  {'FastLUT (256³)':<20} {lut_time*1000:>10.4f} {poly_time/lut_time:>9.1f}x")


# ---------------------------------------------------------------------------
# 5. The Green Problem
# ---------------------------------------------------------------------------


def benchmark_green_problem():
    """Blue + Yellow across all t values — the signature test."""
    print("\n" + "=" * 80)
    print("  THE GREEN PROBLEM (Blue + Yellow across t)")
    print("=" * 80)

    blue = (0, 33, 133)
    yellow = (252, 211, 0)

    print(f"\n  {'t':>6} {'Poly':>14} {'Mixbox':>14} {'RGB':>14} {'dE(Poly)':>10} {'dE(RGB)':>10}")
    print("  " + "-" * 70)

    for t_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ref = mixbox.lerp(blue, yellow, t_val)
        poly_result = poly_mixer.lerp(*blue, *yellow, t_val)
        rgb_result = rgb_lerp(blue, yellow, t_val)

        de_poly = delta_e(poly_result, ref)
        de_rgb = delta_e(rgb_result, ref)

        marker = "✓" if de_poly < 3.0 else "~" if de_poly < 5.0 else "✗"
        print(f"  {t_val:>6.1f} {str(poly_result):>14} {str(ref):>14} {str(rgb_result):>14} {de_poly:>9.2f} {de_rgb:>9.2f}  {marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Polynomial Mixer benchmark")
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of random pairs for large-scale test (default: 10000)",
    )
    args = parser.parse_args()

    benchmark_large_scale_delta_e(args.samples)
    benchmark_boundary_stress()
    benchmark_multi_t_sweep()
    benchmark_speed()
    benchmark_green_problem()

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

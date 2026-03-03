#!/usr/bin/env python3
"""
Experiment F: PolyMixer Endpoint + Gray Clamping

Compares two inference paths using the same trained polynomial model:
1) Baseline: direct polynomial prediction
2) Clamped:  baseline blended toward RGB lerp near t edges, with gray fallback

This does not retrain the model; it is a pure inference-time wrapper.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import mixbox

    HAS_MIXBOX = True
except ImportError:
    HAS_MIXBOX = False


GRAY_CHANNEL_THRESHOLD = 3


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

    x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    return x, y, z


def xyz_to_lab(x: float, y: float, z: float):
    """Convert XYZ to CIELAB (D65)."""
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t):
        if t > 0.008856:
            return t ** (1.0 / 3.0)
        return 7.787 * t + (16.0 / 116.0)

    l = 116.0 * f(y / yn) - 16.0
    a = 500.0 * (f(x / xn) - f(y / yn))
    b = 200.0 * (f(y / yn) - f(z / zn))
    return l, a, b


def delta_e(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """CIE76 Delta-E between two RGB triplets."""
    l1, a1, b1 = xyz_to_lab(*rgb_to_xyz(int(rgb1[0]), int(rgb1[1]), int(rgb1[2])))
    l2, a2, b2 = xyz_to_lab(*rgb_to_xyz(int(rgb2[0]), int(rgb2[1]), int(rgb2[2])))
    return float(np.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2))


def baseline_predict(model, x: np.ndarray) -> np.ndarray:
    """Original baseline polynomial inference path."""
    pred = model.predict(x)
    return np.clip(pred, 0.0, 255.0)


def clamped_predict(x: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Apply clamping wrapper:
    - exact endpoints
    - gray fallback
    - endpoint damping: base + (4*t*(1-t))*(pred-base)
    """
    t = np.clip(x[:, 6], 0.0, 1.0)
    c1 = x[:, 0:3]
    c2 = x[:, 3:6]

    base = (1.0 - t)[:, None] * c1 + t[:, None] * c2
    strength = (4.0 * t * (1.0 - t))[:, None]
    out = base + strength * (pred - base)

    c1_span = np.max(c1, axis=1) - np.min(c1, axis=1)
    c2_span = np.max(c2, axis=1) - np.min(c2, axis=1)
    gray_mask = (c1_span <= GRAY_CHANNEL_THRESHOLD) & (c2_span <= GRAY_CHANNEL_THRESHOLD)
    out = np.where(gray_mask[:, None], base, out)

    out = np.where((t <= 0.0)[:, None], c1, out)
    out = np.where((t >= 1.0)[:, None], c2, out)

    return np.clip(out, 0.0, 255.0)


def to_uint8_triplets(x: np.ndarray) -> np.ndarray:
    """Match PolyMixer behavior: clip then truncate toward zero."""
    return np.clip(x, 0.0, 255.0).astype(int)


def generate_samples(n: int, seed: int) -> np.ndarray:
    """Create random (r1,g1,b1,r2,g2,b2,t) samples."""
    rng = np.random.default_rng(seed)
    c1 = rng.integers(0, 256, size=(n, 3))
    c2 = rng.integers(0, 256, size=(n, 3))
    t = rng.random((n, 1))
    return np.hstack([c1, c2, t]).astype(float)


def summarize_drift(x: np.ndarray, baseline_rgb: np.ndarray, clamped_rgb: np.ndarray) -> Dict[str, Any]:
    """Summarize how much clamping moves output away from baseline and toward linear base."""
    t = x[:, 6]
    base = (1.0 - t)[:, None] * x[:, 0:3] + t[:, None] * x[:, 3:6]
    base_rgb = to_uint8_triplets(base)

    shift = np.mean(np.abs(clamped_rgb - baseline_rgb), axis=1)
    base_dist_baseline = np.mean(np.abs(baseline_rgb - base_rgb), axis=1)
    base_dist_clamped = np.mean(np.abs(clamped_rgb - base_rgb), axis=1)

    edge_mask = (t < 0.1) | (t > 0.9)
    mid_mask = (t >= 0.4) & (t <= 0.6)

    return {
        "mean_channel_shift_all": float(np.mean(shift)),
        "mean_channel_shift_edge": float(np.mean(shift[edge_mask])) if np.any(edge_mask) else 0.0,
        "mean_channel_shift_mid": float(np.mean(shift[mid_mask])) if np.any(mid_mask) else 0.0,
        "mean_dist_to_linear_all": {
            "baseline": float(np.mean(base_dist_baseline)),
            "clamped": float(np.mean(base_dist_clamped)),
        },
        "mean_dist_to_linear_edge": {
            "baseline": float(np.mean(base_dist_baseline[edge_mask])) if np.any(edge_mask) else 0.0,
            "clamped": float(np.mean(base_dist_clamped[edge_mask])) if np.any(edge_mask) else 0.0,
        },
    }


def summarize_mixbox_accuracy(
    x: np.ndarray, baseline_rgb: np.ndarray, clamped_rgb: np.ndarray, max_samples: int
) -> Dict[str, Any]:
    """Compute Delta-E vs Mixbox (optional dependency)."""
    if not HAS_MIXBOX:
        return {"available": False, "reason": "mixbox not installed"}

    n = min(len(x), max_samples)
    truth = np.empty((n, 3), dtype=int)
    for i in range(n):
        c1 = tuple(int(v) for v in x[i, 0:3])
        c2 = tuple(int(v) for v in x[i, 3:6])
        t = float(x[i, 6])
        truth[i] = np.array(mixbox.lerp(c1, c2, t), dtype=int)

    baseline_de = np.array([delta_e(baseline_rgb[i], truth[i]) for i in range(n)])
    clamped_de = np.array([delta_e(clamped_rgb[i], truth[i]) for i in range(n)])

    return {
        "available": True,
        "samples": n,
        "baseline_mean_de": float(np.mean(baseline_de)),
        "clamped_mean_de": float(np.mean(clamped_de)),
        "delta_mean_de": float(np.mean(clamped_de - baseline_de)),
        "baseline_median_de": float(np.median(baseline_de)),
        "clamped_median_de": float(np.median(clamped_de)),
    }


def benchmark_speed(model, x: np.ndarray, repeats: int) -> Dict[str, float]:
    """Speed benchmark for baseline and clamped inference paths."""
    # Warmup
    for _ in range(3):
        pred = baseline_predict(model, x)
        _ = clamped_predict(x, pred)

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = baseline_predict(model, x)
    t1 = time.perf_counter()

    preds = baseline_predict(model, x)
    t2 = time.perf_counter()
    for _ in range(repeats):
        _ = clamped_predict(x, preds)
    t3 = time.perf_counter()

    baseline_total_ms = ((t1 - t0) * 1000.0) / repeats
    clamp_overlay_ms = ((t3 - t2) * 1000.0) / repeats
    clamped_total_ms = baseline_total_ms + clamp_overlay_ms

    n = len(x)
    return {
        "batch_size": n,
        "baseline_ms_per_batch": baseline_total_ms,
        "clamp_overlay_ms_per_batch": clamp_overlay_ms,
        "clamped_ms_per_batch": clamped_total_ms,
        "baseline_us_per_mix": (baseline_total_ms * 1000.0) / n,
        "clamped_us_per_mix": (clamped_total_ms * 1000.0) / n,
        "overhead_percent": (clamp_overlay_ms / baseline_total_ms * 100.0) if baseline_total_ms > 0 else 0.0,
    }


def load_model(cache_dir: Path):
    model_path = cache_dir / "poly_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with:\n"
            "  python scripts/train_poly_model.py"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f), model_path


def run_experiment(args) -> Dict[str, Any]:
    model, model_path = load_model(Path(args.cache_dir))
    x = generate_samples(args.samples, args.seed)

    pred = baseline_predict(model, x)
    baseline_rgb = to_uint8_triplets(pred)
    clamped_rgb = to_uint8_triplets(clamped_predict(x, pred))

    speed = benchmark_speed(model, x, args.repeats)
    drift = summarize_drift(x, baseline_rgb, clamped_rgb)
    mixbox_stats = summarize_mixbox_accuracy(x, baseline_rgb, clamped_rgb, args.mixbox_samples)

    return {
        "experiment": "Experiment F - Poly Clamp Wrapper",
        "model_path": str(model_path),
        "samples": args.samples,
        "seed": args.seed,
        "speed": speed,
        "drift": drift,
        "mixbox_accuracy": mixbox_stats,
    }


def print_summary(results: Dict[str, Any]):
    print("=" * 78)
    print("  Experiment F: Poly Clamp Wrapper")
    print("=" * 78)
    print(f"Model: {results['model_path']}")
    print(f"Samples: {results['samples']:,}")

    speed = results["speed"]
    print("\nSpeed:")
    print(f"  Baseline: {speed['baseline_us_per_mix']:.3f} us/mix")
    print(f"  Clamped:  {speed['clamped_us_per_mix']:.3f} us/mix")
    print(f"  Overhead: {speed['overhead_percent']:.1f}% (wrapper only)")

    drift = results["drift"]
    print("\nOutput Drift vs Baseline:")
    print(f"  Mean channel shift (all):  {drift['mean_channel_shift_all']:.3f}")
    print(f"  Mean channel shift (edge): {drift['mean_channel_shift_edge']:.3f}")
    print(f"  Mean channel shift (mid):  {drift['mean_channel_shift_mid']:.3f}")

    print("\nDistance to Plain RGB Lerp (lower means closer to linear blend):")
    print(
        "  All samples:  baseline "
        f"{drift['mean_dist_to_linear_all']['baseline']:.3f} -> "
        f"clamped {drift['mean_dist_to_linear_all']['clamped']:.3f}"
    )
    print(
        "  Edge samples: baseline "
        f"{drift['mean_dist_to_linear_edge']['baseline']:.3f} -> "
        f"clamped {drift['mean_dist_to_linear_edge']['clamped']:.3f}"
    )

    mx = results["mixbox_accuracy"]
    print("\nMixbox Accuracy:")
    if not mx["available"]:
        print(f"  Skipped ({mx['reason']})")
    else:
        print(f"  Samples: {mx['samples']}")
        print(f"  Mean dE: baseline {mx['baseline_mean_de']:.3f} -> clamped {mx['clamped_mean_de']:.3f}")
        print(f"  Delta mean dE (clamped - baseline): {mx['delta_mean_de']:+.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment F: Poly clamp wrapper benchmark")
    parser.add_argument("--cache-dir", type=str, default="models", help="Directory containing poly_model.pkl")
    parser.add_argument("--samples", type=int, default=20000, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--repeats", type=int, default=20, help="Benchmark repetitions")
    parser.add_argument(
        "--mixbox-samples",
        type=int,
        default=2000,
        help="Max samples for dE vs Mixbox (if mixbox is installed)",
    )
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_experiment(args)
    print_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

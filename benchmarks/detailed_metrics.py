#!/usr/bin/env python3
"""
Detailed Benchmark: Comprehensive performance and resource metrics

Measures:
- Speed (initialization, inference, batch processing)
- Accuracy (Delta-E vs Mixbox)
- Resource usage (file sizes, memory, LOC)
- Dependencies and complexity

Usage:
    python benchmarks/detailed_metrics.py
    python benchmarks/detailed_metrics.py --output metrics.json
"""

import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from filament_mixer import FilamentMixer, CMYW_PALETTE

# Try to import production models
try:
    from filament_mixer import PolyMixer
    HAS_POLY = True
except ImportError:
    HAS_POLY = False

try:
    from filament_mixer import GPMixer
    HAS_GP = True
except ImportError:
    HAS_GP = False

# Try to import mixbox for ground truth
try:
    import mixbox
    HAS_MIXBOX = True
except ImportError:
    HAS_MIXBOX = False


# Try to import FastLUTMixer (optional LUT cache)
try:
    from filament_mixer import FastLUTMixer
    HAS_LUT = FastLUTMixer is not None
except Exception:
    FastLUTMixer = None
    HAS_LUT = False

# ===========================================================================
# Color Math Utilities
# ===========================================================================

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

    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    return X, Y, Z


def xyz_to_lab(X: float, Y: float, Z: float):
    """Convert CIE XYZ to CIELAB (D65)."""
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t):
        if t > 0.008856:
            return t ** (1 / 3)
        return 7.787 * t + 16 / 116

    L = 116 * f(Y / Yn) - 16
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))
    return L, a, b


def delta_e(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """CIE76 Delta-E between two sRGB colors."""
    L1, a1, b1 = xyz_to_lab(*rgb_to_xyz(*rgb1))
    L2, a2, b2 = xyz_to_lab(*rgb_to_xyz(*rgb2))
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def rgb_lerp(c1, c2, t):
    """Naive RGB linear interpolation."""
    return tuple(int((1 - t) * c1[i] + t * c2[i]) for i in range(3))


# ===========================================================================
# Test Data
# ===========================================================================

TEST_PAIRS = [
    ("Blue + Yellow", (0, 33, 133), (252, 211, 0)),
    ("Red + Blue", (255, 39, 2), (0, 33, 133)),
    ("Red + Yellow", (255, 39, 2), (252, 211, 0)),
    ("Magenta + Yellow", (128, 2, 46), (252, 211, 0)),
    ("Blue + White", (0, 33, 133), (255, 255, 255)),
    ("Red + White", (255, 39, 2), (255, 255, 255)),
    ("Green + Magenta", (0, 60, 50), (128, 2, 46)),
]


# ===========================================================================
# File Size & Resource Metrics
# ===========================================================================

def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    if path.exists():
        return path.stat().st_size
    return 0


def count_python_lines(path: Path) -> Dict[str, int]:
    """Count lines of code in a Python file."""
    if not path.exists():
        return {"total": 0, "code": 0, "comments": 0, "blank": 0}
    
    total = 0
    code = 0
    comments = 0
    blank = 0
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_docstring = False
        for line in f:
            total += 1
            stripped = line.strip()
            
            # Check for docstring delimiters
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                comments += 1
                continue
            
            if in_docstring:
                comments += 1
            elif not stripped:
                blank += 1
            elif stripped.startswith('#'):
                comments += 1
            else:
                code += 1
    
    return {"total": total, "code": code, "comments": comments, "blank": blank}


def analyze_source_complexity(module_name: str, source_files: List[Path]) -> Dict[str, Any]:
    """Analyze source code complexity."""
    total_lines = 0
    total_code = 0
    total_comments = 0
    total_blank = 0
    file_count = 0
    
    for file_path in source_files:
        if file_path.exists():
            stats = count_python_lines(file_path)
            total_lines += stats["total"]
            total_code += stats["code"]
            total_comments += stats["comments"]
            total_blank += stats["blank"]
            file_count += 1
    
    return {
        "module": module_name,
        "files": file_count,
        "lines": {
            "total": total_lines,
            "code": total_code,
            "comments": total_comments,
            "blank": total_blank
        }
    }


def get_model_metrics(name: str, model_path: Path) -> Dict[str, Any]:
    """Get model file metrics."""
    size = get_file_size(model_path)
    return {
        "name": name,
        "path": str(model_path),
        "size_bytes": size,
        "size_kb": round(size / 1024, 2),
        "size_mb": round(size / (1024 * 1024), 2),
        "exists": model_path.exists()
    }


# ===========================================================================
# Performance Benchmarks
# ===========================================================================

def benchmark_initialization(mixer_class, *args, **kwargs) -> Tuple[Any, float, int]:
    """Benchmark initialization time and memory."""
    tracemalloc.start()
    start_time = time.perf_counter()
    
    mixer = mixer_class(*args, **kwargs)
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    init_time = (end_time - start_time) * 1000  # ms
    return mixer, init_time, peak


def benchmark_inference(mixer, test_pairs: List, n_iterations: int = 100) -> Dict[str, Any]:
    """Benchmark inference performance."""
    times = []
    
    # Warmup
    for _ in range(10):
        for _, c1, c2 in test_pairs:
            _ = mixer.lerp(*c1, *c2, 0.5)
    
    # Actual benchmark
    for _ in range(n_iterations):
        start = time.perf_counter()
        for _, c1, c2 in test_pairs:
            _ = mixer.lerp(*c1, *c2, 0.5)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    per_call = times / len(test_pairs)
    
    return {
        "iterations": n_iterations,
        "samples_per_iteration": len(test_pairs),
        "total_ms": {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times))
        },
        "per_call_ms": {
            "mean": float(np.mean(per_call)),
            "std": float(np.std(per_call)),
            "min": float(np.min(per_call)),
            "max": float(np.max(per_call))
        }
    }


def benchmark_accuracy(mixer, test_pairs: List) -> Dict[str, Any]:
    """Benchmark accuracy against ground truth."""
    if not HAS_MIXBOX:
        return {"available": False, "reason": "mixbox not installed"}
    
    deltas = []
    results = []
    
    for name, c1, c2 in test_pairs:
        pred = mixer.lerp(*c1, *c2, 0.5)
        truth = mixbox.lerp(c1, c2, 0.5)
        
        # Convert numpy types to native Python types for JSON serialization
        pred = tuple(int(x) for x in pred)
        truth = tuple(int(x) for x in truth)
        
        de = delta_e(pred, truth)
        deltas.append(de)
        
        results.append({
            "test": name,
            "predicted": list(pred),
            "ground_truth": list(truth),
            "delta_e": float(de)
        })
    
    deltas = np.array(deltas)
    
    return {
        "available": True,
        "samples": len(test_pairs),
        "delta_e": {
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
            "median": float(np.median(deltas))
        },
        "details": results
    }


# ===========================================================================
# Main Benchmark Runner
# ===========================================================================

def benchmark_mixer(name: str, mixer_class, init_args: tuple, init_kwargs: dict,
                   source_files: List[Path], model_path: Path = None) -> Dict[str, Any]:
    """Run comprehensive benchmark for a mixer."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")
    
    # 1. Source code metrics
    print("  [1/5] Analyzing source code...")
    code_metrics = analyze_source_complexity(name, source_files)
    
    # 2. Model metrics
    model_metrics = None
    if model_path:
        print("  [2/5] Analyzing model file...")
        model_metrics = get_model_metrics(name, model_path)
    else:
        print("  [2/5] No model file (physics-based)...")
    
    # 3. Initialization
    print("  [3/5] Benchmarking initialization...")
    mixer, init_time, init_memory = benchmark_initialization(mixer_class, *init_args, **init_kwargs)
    
    # 4. Inference performance
    print("  [4/5] Benchmarking inference performance...")
    perf_metrics = benchmark_inference(mixer, TEST_PAIRS, n_iterations=100)
    
    # 5. Accuracy
    print("  [5/5] Benchmarking accuracy...")
    accuracy_metrics = benchmark_accuracy(mixer, TEST_PAIRS)
    
    return {
        "name": name,
        "source_code": code_metrics,
        "model": model_metrics,
        "initialization": {
            "time_ms": round(init_time, 4),
            "memory_peak_kb": round(init_memory / 1024, 2),
            "memory_peak_mb": round(init_memory / (1024 * 1024), 2)
        },
        "performance": perf_metrics,
        "accuracy": accuracy_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Detailed FilamentMixer benchmarks")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--iterations", "-n", type=int, default=100, 
                       help="Number of iterations for performance benchmark")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  DETAILED BENCHMARK: FilamentMixer Resource & Performance Metrics")
    print("="*70)
    
    root = Path(__file__).parent.parent
    src_dir = root / "src" / "filament_mixer"
    models_dir = root / "models"
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_pairs": len(TEST_PAIRS),
        "has_mixbox": HAS_MIXBOX,
        "mixers": []
    }
    
    # Benchmark 1: Naive RGB Lerp (baseline)
    print("\n" + "="*70)
    print("BASELINE: Naive RGB Lerp")
    print("="*70)
    rgb_results = {
        "name": "RGB Lerp",
        "source_code": {
            "module": "RGB Lerp",
            "files": 0,
            "lines": {"total": 0, "code": 0, "comments": 0, "blank": 0}
        },
        "model": None,
        "initialization": {"time_ms": 0, "memory_peak_kb": 0, "memory_peak_mb": 0},
        "performance": {
            "per_call_ms": {"mean": 0.0001, "std": 0, "min": 0, "max": 0}
        },
        "accuracy": benchmark_accuracy(
            type('RGBMixer', (), {'lerp': lambda self, *args: rgb_lerp((args[0], args[1], args[2]), 
                                                                       (args[3], args[4], args[5]), 
                                                                       args[6])})(),
            TEST_PAIRS
        )
    }
    results["mixers"].append(rgb_results)
    
    # Benchmark 2: Physics Engine (FilamentMixer)
    physics_files = [
        src_dir / "km_core.py",
        src_dir / "pigments.py",
        src_dir / "unmixer.py",
        src_dir / "api.py"
    ]
    physics_results = benchmark_mixer(
        "FilamentMixer (Physics)",
        FilamentMixer,
        (CMYW_PALETTE,),
        {},
        physics_files
    )
    results["mixers"].append(physics_results)
    
    # Benchmark: FastLUT (256^3) — placed after FilamentMixer
    if HAS_LUT:
        lut_files = [src_dir / "lut.py"]
        lut_cache = root / "lut_cache"
        unmix_cache = lut_cache / "unmix_lut_256.pkl"
        mix_cache = lut_cache / "mix_lut_256.pkl"

        if unmix_cache.exists() and mix_cache.exists():
            try:
                lut_results = benchmark_mixer(
                    "FastLUT",
                    FastLUTMixer.from_cache,
                    (str(lut_cache),),
                    {"resolution": 256},
                    lut_files,
                    unmix_cache,
                )
                results["mixers"].append(lut_results)
            except Exception as e:
                print(f"\n⚠️  Failed to initialize FastLUT: {e}")
        else:
            print("\n⚠️  FastLUT cache files not found (run: python scripts/generate_lut.py)")
    else:
        print("\n⚠️  FastLUT support not available (missing optional deps)")
    
    # Benchmark 3: PolyMixer
    if HAS_POLY:
        poly_files = [src_dir / "poly_mixer.py"]
        poly_model = models_dir / "poly_model.pkl"
        poly_results = benchmark_mixer(
            "PolyMixer",
            PolyMixer.from_cache,
            (str(models_dir),),
            {},
            poly_files,
            poly_model
        )
        results["mixers"].append(poly_results)
    else:
        print("\n⚠️  PolyMixer not available (run: python scripts/train_poly_model.py)")
    
    # Benchmark 4: GPMixer
    if HAS_GP:
        gp_files = [src_dir / "gp_mixer.py"]
        gp_model = models_dir / "gp_model.pkl"
        gp_results = benchmark_mixer(
            "GPMixer",
            GPMixer.from_cache,
            (str(models_dir),),
            {},
            gp_files,
            gp_model
        )
        results["mixers"].append(gp_results)
    else:
        print("\n⚠️  GPMixer not available (run: python scripts/train_gp_model.py)")
    
    

    # Benchmark 5: Mixbox (reference implementation)
    if HAS_MIXBOX:
        print("\n" + "="*70)
        print("Mixbox (Reference)")
        print("="*70)

        class MixboxWrapper:
            def __init__(self):
                pass

            def lerp(self, r1, g1, b1, r2, g2, b2, t):
                # mixbox.lerp returns array-like in 0-255 range
                out = mixbox.lerp((r1, g1, b1), (r2, g2, b2), t)
                return tuple(int(x) for x in out)

        mixbox_files = []
        mixbox_results = benchmark_mixer(
            "Mixbox",
            MixboxWrapper,
            (),
            {},
            mixbox_files,
            None,
        )
        results["mixers"].append(mixbox_results)
    else:
        print("\n⚠️  Mixbox not available (pip install pymixbox)")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Method':<25} {'Speed':<15} {'Accuracy':<12} {'Init':<12} {'Model':<12} {'LOC':<8}")
    print("-" * 85)
    
    for mixer in results["mixers"]:
        name = mixer["name"]
        speed = mixer["performance"].get("per_call_ms", {}).get("mean", 0)
        speed_str = f"{speed:.4f} ms" if speed > 0.0001 else "< 0.0001 ms"
        
        if mixer["accuracy"]["available"]:
            acc = mixer["accuracy"]["delta_e"]["mean"]
            acc_str = f"dE {acc:.2f}"
        else:
            acc_str = "N/A"
        
        init_time = mixer["initialization"]["time_ms"]
        if init_time > 100:
            init_str = f"{init_time/1000:.2f} s"
        elif init_time > 1:
            init_str = f"{init_time:.1f} ms"
        else:
            init_str = f"{init_time:.2f} ms"
        
        if mixer["model"]:
            size_kb = mixer['model']['size_kb']
            if size_kb > 1024:
                model_str = f"{size_kb/1024:.1f} MB"
            else:
                model_str = f"{size_kb:.1f} KB"
        else:
            model_str = "—"
        
        loc = mixer["source_code"]["lines"]["code"]
        
        print(f"{name:<25} {speed_str:<15} {acc_str:<12} {init_str:<12} {model_str:<12} {loc:<8}")
    
    # Print additional details
    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)
    
    for mixer in results["mixers"]:
        print(f"\n{mixer['name']}:")
        print(f"  Source Files: {mixer['source_code']['files']}")
        print(f"  Lines of Code: {mixer['source_code']['lines']['code']} "
              f"(+{mixer['source_code']['lines']['comments']} comments, "
              f"+{mixer['source_code']['lines']['blank']} blank)")
        
        if mixer["model"]:
            print(f"  Model Size: {mixer['model']['size_mb']:.2f} MB "
                  f"({mixer['model']['size_bytes']:,} bytes)")
        
        init = mixer["initialization"]
        print(f"  Initialization: {init['time_ms']:.2f} ms, "
              f"Peak Memory: {init['memory_peak_mb']:.2f} MB")
        
        perf = mixer["performance"]["per_call_ms"]
        print(f"  Inference: {perf['mean']:.4f} ms "
              f"(±{perf['std']:.4f} ms, "
              f"min={perf['min']:.4f}, max={perf['max']:.4f})")
        
        if mixer["accuracy"]["available"]:
            acc = mixer["accuracy"]["delta_e"]
            print(f"  Accuracy: Mean dE {acc['mean']:.2f} "
                  f"(±{acc['std']:.2f}, "
                  f"median={acc['median']:.2f}, "
                  f"min={acc['min']:.2f}, max={acc['max']:.2f})")
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

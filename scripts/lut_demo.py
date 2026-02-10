#!/usr/bin/env python3
"""
Demo: Fast LUT-based color mixing

Shows how to use precomputed LUT tables for instant color mixing
without runtime optimization.

Usage:
    # First generate the LUT (do this once):
    python scripts/generate_lut.py --resolution 64

    # Then run this demo:
    python scripts/lut_demo.py
"""

import time
import numpy as np
from filament_mixer import FilamentMixer, CMYW_PALETTE

# Try to import LUT mixer
try:
    from filament_mixer import FastLUTMixer
    has_lut = True
except ImportError:
    print("Warning: LUT support not available (missing tqdm/Pillow)")
    has_lut = False


def benchmark_comparison():
    """Compare speed of regular mixer vs LUT-based mixer."""
    
    # Test colors (from Mixbox benchmark)
    test_pairs = [
        ("Blue + Yellow", (0, 33, 133), (252, 211, 0)),
        ("Red + Blue", (255, 39, 2), (0, 33, 133)),
        ("Red + Yellow", (255, 39, 2), (252, 211, 0)),
    ]
    
    print("=" * 70)
    print("  LUT Speed Comparison")
    print("=" * 70)
    
    # Regular mixer
    print("\n1. Regular FilamentMixer (runtime optimization)")
    mixer = FilamentMixer(CMYW_PALETTE)
    
    start = time.time()
    for _ in range(100):
        for name, c1, c2 in test_pairs:
            result = mixer.lerp(*c1, *c2, 0.5)
    regular_time = (time.time() - start) * 1000 / 100 / len(test_pairs)
    
    print(f"   Average time per lerp: {regular_time:.2f} ms")
    
    # LUT mixer
    if has_lut:
        print("\n2. FastLUTMixer (precomputed lookups)")
        try:
            lut_mixer = FastLUTMixer.from_cache("lut_cache", resolution=64)
            
            start = time.time()
            for _ in range(100):
                for name, c1, c2 in test_pairs:
                    result = lut_mixer.lerp(*c1, *c2, 0.5)
            lut_time = (time.time() - start) * 1000 / 100 / len(test_pairs)
            
            print(f"   Average time per lerp: {lut_time:.2f} ms")
            print(f"\n   → SPEEDUP: {regular_time / lut_time:.1f}x faster!")
            
            # Show color comparison
            print("\n" + "=" * 70)
            print("  Color Comparison (Regular vs LUT)")
            print("=" * 70)
            for name, c1, c2 in test_pairs:
                regular = mixer.lerp(*c1, *c2, 0.5)
                lut = lut_mixer.lerp(*c1, *c2, 0.5)
                diff = np.linalg.norm(np.array(regular) - np.array(lut))
                print(f"\n  {name}")
                print(f"    Regular: {regular}")
                print(f"    LUT:     {lut}")
                print(f"    Diff:    {diff:.1f}")
        
        except FileNotFoundError as e:
            print(f"   Error: {e}")
            print("\n   Run this first:")
            print("     python scripts/generate_lut.py --resolution 64")
    else:
        print("\n2. FastLUTMixer: Not available (install with: pip install -e '.[lut]')")


def main():
    benchmark_comparison()


if __name__ == "__main__":
    main()

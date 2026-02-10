#!/usr/bin/env python3
"""
Generate LUT tables for fast color mixing.

This precomputes all RGB->concentration and concentration->RGB mappings,
allowing instant color mixing without runtime optimization.

Usage:
    python scripts/generate_lut.py
    python scripts/generate_lut.py --resolution 64  # smaller for testing
"""

import argparse
from pathlib import Path
from filament_mixer import CMYW_PALETTE
from filament_mixer.lut import LUTGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate LUT tables")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="LUT resolution (256 for full 8-bit, 64 for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lut_data",
        help="Output directory for LUT files",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="lut_cache",
        help="Directory for pickle cache files",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save LUTs as PNG images (requires Pillow)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Generating LUT Tables (Resolution: {args.resolution})")
    print("=" * 70)
    print(f"\nPigment palette: CMYW (Cyan, Magenta, Yellow, White)")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")

    generator = LUTGenerator(CMYW_PALETTE, resolution=args.resolution)

    # Generate unmix LUT (RGB -> concentrations)
    print("\n" + "=" * 70)
    print("  Step 1: Generating UNMIX LUT (RGB -> Concentrations)")
    print("=" * 70)
    unmix_cache = cache_dir / f"unmix_lut_{args.resolution}.pkl"
    generator.generate_unmix_lut(cache_file=str(unmix_cache))

    # Generate mix LUT (concentrations -> RGB)
    print("\n" + "=" * 70)
    print("  Step 2: Generating MIX LUT (Concentrations -> RGB)")
    print("=" * 70)
    mix_cache = cache_dir / f"mix_lut_{args.resolution}.pkl"
    generator.generate_mix_lut(cache_file=str(mix_cache))

    # Save as PNG if requested
    if args.save_png:
        print("\n" + "=" * 70)
        print("  Step 3: Saving LUTs as PNG")
        print("=" * 70)
        generator.save_as_png(str(output_dir))

    print("\n" + "=" * 70)
    print("  DONE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - Unmix cache: {unmix_cache}")
    print(f"  - Mix cache: {mix_cache}")
    if args.save_png:
        print(f"  - PNG images: {output_dir}/")
    
    print(f"\nLUT table info:")
    print(f"  - Resolution: {args.resolution}^3 = {args.resolution**3:,} entries")
    print(f"  - Memory: ~{(args.resolution**3 * 3 * 4 / 1024 / 1024):.1f} MB per table")
    print(f"  - Total entries: {args.resolution**3 * 2:,}")


if __name__ == "__main__":
    main()

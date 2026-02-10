"""
Lookup Table (LUT) Generator for Fast Color Mixing

Generates two 256^3 lookup tables:

1. **unmix_lut** — RGB -> ``[c1, c2, c3]`` concentrations (4th is implicit).
2. **mix_lut** — ``[c1, c2, c3]`` -> RGB.

This precomputes the expensive optimization and spectral integration so
that runtime mixing can be done via table lookups + interpolation.
"""

import numpy as np
from pathlib import Path
from typing import List
import pickle

from .km_core import Pigment, KubelkaMunk, uint8_to_rgb, rgb_to_uint8
from .unmixer import RGBUnmixer


class LUTGenerator:
    """Generates lookup tables for fast pigment mixing."""

    def __init__(self, pigments: List[Pigment], resolution: int = 256):
        """
        Args:
            pigments: List of 4 pigments (CMYW, CMYK, RYBW, etc.).
            resolution: LUT resolution (256 for 8-bit).
        """
        assert len(pigments) == 4, "Must have exactly 4 pigments"

        self.pigments = pigments
        self.resolution = resolution
        self.km = KubelkaMunk()
        self.unmixer = RGBUnmixer(pigments, self.km)

        self.unmix_lut = None  # [res, res, res, 3]
        self.mix_lut = None  # [res, res, res, 3]

    def generate_unmix_lut(self, cache_file: str | None = None) -> np.ndarray:
        """
        Generate the RGB -> concentrations LUT.

        Uses coarse-to-fine strategy: compute every 8th value, then
        trilinearly interpolate the rest.
        """
        from tqdm import tqdm

        if cache_file and Path(cache_file).exists():
            print(f"Loading unmix LUT from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                self.unmix_lut = pickle.load(f)
            return self.unmix_lut

        res = self.resolution
        print(f"Generating unmix LUT ({res}^3 = {res ** 3:,} colors)")
        print("This will take a while... (~30-60 minutes)")

        self.unmix_lut = np.zeros((res, res, res, 3), dtype=np.float32)

        # Coarse pass: every 8th value
        step = 8
        print("\nCoarse pass (1/8 resolution)...")
        coarse_set: set = set()
        for r in tqdm(range(0, res, step)):
            for g in range(0, res, step):
                for b in range(0, res, step):
                    rgb = uint8_to_rgb((r, g, b))
                    conc = self.unmixer.unmix(rgb)
                    self.unmix_lut[r, g, b, :] = conc[:3]
                    coarse_set.add((r, g, b))

        # Interpolate the rest
        print("\nInterpolating remaining values...")
        for r in tqdm(range(res)):
            for g in range(res):
                for b in range(res):
                    if (r, g, b) in coarse_set:
                        continue

                    r0 = (r // step) * step
                    g0 = (g // step) * step
                    b0 = (b // step) * step
                    r1 = min(r0 + step, res - 1)
                    g1 = min(g0 + step, res - 1)
                    b1 = min(b0 + step, res - 1)

                    rx = (r - r0) / step if step > 0 else 0
                    gx = (g - g0) / step if step > 0 else 0
                    bx = (b - b0) / step if step > 0 else 0

                    c000 = self.unmix_lut[r0, g0, b0]
                    c001 = self.unmix_lut[r0, g0, b1]
                    c010 = self.unmix_lut[r0, g1, b0]
                    c011 = self.unmix_lut[r0, g1, b1]
                    c100 = self.unmix_lut[r1, g0, b0]
                    c101 = self.unmix_lut[r1, g0, b1]
                    c110 = self.unmix_lut[r1, g1, b0]
                    c111 = self.unmix_lut[r1, g1, b1]

                    c00 = c000 * (1 - rx) + c100 * rx
                    c01 = c001 * (1 - rx) + c101 * rx
                    c10 = c010 * (1 - rx) + c110 * rx
                    c11 = c011 * (1 - rx) + c111 * rx

                    c0 = c00 * (1 - gx) + c10 * gx
                    c1 = c01 * (1 - gx) + c11 * gx

                    self.unmix_lut[r, g, b, :] = c0 * (1 - bx) + c1 * bx

        if cache_file:
            print(f"\nSaving unmix LUT to cache: {cache_file}")
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.unmix_lut, f)

        return self.unmix_lut

    def generate_mix_lut(self, cache_file: str | None = None) -> np.ndarray:
        """Generate the concentrations -> RGB LUT (much faster than unmix)."""
        from tqdm import tqdm

        if cache_file and Path(cache_file).exists():
            print(f"Loading mix LUT from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                self.mix_lut = pickle.load(f)
            return self.mix_lut

        res = self.resolution
        print(f"\nGenerating mix LUT ({res}^3 = {res ** 3:,} combinations)")

        self.mix_lut = np.zeros((res, res, res, 3), dtype=np.float32)

        for c1_idx in tqdm(range(res)):
            for c2_idx in range(res):
                for c3_idx in range(res):
                    c1 = c1_idx / 255.0
                    c2 = c2_idx / 255.0
                    c3 = c3_idx / 255.0
                    c4 = 1.0 - (c1 + c2 + c3)

                    if c4 < 0:
                        continue

                    conc = np.array([c1, c2, c3, c4])
                    rgb = self.km.mix_pigments_to_rgb(
                        self.pigments, conc, apply_gamma=True
                    )
                    self.mix_lut[c1_idx, c2_idx, c3_idx, :] = rgb

        if cache_file:
            print(f"\nSaving mix LUT to cache: {cache_file}")
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.mix_lut, f)

        return self.mix_lut

    def save_as_png(self, output_dir: str):
        """Save LUTs as tiled 4096x4096 PNG images (Mixbox-style)."""
        from PIL import Image

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.unmix_lut is not None:
            print("\nSaving unmix LUT as PNG...")
            self._save_lut_as_png(self.unmix_lut, output_path / "unmix_lut.png")

        if self.mix_lut is not None:
            print("Saving mix LUT as PNG...")
            self._save_lut_as_png(self.mix_lut, output_path / "mix_lut.png")

    def _save_lut_as_png(self, lut: np.ndarray, filepath: Path):
        """Tile a 3D LUT into a 2D image."""
        from PIL import Image

        grid_size = 16
        tile_size = self.resolution
        img_size = grid_size * tile_size

        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        for slice_idx in range(self.resolution):
            row = slice_idx // grid_size
            col = slice_idx % grid_size
            y = row * tile_size
            x = col * tile_size

            slice_data = (lut[:, :, slice_idx, :] * 255).astype(np.uint8)
            img[y : y + tile_size, x : x + tile_size, :] = slice_data

        Image.fromarray(img).save(filepath)
        print(f"  Saved: {filepath}")

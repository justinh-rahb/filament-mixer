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

        Uses coarse-to-fine strategy: compute on a sparse grid (step 8 + boundary),
        then interpolate using scipy.interpolate.RegularGridInterpolator.
        """
        from tqdm import tqdm
        from scipy.interpolate import RegularGridInterpolator

        if cache_file and Path(cache_file).exists():
            print(f"Loading unmix LUT from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                self.unmix_lut = pickle.load(f)
            return self.unmix_lut

        res = self.resolution
        print(f"Generating unmix LUT ({res}^3 = {res ** 3:,} colors)")
        print("Using vectorized generation (should take ~2-5 minutes)...")

        # Define sparse grid points (include 0, 8, 16... and ensure 255 is included)
        step = 8
        grid_points = sorted(list(set(range(0, res, step)) | {res - 1}))
        n_points = len(grid_points)
        print(f"Sparse grid size: {n_points}^3 = {n_points**3:,} samples")

        # Compute values at grid points
        coarse_data = np.zeros((n_points, n_points, n_points, 3), dtype=np.float32)
        
        # Flatten grid for batch processing? 
        # unmix() is constrained optimization, tough to batch without parallel processing.
        # We'll stick to a loop for the coarse grid, it's small enough (33^3 = 35k).
        
        print("Computing sparse samples...")
        iterations = n_points**3
        with tqdm(total=iterations) as pbar:
            for i, r in enumerate(grid_points):
                for j, g in enumerate(grid_points):
                    for k, b in enumerate(grid_points):
                        rgb = uint8_to_rgb((r, g, b))
                        conc = self.unmixer.unmix(rgb)
                        coarse_data[i, j, k, :] = conc[:3]
                        pbar.update(1)

        # Create interpolator
        print("\nInterpolating full tables...")
        interp_func = RegularGridInterpolator(
            (grid_points, grid_points, grid_points), 
            coarse_data, 
            bounds_error=False, 
            fill_value=None
        )

        # Generate target coordinates
        # We need to evaluate at every 0..255 point
        # Using mgrid to create coordinate volume
        # Note: mgrid is memory intensive for 256^3 * 3 coords (~200MB), which is fine
        grid_coords = np.mgrid[0:res, 0:res, 0:res].transpose(1, 2, 3, 0)
        
        self.unmix_lut = interp_func(grid_coords).astype(np.float32)

        if cache_file:
            print(f"\nSaving unmix LUT to cache: {cache_file}")
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.unmix_lut, f)

        return self.unmix_lut

    def generate_mix_lut(self, cache_file: str | None = None) -> np.ndarray:
        """Generate the concentrations -> RGB LUT (fully vectorized)."""
        from tqdm import tqdm

        if cache_file and Path(cache_file).exists():
            print(f"Loading mix LUT from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                self.mix_lut = pickle.load(f)
            return self.mix_lut

        res = self.resolution
        print(f"\nGenerating mix LUT ({res}^3 = {res ** 3:,} combinations)")
        print("Using vectorized generation...")

        self.mix_lut = np.zeros((res, res, res, 3), dtype=np.float32)

        # Use indices to generate concentrations
        # c1, c2, c3 range from 0 to 1
        idxs = np.arange(res, dtype=np.float32)
        concs = idxs / 255.0
        
        # Create meshgrid of concentrations
        # Memory: 256^3 * 3 floats = 192MB quite manageable
        c1, c2, c3 = np.meshgrid(concs, concs, concs, indexing='ij')
        
        # Calculate c4
        c4 = 1.0 - (c1 + c2 + c3)
        
        # Mask valid combinations
        mask = c4 >= -1e-5  # slightly loose tolerance
        valid_indices = np.where(mask)
        
        # Extract valid concentrations (N, 4)
        N = len(valid_indices[0])
        print(f"Computing {N:,} valid mixtures...")
        
        # Valid concentrations array
        # Shape (N, 4)
        c1_valid = c1[mask]
        c2_valid = c2[mask]
        c3_valid = c3[mask]
        c4_valid = c4[mask]
        
        # Normalize to ensure sum=1 exactly (fix tolerance issues)
        s = c1_valid + c2_valid + c3_valid + c4_valid
        c1_valid /= s
        c2_valid /= s
        c3_valid /= s
        c4_valid /= s
        
        concentrations = np.stack([c1_valid, c2_valid, c3_valid, c4_valid], axis=1)

        # Vectorized pigment mixing
        # We need to compute: R = KM(K, S) -> XYZ -> RGB
        
        # 1. Mix K and S
        # K_mix: (N, 38)
        K_mix = np.zeros((N, 38), dtype=np.float32)
        S_mix = np.zeros((N, 38), dtype=np.float32)
        
        # Manual mix loop (only 4 pigments, so loop is fine)
        for i, pigment in enumerate(self.pigments):
            # c is (N, 1)
            c = concentrations[:, i : i+1]
            # pigment.K is (1, 38) (broadcasted)
            K_mix += c * pigment.K[np.newaxis, :]
            S_mix += c * pigment.S[np.newaxis, :]
            
        # 2. Compute Reflectance (Equation 2)
        # Element-wise operations support (N, 38)
        S_safe = np.where(S_mix == 0, 1e-10, S_mix)
        a = K_mix / S_safe
        b = np.sqrt(a * a + 2 * a)
        R = 1 + a - b
        R = np.clip(R, 0, 1)
        
        # 3. Saunderson Correction (Equation 6)
        k1 = self.km.k1
        k2 = self.km.k2
        numerator = (1 - k1) * (1 - k2) * R
        denominator = 1 - k2 * R
        denominator = np.where(denominator == 0, 1e-10, denominator)
        R_prime = numerator / denominator
        R_prime = np.clip(R_prime, 0, 1)
        
        # 4. XYZ Integration
        # CIE data: (38,)
        # weighted_R: (N, 38)
        from filament_mixer.km_core import (
            CIE_X_BAR, CIE_Y_BAR, CIE_Z_BAR, 
            D65_ILLUMINANT, CIE_WAVELENGTHS
        )
        
        weighted_R = R_prime * D65_ILLUMINANT[np.newaxis, :]
        
        # Integrate along axis 1 (wavelengths)
        try:
            from numpy import trapezoid as trapz
        except ImportError:
            from numpy import trapz
            
        X = trapz(CIE_X_BAR[np.newaxis, :] * weighted_R, CIE_WAVELENGTHS, axis=1)
        Y = trapz(CIE_Y_BAR[np.newaxis, :] * weighted_R, CIE_WAVELENGTHS, axis=1)
        Z = trapz(CIE_Z_BAR[np.newaxis, :] * weighted_R, CIE_WAVELENGTHS, axis=1)
        
        X /= self.km.Y_D65
        Y /= self.km.Y_D65
        Z /= self.km.Y_D65
        
        # 5. RGB conversion
        # XYZ to Linear RGB
        M = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570],
        ])
        
        # Stack XYZ: (3, N)
        XYZ = np.stack([X, Y, Z], axis=0)
        rgb_linear = M @ XYZ  # (3, 3) @ (3, N) -> (3, N)
        
        # Gamma correction
        # Vectorized gamma function
        # gamma(x) = 12.92x if x <= 0.0031308 else 1.055 * x^(1/2.4) - 0.055
        rgb_linear = rgb_linear.T # (N, 3)
        
        mask_low = rgb_linear <= 0.0031308
        mask_high = ~mask_low
        
        rgb_final = np.zeros_like(rgb_linear)
        rgb_final[mask_low] = 12.92 * rgb_linear[mask_low]
        # Avoid negative bases in power
        # Although linear RGB can be negative (out of gamut), we usually clip first?
        # Mixbox implementation clips RGB to [0,1] at end.
        # But for gamma power, negative values are NaNs.
        # We should clip before gamma? Or handle abs?
        # Standard: clip to 0 before gamma.
        rgb_safe = np.clip(rgb_linear, 0, None)
        rgb_final[mask_high] = 1.055 * (rgb_safe[mask_high] ** (1 / 2.4)) - 0.055
        
        rgb_final = np.clip(rgb_final, 0, 1)
        
        # Fill result array
        self.mix_lut[mask] = rgb_final

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


class FastLUTMixer:
    """
    Fast color mixer using precomputed LUT tables.
    
    Provides the same API as FilamentMixer but uses table lookups
    instead of runtime optimization for ~1000x speedup.
    """

    def __init__(self, unmix_lut: np.ndarray, mix_lut: np.ndarray):
        """
        Args:
            unmix_lut: Precomputed RGB->concentration table [256,256,256,3]
            mix_lut: Precomputed concentration->RGB table [256,256,256,3]
        """
        self.unmix_lut = unmix_lut
        self.mix_lut = mix_lut
        self.resolution = unmix_lut.shape[0]

    @classmethod
    def from_cache(cls, cache_dir: str, resolution: int = 256):
        """Load LUT tables from cache files."""
        import pickle
        
        cache_path = Path(cache_dir)
        unmix_file = cache_path / f"unmix_lut_{resolution}.pkl"
        mix_file = cache_path / f"mix_lut_{resolution}.pkl"
        
        if not unmix_file.exists() or not mix_file.exists():
            raise FileNotFoundError(
                f"LUT cache files not found. Generate them first with:\n"
                f"  python scripts/generate_lut.py --resolution {resolution}"
            )
        
        with open(unmix_file, "rb") as f:
            unmix_lut = pickle.load(f)
        with open(mix_file, "rb") as f:
            mix_lut = pickle.load(f)
        
        return cls(unmix_lut, mix_lut)

    def rgb_to_latent(self, r: int, g: int, b: int) -> np.ndarray:
        """Convert RGB to 7D latent (fast LUT lookup)."""
        # Scale RGB from 0-255 to LUT resolution
        scale = (self.resolution - 1) / 255.0
        r_idx = int(r * scale)
        g_idx = int(g * scale)
        b_idx = int(b * scale)
        
        # Simple nearest-neighbor lookup
        conc = self.unmix_lut[r_idx, g_idx, b_idx, :]
        
        # Reconstruct RGB to compute residual
        c4 = 1.0 - np.sum(conc)
        full_conc = np.array([conc[0], conc[1], conc[2], c4])
        
        # Get mixed RGB from concentration
        c1_idx = int(conc[0] * (self.resolution - 1))
        c2_idx = int(conc[1] * (self.resolution - 1))
        c3_idx = int(conc[2] * (self.resolution - 1))
        
        if c4 >= 0:
            mixed_rgb = self.mix_lut[c1_idx, c2_idx, c3_idx, :]
        else:
            mixed_rgb = np.array([r/255.0, g/255.0, b/255.0])
        
        residual = np.array([r/255.0, g/255.0, b/255.0]) - mixed_rgb
        
        latent = np.zeros(7)
        latent[0:3] = conc
        latent[3] = c4
        latent[4:7] = residual
        return latent

    def latent_to_rgb(self, latent: np.ndarray) -> tuple[int, int, int]:
        """Convert 7D latent to RGB (fast LUT lookup)."""
        conc = latent[0:3]
        residual = latent[4:7]
        
        c1_idx = int(np.clip(conc[0] * (self.resolution - 1), 0, self.resolution - 1))
        c2_idx = int(np.clip(conc[1] * (self.resolution - 1), 0, self.resolution - 1))
        c3_idx = int(np.clip(conc[2] * (self.resolution - 1), 0, self.resolution - 1))
        
        mixed_rgb = self.mix_lut[c1_idx, c2_idx, c3_idx, :]
        final_rgb = mixed_rgb + residual
        final_rgb = np.clip(final_rgb, 0, 1)
        
        return (
            int(final_rgb[0] * 255),
            int(final_rgb[1] * 255),
            int(final_rgb[2] * 255),
        )

    def lerp(
        self,
        r1: int, g1: int, b1: int,
        r2: int, g2: int, b2: int,
        t: float,
    ) -> tuple[int, int, int]:
        """Mix two colors (fast LUT-based)."""
        latent1 = self.rgb_to_latent(r1, g1, b1)
        latent2 = self.rgb_to_latent(r2, g2, b2)
        latent_mix = (1 - t) * latent1 + t * latent2
        return self.latent_to_rgb(latent_mix)

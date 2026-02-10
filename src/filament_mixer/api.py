"""
FilamentMixer - Drop-in Mixbox-compatible API for 3D Printer Filaments

Provides the same API as Mixbox but uses Kubelka-Munk theory for
physically-accurate filament color mixing.

API::

    rgb_to_latent(r, g, b) -> [c1, c2, c3, c4, rR, rG, rB]
    latent_to_rgb(latent)  -> (r, g, b)
    lerp(r1, g1, b1, r2, g2, b2, t) -> (r, g, b)
"""

import numpy as np
from typing import Tuple, List

from .km_core import Pigment, KubelkaMunk, uint8_to_rgb, rgb_to_uint8
from .unmixer import RGBUnmixer


LATENT_SIZE = 7  # 4 concentrations + 3 residuals


class FilamentMixer:
    """
    Mixbox-compatible API for filament color mixing.

    Uses Kubelka-Munk theory to mix filament colors like real pigments.
    """

    def __init__(self, pigments: List[Pigment]):
        """
        Args:
            pigments: List of 4 pigments (e.g. CMYW, CMYK, RYBW).
        """
        assert len(pigments) == 4, "Must provide exactly 4 pigments"

        self.pigments = pigments
        self.km = KubelkaMunk()
        self.unmixer = RGBUnmixer(pigments, self.km)

    def rgb_to_latent(self, r: int, g: int, b: int) -> np.ndarray:
        """
        Encode RGB (0-255) to 7-D latent space.

        Returns:
            ``[c1, c2, c3, c4, rR, rG, rB]`` — 7-D vector.
        """
        rgb_float = uint8_to_rgb((r, g, b))
        concentrations, residual = self.unmixer.unmix_with_residual(rgb_float)

        latent = np.zeros(LATENT_SIZE)
        latent[0:4] = concentrations
        latent[4:7] = residual
        return latent

    def latent_to_rgb(self, latent: np.ndarray) -> Tuple[int, int, int]:
        """
        Decode 7-D latent space to RGB (0-255).
        """
        concentrations = latent[0:4]
        residual = latent[4:7]

        mixed_rgb = self.km.mix_pigments_to_rgb(
            self.pigments, concentrations, apply_gamma=True
        )

        final_rgb = np.array(mixed_rgb) + np.array(residual)
        final_rgb = np.clip(final_rgb, 0, 1)
        return rgb_to_uint8(tuple(final_rgb))

    def lerp(
        self,
        r1: int, g1: int, b1: int,
        r2: int, g2: int, b2: int,
        t: float,
    ) -> Tuple[int, int, int]:
        """
        Mix two RGB colors with pigment-based blending.

        Args:
            r1, g1, b1: First RGB color [0, 255].
            r2, g2, b2: Second RGB color [0, 255].
            t: Mixing ratio [0, 1] (0 = all color1, 1 = all color2).
        """
        latent1 = self.rgb_to_latent(r1, g1, b1)
        latent2 = self.rgb_to_latent(r2, g2, b2)
        latent_mix = (1 - t) * latent1 + t * latent2
        return self.latent_to_rgb(latent_mix)

    def mix_n_colors(
        self,
        colors: List[Tuple[int, int, int]],
        weights: List[float],
    ) -> Tuple[int, int, int]:
        """
        Mix N colors with arbitrary weights.

        Weights are normalized to sum to 1 internally.
        """
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        latents = [self.rgb_to_latent(*rgb) for rgb in colors]

        latent_mix = np.zeros(LATENT_SIZE)
        for latent, weight in zip(latents, weights):
            latent_mix += weight * latent

        return self.latent_to_rgb(latent_mix)

    def get_filament_ratios(self, r: int, g: int, b: int) -> np.ndarray:
        """
        Get filament mixing ratios for a target RGB color.

        **This is the key function for 3D printing** — the returned
        concentrations map directly to M163 G-code commands.

        Returns:
            ``[c1, c2, c3, c4]`` — filament percentages (sum to 1).
        """
        latent = self.rgb_to_latent(r, g, b)
        return latent[0:4]

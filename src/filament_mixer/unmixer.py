"""
RGB Unmixing - Converting RGB colors back to pigment concentrations.

Implements the inverse of the ``mix()`` function using constrained
optimization (Equation 9 from the Mixbox paper).
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple

from .km_core import Pigment, KubelkaMunk


class RGBUnmixer:
    """
    Unmixes RGB colors into pigment concentrations.

    Solves the optimization problem from Equation 9::

        unmix(RGB) = argmin_c ||mix(c) - RGB||^2
        subject to: c_i >= 0 and sum(c_i) = 1
    """

    def __init__(self, pigments: List[Pigment], km: KubelkaMunk | None = None):
        """
        Args:
            pigments: List of N pigments (primary palette).
            km: Kubelka-Munk solver (creates one if not provided).
        """
        self.pigments = pigments
        self.n_pigments = len(pigments)
        self.km = km if km is not None else KubelkaMunk()

    def _rgb_distance(
        self, concentrations: np.ndarray, target_rgb: Tuple[float, float, float]
    ) -> float:
        """Squared L2 distance between mixed color and target."""
        mixed_rgb = self.km.mix_pigments_to_rgb(
            self.pigments, concentrations, apply_gamma=True
        )
        diff = np.array(mixed_rgb) - np.array(target_rgb)
        return np.sum(diff**2)

    def unmix(
        self,
        rgb: Tuple[float, float, float],
        initial_guess: np.ndarray | None = None,
        method: str = "SLSQP",
    ) -> np.ndarray:
        """
        Unmix an RGB color into pigment concentrations.

        Args:
            rgb: Target RGB color in [0, 1] range.
            initial_guess: Starting point for optimization (uniform if *None*).
            method: Optimization method (``'SLSQP'`` or ``'L-BFGS-B'``).

        Returns:
            Array of N concentrations that sum to 1.
        """
        if initial_guess is None:
            initial_guess = np.ones(self.n_pigments) / self.n_pigments

        constraints = [{"type": "eq", "fun": lambda c: np.sum(c) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(self.n_pigments)]

        result = minimize(
            fun=lambda c: self._rgb_distance(c, rgb),
            x0=initial_guess,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-4},
        )

        if not result.success:
            print(f"Warning: Unmixing failed for RGB{rgb}: {result.message}")

        # Normalize to ensure concentrations sum to exactly 1.0
        x = result.x
        x = np.clip(x, 0.0, 1.0)
        x = x / np.sum(x)
        return x

    def unmix_with_residual(
        self, rgb: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Unmix RGB to concentrations + residual (full latent representation).

        Returns:
            concentrations: Pigment concentrations.
            residual: RGB residual (additive correction).
        """
        concentrations = self.unmix(rgb)
        mixed_rgb = self.km.mix_pigments_to_rgb(
            self.pigments, concentrations, apply_gamma=True
        )
        residual = tuple(np.array(rgb) - np.array(mixed_rgb))
        return concentrations, residual

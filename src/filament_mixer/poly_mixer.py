"""
PolyMixer — Polynomial Regression Color Mixer (Experiment A)

Uses a pre-trained degree-3 polynomial regression to map
(R1, G1, B1, R2, G2, B2, t) → (R_mix, G_mix, B_mix).

Trained on Mixbox ground truth, achieving Mean Delta-E ~3.3
with ~0.002ms per mix — essentially free computation.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Tuple

GRAY_CHANNEL_THRESHOLD = 3


class PolyMixer:
    """
    Fast polynomial-based color mixer.

    Loads a pre-trained sklearn pipeline (PolynomialFeatures + LinearRegression)
    and provides the same ``lerp()`` API as FilamentMixer and FastLUTMixer.
    """

    def __init__(self, model):
        """
        Args:
            model: A trained sklearn pipeline (PolynomialFeatures + LinearRegression).
        """
        self.model = model

    @classmethod
    def from_cache(cls, cache_dir: str = "models") -> "PolyMixer":
        """Load a pre-trained polynomial model from cache.

        Args:
            cache_dir: Directory containing ``poly_model.pkl``.
        """
        cache_path = Path(cache_dir) / "poly_model.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Polynomial model not found at {cache_path}. "
                f"Train it first with:\n  python scripts/train_poly_model.py"
            )

        with open(cache_path, "rb") as f:
            model = pickle.load(f)

        return cls(model)

    def lerp(
        self,
        r1: int, g1: int, b1: int,
        r2: int, g2: int, b2: int,
        t: float,
    ) -> Tuple[int, int, int]:
        """
        Mix two RGB colors using polynomial regression.

        Args:
            r1, g1, b1: First RGB color [0, 255].
            r2, g2, b2: Second RGB color [0, 255].
            t: Mixing ratio [0, 1] (0 = all color1, 1 = all color2).

        Returns:
            Mixed RGB color as (r, g, b) tuple.
        """
        t = float(np.clip(t, 0.0, 1.0))
        if t <= 0.0:
            return (r1, g1, b1)
        if t >= 1.0:
            return (r2, g2, b2)

        c1 = np.array([r1, g1, b1], dtype=float)
        c2 = np.array([r2, g2, b2], dtype=float)
        base = (1.0 - t) * c1 + t * c2

        # If both endpoints are near-gray, plain RGB lerp is more stable.
        is_gray1 = (np.max(c1) - np.min(c1)) <= GRAY_CHANNEL_THRESHOLD
        is_gray2 = (np.max(c2) - np.min(c2)) <= GRAY_CHANNEL_THRESHOLD
        if is_gray1 and is_gray2:
            out = np.clip(base, 0, 255).astype(int)
            return (int(out[0]), int(out[1]), int(out[2]))

        X = np.array([[r1, g1, b1, r2, g2, b2, t]])
        pred = np.clip(self.model.predict(X)[0], 0.0, 255.0)

        # Damp polynomial influence near endpoints; keep full strength at t=0.5.
        clamp_strength = 4.0 * t * (1.0 - t)
        out = base + clamp_strength * (pred - base)
        out = np.clip(out, 0, 255).astype(int)
        return (int(out[0]), int(out[1]), int(out[2]))

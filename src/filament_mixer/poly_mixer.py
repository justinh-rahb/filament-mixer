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
        if t <= 0:
            return (r1, g1, b1)
        if t >= 1:
            return (r2, g2, b2)

        X = np.array([[r1, g1, b1, r2, g2, b2, t]])
        pred = self.model.predict(X)[0]
        pred = np.clip(pred, 0, 255).astype(int)
        return (int(pred[0]), int(pred[1]), int(pred[2]))

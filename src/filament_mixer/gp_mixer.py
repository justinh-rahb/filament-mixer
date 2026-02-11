"""
Gaussian Process-based color mixer (Experiment C).

A trained GP model that directly learns the RGB₁ + RGB₂ → RGB_mix mapping
from Mixbox ground truth data. Provides high accuracy without requiring
LUT tables or unmixing.

Performance:
    - Mean Delta-E: ~2.3
    - Speed: ~0.025ms per mix
    - Direct 6D → 3D regression (no concentration space)
"""

import numpy as np
import pickle
import os
from typing import Tuple
from pathlib import Path


class GPMixer:
    """
    Color mixer using a trained Gaussian Process Regressor.
    Drop-in replacement for FilamentMixer with superior accuracy.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the GPMixer with a trained model.
        
        Args:
            model_path: Path to the pickled GP model file.
                       If None, looks for models/gp_model.pkl
        """
        if model_path is None:
            # Default to models directory
            base_path = Path(__file__).parent.parent.parent
            model_path = base_path / "models" / "gp_model.pkl"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"GP model not found at {model_path}\\n"
                f"Please run: python scripts/train_gp_model.py"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.model_path = model_path
        print(f"✓ Loaded GP model from {model_path.name}")
    
    def lerp(
        self, 
        r1: int, g1: int, b1: int, 
        r2: int, g2: int, b2: int, 
        t: float
    ) -> Tuple[int, int, int]:
        """
        Mix two RGB colors using the GP model.
        
        Args:
            r1, g1, b1: First color (RGB 0-255)
            r2, g2, b2: Second color (RGB 0-255)
            t: Mixing ratio (0.0 = all color1, 1.0 = all color2)
        
        Returns:
            Tuple of (r, g, b) for the mixed color (0-255)
        """
        # Handle boundary cases exactly
        if t <= 0.0:
            return (r1, g1, b1)
        if t >= 1.0:
            return (r2, g2, b2)
        
        # Prepare input: [r1, g1, b1, r2, g2, b2, t]
        # Scale RGB to [0, 1], t is already in [0, 1]
        X = np.array([[r1/255., g1/255., b1/255., r2/255., g2/255., b2/255., t]])
        
        # Predict mix (returns scaled [0, 1])
        y_pred = self.model.predict(X)[0]
        
        # Scale back to [0, 255] and clip
        rgb_out = np.clip(y_pred * 255.0, 0, 255).astype(int)
        
        return tuple(rgb_out)
    
    @classmethod
    def from_cache(cls, cache_dir: str = "models"):
        """
        Load GP model from a cache directory.
        
        Args:
            cache_dir: Directory containing gp_model.pkl
        
        Returns:
            GPMixer instance
        """
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            # Relative to project root
            base_path = Path(__file__).parent.parent.parent
            cache_path = base_path / cache_dir
        
        model_file = cache_path / "gp_model.pkl"
        return cls(model_path=str(model_file))
    
    def __repr__(self):
        return f"GPMixer(model={self.model_path.name})"

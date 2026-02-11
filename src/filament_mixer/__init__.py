"""
FilamentMixer — physics-based color mixing for 3D printer filaments.

Uses Kubelka-Munk theory so that mixing blue + yellow yields green
instead of the muddy gray you get with naive RGB interpolation.

Quick start::

    from filament_mixer import FilamentMixer, CMYW_PALETTE

    mixer = FilamentMixer(CMYW_PALETTE)
    green = mixer.lerp(0, 33, 133,  252, 211, 0,  0.5)
"""

from .km_core import (
    CIE_WAVELENGTHS,
    CIE_X_BAR,
    CIE_Y_BAR,
    CIE_Z_BAR,
    D65_ILLUMINANT,
    KubelkaMunk,
    Pigment,
    rgb_to_uint8,
    uint8_to_rgb,
)
from .pigments import (
    CMYK_PALETTE,
    CMYW_PALETTE,
    RYBW_PALETTE,
    create_black_filament,
    create_blue_filament,
    create_cyan_filament,
    create_magenta_filament,
    create_red_filament,
    create_white_filament,
    create_yellow_filament,
    gaussian_peak,
)
from .unmixer import RGBUnmixer
from .api import FilamentMixer, LATENT_SIZE

# Optional LUT support (requires tqdm/Pillow)
try:
    from .lut import LUTGenerator, FastLUTMixer
    _HAS_LUT = True
except ImportError:
    _HAS_LUT = False
    LUTGenerator = None
    FastLUTMixer = None

# Optional Polynomial Mixer support (requires sklearn)
try:
    from .poly_mixer import PolyMixer
    _HAS_POLY = True
except ImportError:
    _HAS_POLY = False
    PolyMixer = None

# Optional GP mixer (requires scikit-learn)
try:
    from .gp_mixer import GPMixer
    _HAS_GP = True
except ImportError:
    _HAS_GP = False
    GPMixer = None

__all__ = [
    # Core
    "KubelkaMunk",
    "Pigment",
    "CIE_WAVELENGTHS",
    "CIE_X_BAR",
    "CIE_Y_BAR",
    "CIE_Z_BAR",
    "D65_ILLUMINANT",
    "rgb_to_uint8",
    "uint8_to_rgb",
    # Pigments / palettes
    "gaussian_peak",
    "create_cyan_filament",
    "create_magenta_filament",
    "create_yellow_filament",
    "create_black_filament",
    "create_white_filament",
    "create_red_filament",
    "create_blue_filament",
    "CMYK_PALETTE",
    "CMYW_PALETTE",
    "RYBW_PALETTE",
    # Unmixer
    "RGBUnmixer",
    # API
    "FilamentMixer",
    "LATENT_SIZE",
    # LUT (optional)
    "LUTGenerator",
    "FastLUTMixer",
    # Polynomial Mixer (optional)
    "PolyMixer",
    # GP Mixer (optional)
    "GPMixer",
]

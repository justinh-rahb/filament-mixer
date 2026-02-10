"""
Pigment Library for 3D Printer Filaments

Generates realistic K/S spectral curves for common filament colors.
Based on paint pigment data but adapted for plastic filaments.
"""

import numpy as np

from .km_core import Pigment, CIE_WAVELENGTHS


def gaussian_peak(wavelengths, center, width, height):
    """Create a Gaussian-shaped spectral peak."""
    return height * np.exp(-((wavelengths - center) / width) ** 2)


def create_cyan_filament() -> Pigment:
    """Cyan filament (approximates Phthalo Blue behavior).

    Tight red/orange absorption; transmits blue (400-490nm) and green (490-560nm).
    Previous broad peaks bled absorption into the green band, darkening
    Blue+Yellow mixes.
    """
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.02
    # Red absorption — narrow to keep green (530nm) clean
    K += gaussian_peak(CIE_WAVELENGTHS, 640, 60, 2.5)
    # Orange absorption — shifted to 600nm and tight so it doesn't eat green
    K += gaussian_peak(CIE_WAVELENGTHS, 600, 30, 1.5)
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Cyan", K, S)


def create_magenta_filament() -> Pigment:
    """Magenta filament (approximates Quinacridone Magenta).

    Absorbs green strongly, passes red and blue.
    Asymmetric dual-peak profile: steep cutoff on the blue side (preserves
    blue in tinted mixes like Red+White) while extending slightly toward
    yellow-green for fuller green absorption.
    """
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.03
    # Lower-green peak — narrow to avoid blue leakage
    K += gaussian_peak(CIE_WAVELENGTHS, 530, 25, 3.0)
    # Upper-green peak — extends absorption toward yellow-green
    K += gaussian_peak(CIE_WAVELENGTHS, 555, 30, 2.5)
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Magenta", K, S)


def create_yellow_filament() -> Pigment:
    """Yellow filament (approximates Hansa Yellow).

    Moderate blue-violet absorption edge, very high reflectance in yellow-red.
    Reduced K height avoids crushing blue channel in tinted mixes.
    """
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.02
    # Blue absorption — moderate height to preserve blue in tints
    K += gaussian_peak(CIE_WAVELENGTHS, 450, 45, 2.5)
    # Violet tail
    K += gaussian_peak(CIE_WAVELENGTHS, 400, 40, 2.0)
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Yellow", K, S)


def create_black_filament() -> Pigment:
    """Black filament - absorbs everything uniformly."""
    K = np.ones(len(CIE_WAVELENGTHS)) * 3.0
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.3
    return Pigment("Black", K, S)


def create_white_filament() -> Pigment:
    """White filament (Titanium White equivalent).

    Low absorption, moderate scattering.  Previous S=2.5 overwhelmed
    other pigments in X+White mixes, desaturating them to flat pastels.
    """
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.008
    S = np.ones(len(CIE_WAVELENGTHS)) * 1.2
    return Pigment("White", K, S)


def create_red_filament() -> Pigment:
    """Red filament - absorbs blue and green, reflects red."""
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.1
    K += gaussian_peak(CIE_WAVELENGTHS, 450, 60, 2.0)
    K += gaussian_peak(CIE_WAVELENGTHS, 520, 50, 1.5)
    K += gaussian_peak(CIE_WAVELENGTHS, 490, 40, 1.0)
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.6
    return Pigment("Red", K, S)


def create_blue_filament() -> Pigment:
    """Blue filament - absorbs red and yellow, reflects blue."""
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.1
    K += gaussian_peak(CIE_WAVELENGTHS, 650, 70, 2.0)
    K += gaussian_peak(CIE_WAVELENGTHS, 580, 50, 1.5)
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.6
    return Pigment("Blue", K, S)


# ---------------------------------------------------------------------------
# Predefined palettes
# ---------------------------------------------------------------------------

CMYK_PALETTE = [
    create_cyan_filament(),
    create_magenta_filament(),
    create_yellow_filament(),
    create_black_filament(),
]

CMYW_PALETTE = [
    create_cyan_filament(),
    create_magenta_filament(),
    create_yellow_filament(),
    create_white_filament(),
]

RYBW_PALETTE = [
    create_red_filament(),
    create_yellow_filament(),
    create_blue_filament(),
    create_white_filament(),
]

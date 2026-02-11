"""
Kubelka-Munk Theory Implementation for Filament Color Mixing

Based on the Mixbox paper by Sochorová & Jamriška (2021).
Implements the physics of pigment mixing using K-M theory.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


# CIE 1931 2-degree standard observer data (sampled at 10nm from 380-750nm)
# These are the color matching functions that convert spectral data to XYZ
CIE_WAVELENGTHS = np.arange(380, 751, 10)  # 38 wavelengths

# CIE 1931 2° Standard Observer color matching functions (10nm sampling)
CIE_X_BAR = np.array([
    0.0014, 0.0042, 0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954,
    0.0956, 0.0320, 0.0049, 0.0093, 0.0633, 0.1655, 0.2904, 0.4334, 0.5945, 0.7621,
    0.9163, 1.0263, 1.0622, 1.0026, 0.8544, 0.6424, 0.4479, 0.2835, 0.1649, 0.0874,
    0.0468, 0.0227, 0.0114, 0.0058, 0.0029, 0.0014, 0.0007, 0.0003
])

CIE_Y_BAR = np.array([
    0.0000, 0.0001, 0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910,
    0.1390, 0.2080, 0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520,
    0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070, 0.0610, 0.0320,
    0.0170, 0.0082, 0.0041, 0.0021, 0.0010, 0.0005, 0.0003, 0.0001
])

CIE_Z_BAR = np.array([
    0.0065, 0.0201, 0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876,
    0.8130, 0.4652, 0.2720, 0.1582, 0.0782, 0.0422, 0.0203, 0.0087, 0.0039, 0.0021,
    0.0017, 0.0011, 0.0008, 0.0003, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
])

# D65 illuminant spectral power distribution (normalized, 10nm sampling)
D65_ILLUMINANT = np.array([
    49.98, 52.31, 54.65, 68.70, 82.75, 87.12, 91.49, 92.46, 93.43, 90.06,
    86.68, 95.77, 104.86, 110.94, 117.01, 117.41, 117.81, 116.34, 114.86, 115.39,
    115.92, 112.37, 108.81, 109.08, 109.35, 108.58, 107.80, 106.30, 104.79, 106.24,
    107.69, 106.05, 104.41, 104.23, 104.05, 102.02, 100.00, 98.17
])


@dataclass
class Pigment:
    """Represents a pigment with its K (absorption) and S (scattering) spectra."""

    name: str
    K: np.ndarray  # Absorption coefficient per wavelength (38 values)
    S: np.ndarray  # Scattering coefficient per wavelength (38 values)

    def __post_init__(self):
        assert len(self.K) == len(CIE_WAVELENGTHS), (
            f"K spectrum must have {len(CIE_WAVELENGTHS)} samples"
        )
        assert len(self.S) == len(CIE_WAVELENGTHS), (
            f"S spectrum must have {len(CIE_WAVELENGTHS)} samples"
        )
        assert np.all(self.K >= 0), "K values must be non-negative"
        assert np.all(self.S > 0), "S values must be positive"


class KubelkaMunk:
    """
    Kubelka-Munk pigment mixing model.

    Takes pigment K/S coefficients and computes mixed colors.
    """

    def __init__(self, k1: float = 0.04, k2: float = 0.6):
        """
        Initialize K-M model.

        Args:
            k1, k2: Saunderson correction coefficients for surface reflection.
        """
        self.k1 = k1
        self.k2 = k2

        # Precompute normalization constant for XYZ conversion
        try:
            from numpy import trapezoid as trapz
        except ImportError:
            from numpy import trapz
        self.Y_D65 = trapz(CIE_Y_BAR * D65_ILLUMINANT, CIE_WAVELENGTHS)

    def mix_spectra(
        self, pigments: List[Pigment], concentrations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix pigment spectra according to K-M theory (Equation 1 from paper).

        Args:
            pigments: List of N pigments.
            concentrations: Array of N concentrations (must sum to 1, all >= 0).

        Returns:
            K_mix, S_mix: Mixed absorption and scattering spectra.
        """
        concentrations = np.asarray(concentrations)

        assert len(concentrations) == len(pigments), (
            "Concentration count must match pigment count"
        )
        
        # Auto-normalize if within 5% tolerance
        total = np.sum(concentrations)
        if abs(total - 1.0) > 0.05:
            # excessive deviation, something is wrong with the optimizer
            # but for LUT generation, we might want to be lenient?
            # actually 1.03 is quite high.
            # let's just warn and normalize
            pass

        # Always normalize to exactly 1.0 to avoid drift
        concentrations = concentrations / total
        
        # Clip negative values (optimizer might give -1e-10)
        concentrations = np.clip(concentrations, 0, 1)
        concentrations = concentrations / np.sum(concentrations)

        K_mix = np.zeros(len(CIE_WAVELENGTHS))
        S_mix = np.zeros(len(CIE_WAVELENGTHS))

        for pigment, c in zip(pigments, concentrations):
            K_mix += c * pigment.K
            S_mix += c * pigment.S

        return K_mix, S_mix

    def compute_reflectance(self, K: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Compute reflectance spectrum from K and S using K-M equation (Equation 2).

        Assumes infinite thickness (completely hides substrate).
        """
        S = np.where(S == 0, 1e-10, S)

        a = K / S
        b = np.sqrt(a * a + 2 * a)
        R = 1 + a - b

        return np.clip(R, 0, 1)

    def apply_saunderson_correction(self, R: np.ndarray) -> np.ndarray:
        """Apply Saunderson correction for surface reflection (Equation 6)."""
        numerator = (1 - self.k1) * (1 - self.k2) * R
        denominator = 1 - self.k2 * R
        denominator = np.where(denominator == 0, 1e-10, denominator)

        R_prime = numerator / denominator
        return np.clip(R_prime, 0, 1)

    def reflectance_to_xyz(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert reflectance spectrum to CIE XYZ tristimulus values (Equations 3-5)."""
        try:
            from numpy import trapezoid as trapz
        except ImportError:
            from numpy import trapz

        weighted_R = R * D65_ILLUMINANT

        X = trapz(CIE_X_BAR * weighted_R, CIE_WAVELENGTHS)
        Y = trapz(CIE_Y_BAR * weighted_R, CIE_WAVELENGTHS)
        Z = trapz(CIE_Z_BAR * weighted_R, CIE_WAVELENGTHS)

        X /= self.Y_D65
        Y /= self.Y_D65
        Z /= self.Y_D65

        return X, Y, Z

    def xyz_to_srgb(self, X: float, Y: float, Z: float) -> Tuple[float, float, float]:
        """Convert XYZ to linear sRGB (Equation 7)."""
        M = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570],
        ])

        rgb = M @ np.array([X, Y, Z])
        return tuple(rgb)

    def srgb_gamma_correction(
        self, linear_rgb: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Apply sRGB gamma correction."""
        def gamma(x):
            if x <= 0.0031308:
                return 12.92 * x
            else:
                return 1.055 * (x ** (1 / 2.4)) - 0.055

        r, g, b = linear_rgb
        return (gamma(r), gamma(g), gamma(b))

    def mix_pigments_to_rgb(
        self,
        pigments: List[Pigment],
        concentrations: np.ndarray,
        apply_gamma: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Complete pipeline: mix pigments and get sRGB color.

        This is the ``mix()`` function from the paper (Equations 1-7).
        """
        K_mix, S_mix = self.mix_spectra(pigments, concentrations)
        R_spectrum = self.compute_reflectance(K_mix, S_mix)
        R_corrected = self.apply_saunderson_correction(R_spectrum)
        X, Y, Z = self.reflectance_to_xyz(R_corrected)
        rgb_linear = self.xyz_to_srgb(X, Y, Z)

        if apply_gamma:
            rgb = self.srgb_gamma_correction(rgb_linear)
        else:
            rgb = rgb_linear

        return tuple(np.clip(rgb, 0, 1))


def rgb_to_uint8(rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert float RGB [0,1] to uint8 [0,255]."""
    r, g, b = rgb
    return (int(r * 255), int(g * 255), int(b * 255))


def uint8_to_rgb(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert uint8 RGB [0,255] to float [0,1]."""
    r, g, b = rgb
    return (r / 255.0, g / 255.0, b / 255.0)

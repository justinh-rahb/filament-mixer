"""Basic tests for filament_mixer."""

import numpy as np
import pytest


def test_import():
    """Package imports successfully."""
    from filament_mixer import FilamentMixer, CMYW_PALETTE, KubelkaMunk, Pigment

    assert FilamentMixer is not None
    assert len(CMYW_PALETTE) == 4


def test_pure_pigment_colors():
    """Each pure pigment produces a valid RGB color."""
    from filament_mixer import KubelkaMunk, CMYW_PALETTE, rgb_to_uint8

    km = KubelkaMunk()
    for i, pigment in enumerate(CMYW_PALETTE):
        conc = np.zeros(4)
        conc[i] = 1.0
        rgb = km.mix_pigments_to_rgb(CMYW_PALETTE, conc)
        rgb8 = rgb_to_uint8(rgb)

        # All channels should be in valid range
        assert all(0 <= c <= 255 for c in rgb8), f"{pigment.name} produced invalid RGB: {rgb8}"


def test_concentrations_sum_to_one():
    """Unmixing always produces concentrations that sum to 1."""
    from filament_mixer import FilamentMixer, CMYW_PALETTE

    mixer = FilamentMixer(CMYW_PALETTE)
    ratios = mixer.get_filament_ratios(128, 200, 80)

    assert ratios.shape == (4,)
    assert np.allclose(np.sum(ratios), 1.0, atol=1e-4)
    assert np.all(ratios >= 0)


def test_lerp_endpoints():
    """lerp(t=0) returns color1, lerp(t=1) returns color2."""
    from filament_mixer import FilamentMixer, CMYW_PALETTE

    mixer = FilamentMixer(CMYW_PALETTE)
    c1 = (0, 33, 133)
    c2 = (252, 211, 0)

    result_0 = mixer.lerp(*c1, *c2, 0.0)
    result_1 = mixer.lerp(*c1, *c2, 1.0)

    # Should be close to the input colors (within rounding)
    assert all(abs(a - b) <= 2 for a, b in zip(result_0, c1)), (
        f"lerp(t=0) should return color1: got {result_0}, expected ~{c1}"
    )
    assert all(abs(a - b) <= 2 for a, b in zip(result_1, c2)), (
        f"lerp(t=1) should return color2: got {result_1}, expected ~{c2}"
    )


def test_blue_yellow_makes_green():
    """The signature test: blue + yellow should produce green, not gray."""
    from filament_mixer import FilamentMixer, CMYW_PALETTE

    mixer = FilamentMixer(CMYW_PALETTE)
    result = mixer.lerp(0, 33, 133, 252, 211, 0, 0.5)

    r, g, b = result
    # Green channel should be dominant or at least comparable to red
    # and the result should NOT be a desaturated gray
    assert g > b, f"Green channel should exceed blue in the mix, got RGB{result}"


def test_roundtrip_latent():
    """Encoding to latent and decoding back should approximate the original."""
    from filament_mixer import FilamentMixer, CMYW_PALETTE

    mixer = FilamentMixer(CMYW_PALETTE)
    original = (200, 100, 50)

    latent = mixer.rgb_to_latent(*original)
    reconstructed = mixer.latent_to_rgb(latent)

    assert all(abs(a - b) <= 5 for a, b in zip(original, reconstructed)), (
        f"Roundtrip failed: {original} -> {reconstructed}"
    )


def test_mix_n_colors():
    """Mixing N colors with equal weights should work."""
    from filament_mixer import FilamentMixer, CMYW_PALETTE

    mixer = FilamentMixer(CMYW_PALETTE)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    result = mixer.mix_n_colors(colors, [1.0, 1.0, 1.0])

    assert all(0 <= c <= 255 for c in result), f"Invalid RGB: {result}"


def test_palettes_exist():
    """All built-in palettes should have 4 pigments each."""
    from filament_mixer import CMYW_PALETTE, CMYK_PALETTE, RYBW_PALETTE

    for palette in [CMYW_PALETTE, CMYK_PALETTE, RYBW_PALETTE]:
        assert len(palette) == 4
        for pigment in palette:
            assert len(pigment.K) == 38
            assert len(pigment.S) == 38


def test_pigment_validation():
    """Pigment should reject invalid K/S spectra."""
    from filament_mixer import Pigment

    with pytest.raises(AssertionError):
        Pigment("Bad", np.ones(10), np.ones(38))  # Wrong K length

    with pytest.raises(AssertionError):
        Pigment("Bad", np.ones(38) * -1, np.ones(38))  # Negative K

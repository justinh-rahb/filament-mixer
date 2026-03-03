"""Tests for the direct cubic PolyMixer implementation."""

from filament_mixer.poly_mixer import PolyMixer


def test_poly_lerp_endpoints_are_exact():
    mixer = PolyMixer()
    c1 = (12, 34, 56)
    c2 = (210, 180, 150)

    assert mixer.lerp(*c1, *c2, 0.0) == c1
    assert mixer.lerp(*c1, *c2, 1.0) == c2


def test_poly_lerp_clamps_t_outside_range():
    mixer = PolyMixer()
    c1 = (10, 20, 30)
    c2 = (200, 210, 220)

    assert mixer.lerp(*c1, *c2, -1.0) == c1
    assert mixer.lerp(*c1, *c2, 2.0) == c2


def test_poly_lerp_gray_inputs_use_linear_blend():
    mixer = PolyMixer()

    # Both endpoints are near-gray (max-min <= 3), so polynomial path is skipped.
    out = mixer.lerp(100, 101, 99, 200, 202, 201, 0.5)
    assert out == (150, 152, 150)


def test_poly_lerp_blue_yellow_remains_greenish():
    mixer = PolyMixer()
    out = mixer.lerp(0, 33, 133, 252, 211, 0, 0.5)
    r, g, b = out
    assert g > b, f"expected green-ish midpoint, got {out}"


def test_poly_from_cache_is_compatibility_constructor():
    mixer = PolyMixer.from_cache("does-not-matter")
    out = mixer.lerp(0, 0, 0, 255, 255, 255, 0.5)
    assert out == (128, 128, 128)

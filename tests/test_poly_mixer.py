"""Tests for PolyMixer clamping behavior."""

import numpy as np

from filament_mixer.poly_mixer import PolyMixer


class DummyModel:
    """Minimal stand-in model returning a fixed prediction."""

    def __init__(self, pred):
        self.pred = np.asarray(pred, dtype=float).reshape(1, 3)

    def predict(self, X):
        _ = X
        return self.pred


def test_poly_lerp_endpoints_are_exact():
    mixer = PolyMixer(DummyModel([0, 255, 0]))
    c1 = (12, 34, 56)
    c2 = (210, 180, 150)

    assert mixer.lerp(*c1, *c2, 0.0) == c1
    assert mixer.lerp(*c1, *c2, 1.0) == c2


def test_poly_lerp_gray_inputs_use_linear_blend():
    mixer = PolyMixer(DummyModel([255, 0, 255]))

    # Both endpoints are near-gray (max-min <= 3), so model output is ignored.
    out = mixer.lerp(100, 101, 99, 200, 202, 201, 0.5)
    assert out == (150, 151, 150)


def test_poly_lerp_edge_damping_biases_toward_linear_near_endpoints():
    mixer = PolyMixer(DummyModel([255, 255, 255]))

    # t=0.1 => clamp strength = 4*t*(1-t) = 0.36
    # base for red channel is 25.5, blended = 25.5 + 0.36*(255 - 25.5) = 108.12 -> int 108
    out = mixer.lerp(0, 10, 0, 255, 100, 30, 0.1)
    assert out == (108, 103, 93)


def test_poly_lerp_midpoint_keeps_full_model_strength():
    mixer = PolyMixer(DummyModel([10, 20, 30]))
    out = mixer.lerp(0, 10, 0, 255, 100, 30, 0.5)
    assert out == (10, 20, 30)

"""Tests for the no-reference fusion metrics.

Unit tests are ported from paper-exp tests/test_metrics.py (the EN/SD/SF/AG
sections, paper-exp 42024cd 2026-08-29); the equivalence tests compare the
port against the original implementation when paper-exp is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantiloom_fusion.evaluation.no_reference import (
    DISPLAY_SCALE,
    average_gradient,
    entropy,
    spatial_frequency,
    standard_deviation,
)

from paper_exp_loader import PAPER_EXP, paper_exp_modules


@pytest.fixture
def rng():
    return np.random.default_rng(20240814)


@pytest.fixture
def noise(rng):
    return rng.random((64, 64)).astype(np.float32)


def _ramp(h=64, w=64, axis=1):
    line = np.linspace(0.0, 1.0, w if axis == 1 else h, dtype=np.float64)
    return np.tile(line, (h, 1)) if axis == 1 else np.tile(line[:, None], (1, w))


def _centred_mask(h=64, w=64, margin=16):
    mask = np.zeros((h, w), np.float32)
    mask[margin:h - margin, margin:w - margin] = 1.0
    return mask


# --- Ported from paper-exp tests/test_metrics.py --------------------------


def test_entropy_of_uniform_256_level_ramp_is_exactly_8_bits():
    ramp = (np.arange(256, dtype=np.float32) / 255.0)[None, :].repeat(256, 0)
    assert entropy(ramp) == pytest.approx(8.0, abs=1e-9)


def test_entropy_of_two_equally_likely_values_is_one_bit():
    img = np.zeros((64, 64), np.float32)
    img[:, :32] = 1.0
    assert entropy(img) == pytest.approx(1.0, abs=1e-9)


def test_entropy_of_constant_image_is_zero_not_negative_zero():
    result = entropy(np.full((16, 16), 0.5, np.float32))
    assert result == 0.0
    assert not np.signbit(result)


def test_entropy_clips_out_of_range_input_instead_of_wrapping(noise):
    assert entropy(noise * 3.0) < entropy(noise)
    assert entropy(noise - 1.0) == pytest.approx(0.0, abs=1e-9)


def test_standard_deviation_is_on_the_0_255_scale():
    img = np.zeros((64, 64), np.float32)
    img[:, :32] = 1.0
    assert standard_deviation(img) == pytest.approx(DISPLAY_SCALE / 2, abs=1e-6)


def test_spatial_frequency_matches_its_closed_form_on_a_ramp():
    ramp = _ramp()
    step = DISPLAY_SCALE / 63
    rf_sq = (64 * 63 * step ** 2) / (64 * 64)
    assert spatial_frequency(ramp) == pytest.approx(np.sqrt(rf_sq), rel=1e-9)


def test_average_gradient_matches_forward_difference_definition():
    ramp = _ramp()
    step = DISPLAY_SCALE / 63
    assert average_gradient(ramp) == pytest.approx(np.sqrt(step ** 2 / 2), rel=1e-9)


def test_spatial_frequency_normalizer_is_area_not_difference_count(rng):
    img = rng.random((4, 2)).astype(np.float32)
    display = np.clip(img.astype(np.float64), 0, 1) * DISPLAY_SCALE
    area = float(img.size)
    expected = np.sqrt(
        np.sum(np.diff(display, axis=1) ** 2) / area
        + np.sum(np.diff(display, axis=0) ** 2) / area
    )
    assert spatial_frequency(img) == pytest.approx(expected, rel=1e-9)


def test_spatial_frequency_distinguishes_row_and_column_structure():
    horizontal = _ramp(8, 32, axis=1)
    vertical = _ramp(32, 8, axis=0)
    assert spatial_frequency(horizontal) == pytest.approx(
        spatial_frequency(vertical.T if vertical.shape != horizontal.shape
                          else vertical), rel=1e-9)


def test_gradient_metrics_are_zero_on_a_flat_image():
    flat = np.full((32, 32), 0.3, np.float32)
    assert spatial_frequency(flat) == pytest.approx(0.0, abs=1e-12)
    assert average_gradient(flat) == pytest.approx(0.0, abs=1e-12)


def test_masked_metrics_see_only_the_masked_region(rng):
    mask = _centred_mask()
    img = np.full((64, 64), 0.25, np.float32)
    img[mask > 0.5] = rng.random(int(mask.sum())).astype(np.float32)

    assert standard_deviation(img, mask=1 - mask) == pytest.approx(0.0, abs=1e-9)
    assert entropy(img, mask=1 - mask) == pytest.approx(0.0, abs=1e-9)
    assert standard_deviation(img, mask=mask) > 10.0
    assert standard_deviation(img) > 1.0


def test_masked_gradient_ignores_the_region_boundary():
    mask = _centred_mask()
    img = np.zeros((64, 64), np.float32)
    img[mask < 0.5] = 1.0
    assert average_gradient(img, mask=mask) == pytest.approx(0.0, abs=1e-9)
    assert spatial_frequency(img, mask=mask) == pytest.approx(0.0, abs=1e-9)


def test_empty_mask_is_undefined_not_zero(noise):
    empty = np.zeros_like(noise)
    for fn in (entropy, standard_deviation, spatial_frequency, average_gradient):
        assert np.isnan(fn(noise, mask=empty))


def test_mask_shape_mismatch_is_rejected(noise):
    with pytest.raises(ValueError):
        entropy(noise, mask=np.ones((8, 8), np.float32))


# --- Equivalence against the original implementation ----------------------

paper_exp = pytest.mark.skipif(
    not PAPER_EXP.exists(), reason="paper-exp checkout not present"
)
_METRICS = (entropy, standard_deviation, spatial_frequency, average_gradient)


def _orig_values(img, mask=None):
    with paper_exp_modules():
        from paper_exp.metrics import no_reference as orig

        return {fn.__name__: getattr(orig, fn.__name__)(img, mask=mask)
                for fn in _METRICS}


@paper_exp
def test_equivalence_unmasked(rng):
    img = rng.random((96, 80)).astype(np.float32)
    expected = _orig_values(img)
    for fn in _METRICS:
        assert fn(img) == pytest.approx(expected[fn.__name__], abs=1e-12), fn.__name__


@paper_exp
def test_equivalence_masked(rng):
    img = rng.random((96, 80)).astype(np.float32)
    mask = (rng.random((96, 80)) > 0.3).astype(np.float32)
    expected = _orig_values(img, mask)
    for fn in _METRICS:
        assert fn(img, mask=mask) == pytest.approx(
            expected[fn.__name__], abs=1e-12), fn.__name__


@paper_exp
def test_equivalence_out_of_range_and_empty(rng):
    img = rng.random((32, 32)).astype(np.float32) * 3.0 - 1.0
    empty = np.zeros_like(img)
    expected = _orig_values(img, empty)
    for fn in _METRICS:
        got, exp = fn(img, mask=empty), expected[fn.__name__]
        assert (np.isnan(got) and np.isnan(exp)) or got == pytest.approx(
            exp, abs=1e-12), fn.__name__

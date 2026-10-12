"""Tests for the full-reference metrics and the Metric wrappers.

The PSNR/SSIM unit tests are ported from paper-exp tests/test_metrics.py
(paper-exp 42024cd 2026-08-29).  The wrapper tests exercise the
:class:`Metric` subclasses on synthetic FusionInputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.evaluation import (
    AverageGradient,
    Entropy,
    Metric,
    MsSsim,
    Qabf,
    SCD,
    SpatialFrequency,
    StandardDeviation,
)
from quantiloom_fusion.evaluation.full_ref import PSNR_CEILING_DB, psnr, ssim
from quantiloom_fusion.evaluation.no_reference import entropy
from quantiloom_fusion.fusion import FusionOutput, create
from quantiloom_fusion.geometry import Warp
from quantiloom_fusion.signal.colour import luminance

from paper_exp_loader import PAPER_EXP, paper_exp_modules
from test_fusion_align import _input

H, W = 48, 64


@pytest.fixture
def rng():
    return np.random.default_rng(20240814)


@pytest.fixture
def noise(rng):
    return rng.random((H, W)).astype(np.float32)


def _identity_warp(src, shape=(H, W)):
    h, w = shape
    map_xy = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float64)
    return Warp(Band.VIS, src, map_xy, np.ones((h, w), bool))


def _three_band(rng, support=None):
    """A FusionInput: vis reference + mwir + lwir, identity warps."""
    images = {b: rng.random((H, W)).astype(np.float32)
              for b in (Band.VIS, Band.MWIR, Band.LWIR)}
    valid = np.ones((H, W), bool) if support is None else support
    warps = {}
    for b in (Band.MWIR, Band.LWIR):
        w = _identity_warp(b)
        warps[b] = Warp(Band.VIS, b, w.map_xy, valid)
    return _input(images, warps)


# --- Ported from paper-exp tests/test_metrics.py --------------------------


def test_psnr_of_identical_images_is_capped_not_infinite(noise):
    value = psnr(noise, noise)
    assert np.isfinite(value)
    assert value == PSNR_CEILING_DB


def test_psnr_matches_its_definition(rng):
    ref = rng.random((32, 32))
    img = np.clip(ref + 0.1, 0, 1)
    mse = float(np.mean((img - ref) ** 2))
    assert psnr(img, ref) == pytest.approx(10 * np.log10(1 / mse), rel=1e-9)


def test_full_reference_metrics_reject_shape_mismatch(rng):
    for fn in (psnr, ssim):
        with pytest.raises(ValueError):
            fn(rng.random((64, 64)), rng.random((32, 32)))


def test_ssim_handles_images_smaller_than_the_window(rng):
    small = rng.random((5, 5))
    assert ssim(small, small) == pytest.approx(1.0, abs=1e-6)


# --- Equivalence against the original implementation ----------------------

@pytest.mark.skipif(not PAPER_EXP.exists(), reason="paper-exp checkout not present")
def test_psnr_ssim_equivalence(rng):
    a = rng.random((80, 64)).astype(np.float32)
    b = np.clip(a + 0.05 * rng.random((80, 64)), 0, 1)
    with paper_exp_modules():
        from paper_exp.metrics import full_ref as orig

        assert psnr(a, b) == pytest.approx(orig.psnr(a, b), abs=1e-12)
        assert ssim(a, b) == pytest.approx(orig.ssim(a, b), abs=1e-12)
        assert psnr(a, a) == orig.psnr(a, a)


# --- Metric wrappers --------------------------------------------------------

ALL_WRAPPERS = [Entropy, StandardDeviation, SpatialFrequency, AverageGradient,
                SCD, Qabf, MsSsim]


def test_wrappers_are_metrics_with_names_and_version():
    for cls in ALL_WRAPPERS:
        m = cls()
        assert isinstance(m, Metric) and m.version == 1 and isinstance(m.name, str)
    assert {cls().name for cls in ALL_WRAPPERS} == {
        "EN", "SD", "SF", "AG", "SCD", "Qabf", "MS_SSIM"}


def test_wrappers_score_a_fusion_output(rng):
    x = _three_band(rng)
    out = create("average").fuse(x)
    for cls in ALL_WRAPPERS:
        result = cls()(out, x, sample=None)
        assert set(result) == {cls().name}
        assert np.isfinite(result[cls().name]), cls.__name__


def test_wrapper_mask_restricts_to_common_support(rng):
    # MWIR only exists on the left half; the eval region is its intersection
    # with the other supports, i.e. the left half.
    support = np.zeros((H, W), bool)
    support[:, : W // 2] = True
    x = _three_band(rng, support)
    flat_right = np.zeros((H, W), np.float32)
    flat_right[:, : W // 2] = 0.5
    # Constant inside the region, constant outside too but at a different
    # level: SD inside the region must be zero — it only sees the left half.
    result = StandardDeviation()(FusionOutput(flat_right), x, sample=None)
    assert result["SD"] == pytest.approx(0.0, abs=1e-9)


def test_wrapper_rgb_output_uses_luminance(rng):
    x = _three_band(rng)
    rgb = rng.random((H, W, 3)).astype(np.float32)
    result = Entropy()(FusionOutput(rgb), x, sample=None)
    assert result["EN"] == pytest.approx(entropy(luminance(rgb)), abs=1e-12)

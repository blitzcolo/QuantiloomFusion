"""Masked and noise-aware Laplacian pyramid fusion: limit cases and noise behaviour."""

import cv2
import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.fusion import available, create
from quantiloom_fusion.fusion.classical import LaplacianPyramid
from quantiloom_fusion.fusion.noise_aware import (
    fuse_coefficients,
    level_noise_gain,
    shrink,
)
from quantiloom_fusion.geometry import Warp

from test_fusion_align import _input

H, W = 128, 192
RECT = (slice(40, 88), slice(60, 132))  # footprint rows, cols
LEVELS = 4


def _warp(valid=None):
    map_xy = np.dstack(np.meshgrid(np.arange(W), np.arange(H))).astype(np.float64)
    return Warp(Band.VIS, Band.LWIR, map_xy, np.ones((H, W), bool) if valid is None else valid)


def _rect_valid():
    valid = np.zeros((H, W), bool)
    valid[RECT] = True
    return valid


def _fuse(name, vis, ir, valid=None, **cfg):
    x = _input({Band.VIS: vis, Band.LWIR: ir}, {Band.LWIR: _warp(valid)})
    return create(name, **cfg).fuse(x)


def _noise_aware(vis, ir, sigma, valid=None, kappa=2.5):
    return _fuse("noise_aware_pyramid", vis, ir, valid, sigma={Band.VIS: sigma, Band.LWIR: sigma},
                 levels=LEVELS, kappa=kappa)


def _textured(seed):
    rng = np.random.default_rng(seed)
    return rng.random((H, W)).astype(np.float32)


def test_registered():
    assert {"laplacian_pyramid", "laplacian_pyramid_masked", "noise_aware_pyramid"} <= set(available())


def test_sigma_zero_equals_masked_baseline():
    vis, ir = _textured(1), _textured(2)
    valid = _rect_valid()
    base = _fuse("laplacian_pyramid_masked", vis, ir, valid, levels=LEVELS).image
    aware = _noise_aware(vis, ir, 0.0, valid).image
    np.testing.assert_array_equal(aware, base)


def test_full_masks_and_zero_sigma_equal_standard_pyramid():
    vis, ir = _textured(3), _textured(4)
    std = _fuse("laplacian_pyramid", vis, ir, levels=LEVELS).image
    aware = _noise_aware(vis, ir, 0.0).image
    masked = _fuse("laplacian_pyramid_masked", vis, ir, levels=LEVELS).image
    np.testing.assert_allclose(aware, std, atol=1e-6)
    np.testing.assert_allclose(masked, std, atol=1e-6)


def test_selection_compares_shrunk_magnitudes_not_snr():
    coeffs = np.array([0.40, 0.030]).reshape(2, 1, 1)
    sigma = np.array([0.08, 0.005]).reshape(2, 1, 1)
    ones = np.ones((2, 1, 1))
    np.testing.assert_allclose(shrink(coeffs, sigma, 2.0).ravel(), [0.24, 0.02])
    out = fuse_coefficients(coeffs, sigma, ones, kappa=2.0)
    np.testing.assert_allclose(out, [[0.24]])


def test_shrink_zero_coefficient_and_zero_sigma():
    c = np.array([0.0, 0.3, -0.3])
    np.testing.assert_array_equal(shrink(c, 0.0, 2.5), c)
    np.testing.assert_allclose(shrink(c, 0.1, 2.0), [0.0, 0.1, -0.1], atol=1e-15)
    assert shrink(np.array([0.1]), 0.1, 2.5)[0] == 0.0


def test_weights_are_applied_before_selection():
    coeffs = np.array([0.3, 0.5]).reshape(2, 1, 1)
    weights = np.array([1.0, 0.4]).reshape(2, 1, 1)  # 0.5 * 0.4 = 0.2 < 0.3
    out = fuse_coefficients(coeffs, np.zeros((2, 1, 1)), weights, kappa=None)
    np.testing.assert_allclose(out, [[0.3]])


def test_pure_noise_gives_exactly_zero_detail():
    sigma = 0.02
    rng = np.random.default_rng(5)
    vis = (0.5 + sigma * rng.standard_normal((H, W))).astype(np.float64)
    ir = (0.5 + sigma * rng.standard_normal((H, W))).astype(np.float64)
    out = _noise_aware(vis, ir, sigma, kappa=6.0)
    assert all(not d.any() for d in out.extras["fused_detail"])
    base = _fuse("laplacian_pyramid_masked", vis, ir, levels=LEVELS)
    assert any(d.any() for d in base.extras["fused_detail"])


def _noisy_scene(sigma, seed=6):
    rng = np.random.default_rng(seed)
    vis = 0.5 + sigma * rng.standard_normal((H, W))
    ir = 0.5 + sigma * rng.standard_normal((H, W))
    return vis, ir


def test_footprint_outside_equals_reference_shrunk_coefficients():
    sigma, kappa = 0.02, 2.5
    vis, ir = _noisy_scene(sigma)
    out = _noise_aware(vis, ir, sigma, _rect_valid(), kappa)
    ref_lap = LaplacianPyramid._pyramid(vis, LEVELS)
    far = np.ones((H, W), bool)
    far[20:108, 40:152] = False  # weights are exactly 0 beyond this margin
    for lv, detail in enumerate(out.extras["fused_detail"]):
        want = shrink(ref_lap[lv], sigma * level_noise_gain(LEVELS)[lv], kappa)
        m = far if lv == 0 else np.kron(far[::2**lv, ::2**lv], np.ones((1, 1), bool))[:detail.shape[0], :detail.shape[1]]
        np.testing.assert_allclose(detail[m], want[m], atol=1e-12)


def test_noise_floor_same_inside_and_outside_footprint():
    sigma = 0.02
    inside = (slice(52, 76), slice(76, 116))
    outside = (slice(100, 124), slice(10, 50))

    def highpass_std(img, region):
        return float(np.std((img - cv2.GaussianBlur(img, (0, 0), 2))[region]))

    for seed in (6, 7, 8):
        vis, ir = _noisy_scene(sigma, seed)
        aware = _noise_aware(vis, ir, sigma, _rect_valid()).image.astype(np.float64)
        base = _fuse("laplacian_pyramid_masked", vis, ir, _rect_valid(), levels=LEVELS)
        base = base.image.astype(np.float64)
        assert highpass_std(base, outside) > 0.8 * sigma  # the metric sees unshrunk noise
        for region in (inside, outside):
            assert highpass_std(aware, region) < 0.1 * sigma


def test_boundary_transition_is_smooth_without_overshoot():
    vis = np.full((H, W), 0.3)
    ir = np.full((H, W), 0.7)
    out = _noise_aware(vis, ir, 0.01, _rect_valid()).image.astype(np.float64)
    assert out.min() >= 0.3 - 1e-6 and out.max() <= 0.5 + 1e-6
    row = out[64]
    assert np.all(np.diff(row[:96]) >= -1e-6)  # monotone up across the left edge
    assert np.all(np.diff(row[96:]) <= 1e-6)  # and back down at the right edge
    assert np.abs(np.diff(row)).max() < 0.2 / 3  # no single-pixel step of the 0.2 jump
    assert 0.48 < out[64, 96] <= 0.5 + 1e-6 and out[5, 5] == pytest.approx(0.3, abs=1e-6)


def test_level_sigma_matches_white_noise_std():
    sigma = 0.02
    noise = sigma * np.random.default_rng(11).standard_normal((2048, 2048))
    laps = LaplacianPyramid._pyramid(noise, LEVELS)
    x = _input({Band.VIS: np.zeros((H, W)), Band.LWIR: np.zeros((H, W))}, {Band.LWIR: _warp()})
    out = create("noise_aware_pyramid", sigma={Band.VIS: sigma, Band.LWIR: sigma},
                 levels=LEVELS).fuse(x)
    for lv, got in enumerate(out.extras["sigma_level"][Band.VIS]):
        assert float(np.std(laps[lv])) == pytest.approx(got, rel=0.03)


def test_missing_sigma_and_extras():
    vis, ir = _textured(7), _textured(8)
    x = _input({Band.VIS: vis, Band.LWIR: ir}, {Band.LWIR: _warp()})
    with pytest.raises(ValueError, match="sigma"):
        create("noise_aware_pyramid", sigma={Band.VIS: 0.01}).fuse(x)
    out = create("noise_aware_pyramid", sigma={Band.VIS: 0.01, Band.LWIR: 0.02},
                 noise_info={"fpn": False}).fuse(x)
    assert out.extras["kappa"] == 2.5 and out.extras["noise_info"] == {"fpn": False}
    assert set(out.extras["support"]) == {Band.VIS, Band.LWIR}

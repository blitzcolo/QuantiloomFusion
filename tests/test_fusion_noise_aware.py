"""Masked and noise-aware Laplacian pyramid fusion: limit cases and noise behaviour."""

import cv2
import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.fusion import available, create
from quantiloom_fusion.fusion.classical import LaplacianPyramid
import math

from quantiloom_fusion.dataset.types import Readout
from quantiloom_fusion.fusion.noise_aware import (
    chain_level_gains,
    fuse_coefficients,
    level_noise_gain,
    shrink,
)
from quantiloom_fusion.geometry import Warp
from quantiloom_fusion.geometry.warp import apply_warp
from quantiloom_fusion.signal.noise import vis_luminance_chain

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
    # The masked operators clip at 1 only (a dim RAW input keeps its sub-black noise);
    # the standard pyramid, a paper-exp port, also clips at 0.
    assert aware.min() < 0.0 and masked.min() < 0.0
    np.testing.assert_allclose(np.maximum(aware, 0.0), std, atol=1e-6)
    np.testing.assert_allclose(np.maximum(masked, 0.0), std, atol=1e-6)


@pytest.mark.parametrize("name", ["laplacian_pyramid_masked", "noise_aware_pyramid"])
def test_dim_input_below_zero_passes_through_unrectified(name):
    # A dim VIS RAW input (VisRawLinear keeps the sub-black side) with no IR support anywhere:
    # the fused image is the VIS image itself, negative values and mean included.
    rng = np.random.default_rng(5)
    vis = (2e-4 + 1e-3 * rng.standard_normal((H, W))).astype(np.float32)
    cfg = {"sigma": {Band.VIS: 0.0, Band.LWIR: 0.0}, "gain_model": "white"} if name == "noise_aware_pyramid" else {}
    out = _fuse(name, vis, _textured(6), np.zeros((H, W), bool), levels=LEVELS, **cfg).image
    assert out.min() < 0.0
    np.testing.assert_allclose(out, vis, atol=1e-6)
    assert out.mean() == pytest.approx(vis.mean(), abs=1e-7)


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


def _upsample(ratio, ref=(480, 640)):
    # A source grid 1/ratio the reference's, resampled bilinearly onto it.
    h, w = ref
    native = (math.ceil((h - 1) / ratio) + 1, math.ceil((w - 1) / ratio) + 1)
    j, i = np.meshgrid(np.arange(w), np.arange(h))
    warp = Warp(Band.VIS, Band.LWIR, np.dstack([j / ratio, i / ratio]).astype(np.float64), np.ones(ref, bool))
    return native, warp, (lambda n: apply_warp(n, warp, "linear"))


def _chain(name):
    if name == "vis_demosaic_luma":
        readout = Readout("rggb", 12, 0.0, 1.0, 1.0, (2.0, 1.0, 1.5), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        return (480, 640), vis_luminance_chain(readout)
    native, _, chain = _upsample({"lwir_x1.74": 1.74, "mwir_x1.17": 1.17}[name])
    return native, chain


def _survival(chain, native, gains, kappa=3.0, sigma=0.01, draws=4):
    # Fraction of pure-noise detail coefficients a soft threshold keeps, per level.
    kept, total = np.zeros(LEVELS), np.zeros(LEVELS)
    for seed in range(draws):
        noise = np.random.default_rng(100 + seed).standard_normal(native) * sigma
        laps = LaplacianPyramid._pyramid(np.asarray(chain(noise), np.float64), LEVELS)
        for lv, g in enumerate(gains):
            c = laps[lv][4:-4, 4:-4]
            kept[lv] += np.count_nonzero(shrink(c, sigma * g, kappa))
            total[lv] += c.size
    return list(kept / total)


@pytest.mark.parametrize("name", ["lwir_x1.74", "mwir_x1.17", "vis_demosaic_luma"])
def test_chain_gains_make_pure_noise_survival_match_the_gaussian_tail_at_every_level(name):
    native, chain = _chain(name)
    tail = math.erfc(3.0 / math.sqrt(2.0))  # P(|c| > 3 sigma) = 0.0027
    measured = chain_level_gains(chain, native, LEVELS)
    for level, frac in enumerate(_survival(chain, native, measured)):
        assert 0.3 * tail < frac < 3.0 * tail, (level, frac)
    # The white-noise gain misstates at least one level by a large factor.
    white = _survival(chain, native, level_noise_gain(LEVELS))
    assert max(white) > 5 * tail or min(white) < 0.2 * tail, white


def test_identity_chain_reproduces_the_white_noise_gain():
    measured = chain_level_gains(lambda n: n, (256, 256), LEVELS)
    np.testing.assert_allclose(measured, level_noise_gain(LEVELS), rtol=0.03)


def test_noise_aware_fusion_uses_the_warp_chain_gains_by_default():
    native, warp, _ = _upsample(1.74, (H, W))
    rng = np.random.default_rng(0)
    x = _input({Band.VIS: rng.random((H, W)).astype(np.float32), Band.LWIR: rng.random(native).astype(np.float32)},
               {Band.LWIR: warp})
    out = create("noise_aware_pyramid", sigma={Band.VIS: 0.01, Band.LWIR: 0.01}).fuse(x)
    white = level_noise_gain(LEVELS)
    assert out.extras["gain_model"] == "chain"
    assert out.extras["level_gain"][Band.VIS] == list(white)
    assert out.extras["level_gain"][Band.LWIR][0] < 0.7 * white[0]
    assert out.extras["level_gain"][Band.LWIR][-1] > 1.3 * white[-1]
    plain = create("noise_aware_pyramid", sigma={Band.VIS: 0.01, Band.LWIR: 0.01}, gain_model="white").fuse(x)
    assert plain.extras["level_gain"][Band.LWIR] == list(white)

"""Epipolar inverse-depth sweep and free-search control on a synthetic textured plane."""

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.geometry import oracle_warp
from quantiloom_fusion.registration import (
    EpipolarSweepRegistrar,
    FreeSearchRegistrar,
    InfinityHomographyRegistrar,
    Similarity,
    SweepParams,
    registration_error,
)

from test_geometry_oracle import geometry, raycast
from test_registration import make_obs

D_TRUE = 0.22
Z_TRUE = 1.0 / D_TRUE
BASELINE_M = 0.4
N_HYP = 9
Z_MIN = 2.5  # d_max = 0.4, step 0.05: the bin containing D_TRUE is index 4


def texture(xy: np.ndarray) -> np.ndarray:
    """Smooth world-space texture in [0, 1] (sum of random plane waves, 0.25-0.8 m)."""
    rng = np.random.default_rng(7)
    k = 2 * np.pi / rng.uniform(0.25, 0.8, 40)
    th = rng.uniform(0, 2 * np.pi, 40)
    ph = rng.uniform(0, 2 * np.pi, 40)
    wave = np.cos(xy[..., 0:1] * (k * np.cos(th)) + xy[..., 1:2] * (k * np.sin(th)) + ph).sum(-1)
    return 0.5 + 0.5 * np.tanh(wave / 4.0)


def plane_pair(textured=True, flat_half=True):
    gr = geometry(128, 96, 200.0)
    gs = geometry(64, 48, 80.0, t=(BASELINE_M, 0.0, 0.0), yaw=0.03)
    ref, src = raycast(gr, offset=Z_TRUE), raycast(gs, offset=Z_TRUE)
    t_ref, t_src = texture(ref.world_position[..., :2]), texture(src.world_position[..., :2])
    noise = np.random.default_rng(5)
    flat = lambda xy: (xy[..., 0] < 0.0) & flat_half  # untextured half: the frame's own noise floor
    t_ref = np.where(flat(ref.world_position), 0.5, t_ref) + 0.003 * noise.standard_normal(t_ref.shape)
    t_src = np.where(flat(src.world_position), 0.5, t_src) + 0.003 * noise.standard_normal(t_src.shape)
    if not textured:
        noise = np.random.default_rng(1)
        t_ref = 0.5 + 1e-4 * noise.standard_normal(t_ref.shape)
        t_src = 0.5 + 1e-4 * noise.standard_normal(t_src.shape)
    # VIS-vs-IR stand-in: |2t - 1| is non-monotonic and flips edge polarity.
    return ref, src, t_ref, np.abs(2 * t_src - 1)


def obs(ref, src):
    return make_obs(Band.VIS, ref.geometry), make_obs(Band.LWIR, src.geometry)


@pytest.mark.parametrize("similarity", list(Similarity))
def test_sweep_recovers_true_inverse_depth_bin_with_polarity_flip(similarity):
    ref, src, img_ref, img_src = plane_pair()
    reg = EpipolarSweepRegistrar(SweepParams(similarity=similarity, n_hypotheses=N_HYP, z_min_m=Z_MIN))
    warp = reg.estimate(*obs(ref, src), img_ref, img_src)
    hyp = warp.provenance["hypotheses"]
    confident = warp.provenance["confident"].copy()
    confident[:12], confident[-12:], confident[:, :12], confident[:, -12:] = False, False, False, False
    assert confident.mean() > 0.1
    est_bin = np.abs(warp.provenance["d_map"][confident][:, None] - hyp[None]).argmin(axis=1)
    true_bin = int(np.abs(hyp - D_TRUE).argmin())
    assert (est_bin == true_bin).mean() > 0.8
    oracle = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    homography = InfinityHomographyRegistrar().estimate(*obs(ref, src), None, None)
    assert registration_error(warp, oracle)["epe_mean"] < registration_error(homography, oracle)["epe_mean"]


@pytest.mark.parametrize("similarity", list(Similarity))
def test_single_zero_hypothesis_equals_infinity_homography(similarity):
    ref, src, img_ref, img_src = plane_pair()
    reg = EpipolarSweepRegistrar(SweepParams(similarity=similarity, n_hypotheses=1))
    warp = reg.estimate(*obs(ref, src), img_ref, img_src)
    expected = InfinityHomographyRegistrar().estimate(*obs(ref, src), None, None)
    np.testing.assert_allclose(warp.map_xy, expected.map_xy, atol=1e-9, equal_nan=True)
    np.testing.assert_array_equal(warp.valid, expected.valid)


@pytest.mark.parametrize("similarity", list(Similarity))
def test_textureless_region_has_low_confidence_and_falls_back_to_infinity(similarity):
    ref, src, img_ref, img_src = plane_pair(textured=False)
    reg = EpipolarSweepRegistrar(SweepParams(similarity=similarity, n_hypotheses=N_HYP, z_min_m=Z_MIN))
    warp = reg.estimate(*obs(ref, src), img_ref, img_src)
    assert warp.provenance["confident"].mean() < 0.02
    expected = InfinityHomographyRegistrar().estimate(*obs(ref, src), None, None)
    far_from_fallback = ~warp.provenance["confident"]
    np.testing.assert_allclose(warp.map_xy[far_from_fallback], expected.map_xy[far_from_fallback], atol=1e-9)
    assert (warp.provenance["d_map"][far_from_fallback] == 0).all()


@pytest.mark.parametrize("noisy_side", ["ref", "src"])
@pytest.mark.parametrize("similarity", list(Similarity))
def test_pure_noise_on_either_side_gives_no_confident_pixels_and_homography_map(similarity, noisy_side):
    ref, src, img_ref, img_src = plane_pair(flat_half=False)  # the textured side has no flat region to hide behind
    noise = np.random.default_rng(3)
    if noisy_side == "ref":
        img_ref = 0.5 + 0.02 * noise.standard_normal(img_ref.shape)
    else:
        img_src = 0.5 + 0.02 * noise.standard_normal(img_src.shape)
    sigmas = {"ref_sigma": 0.02 if noisy_side == "ref" else 0.003, "src_sigma": 0.02 if noisy_side == "src" else 0.006}
    params = SweepParams(similarity=similarity, n_hypotheses=N_HYP, z_min_m=Z_MIN, **sigmas)
    warp = EpipolarSweepRegistrar(params).estimate(*obs(ref, src), img_ref, img_src)
    expected = InfinityHomographyRegistrar().estimate(*obs(ref, src), None, None)
    assert not warp.provenance["confident"].any()
    np.testing.assert_allclose(warp.map_xy, expected.map_xy, atol=1e-9, equal_nan=True)


def test_free_search_control_returns_a_warp_with_search_range_of_the_sweep():
    ref, src, img_ref, img_src = plane_pair()
    params = SweepParams(similarity=Similarity.LOCAL_MI, n_hypotheses=N_HYP, z_min_m=Z_MIN)
    warp = FreeSearchRegistrar(params).estimate(*obs(ref, src), img_ref, img_src)
    assert warp.map_xy.shape == (96, 128, 2) and warp.valid.dtype == bool
    expected_disp = 80.0 * BASELINE_M * (1 / Z_MIN)  # f_src * b * d_max
    assert warp.provenance["max_displacement_px"] == pytest.approx(expected_disp, rel=0.05)
    assert np.abs(warp.provenance["offset_map"]).max() <= warp.provenance["max_displacement_px"] + 1.0
    oracle = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    homography = InfinityHomographyRegistrar().estimate(*obs(ref, src), None, None)
    assert registration_error(warp, oracle)["epe_median"] < registration_error(homography, oracle)["epe_median"]

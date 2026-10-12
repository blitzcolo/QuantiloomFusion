"""Classical baselines: registry, N-band behaviour, paper-exp equivalence."""

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.fusion import available, create
from quantiloom_fusion.geometry import Warp

from paper_exp_loader import PAPER_EXP, paper_exp_modules
from test_fusion_align import _input

H, W = 48, 64


def _identity_warp(ref=Band.VIS, src=Band.LWIR):
    map_xy = np.dstack(np.meshgrid(np.arange(W), np.arange(H))).astype(np.float64)
    return Warp(ref, src, map_xy, np.ones((H, W), bool))


def _two_band(vis, ir):
    return _input({Band.VIS: vis, Band.LWIR: ir}, {Band.LWIR: _identity_warp()})


def test_registered():
    assert {"average", "max_abs", "laplacian_pyramid"} <= set(available())


def test_three_band_mean_and_max():
    a = np.full((H, W), 0.1, np.float32)
    b = np.full((H, W), 0.5, np.float32)
    c = np.full((H, W), 0.9, np.float32)
    warps = {Band.MWIR: _identity_warp(Band.VIS, Band.MWIR),
             Band.LWIR: _identity_warp(Band.VIS, Band.LWIR)}
    x = _input({Band.VIS: a, Band.MWIR: b, Band.LWIR: c}, warps)
    assert np.allclose(create("average").fuse(x).image, 0.5)
    assert np.allclose(create("max_abs").fuse(x).image, 0.9)
    out = create("average").fuse(x)
    assert set(out.extras["support"]) == set(Band) and out.image.dtype == np.float32


def test_partial_weights_split_the_rest():
    a = np.full((H, W), 1.0, np.float32)
    b = np.zeros((H, W), np.float32)
    x = _two_band(a, b)
    out = create("average", weights={Band.VIS: 1.0}).fuse(x).image
    assert np.allclose(out, 1.0)


def test_needs_registration_rejects_missing_warp():
    x = _input({Band.VIS: np.zeros((H, W)), Band.LWIR: np.zeros((H, W))}, {})
    with pytest.raises(ValueError, match="registration|warp"):
        create("average").fuse(x)


@pytest.mark.skipif(not PAPER_EXP.exists(), reason="paper-exp checkout not present")
def test_two_band_equivalence_with_paper_exp():
    rng = np.random.default_rng(42)
    vis = rng.random((H, W)).astype(np.float32)
    ir = rng.random((H, W)).astype(np.float32)
    x = _two_band(vis, ir)
    with paper_exp_modules():
        from paper_exp.baselines import classical as orig

        pairs = [
            (create("average"), orig.AverageFusionBaseline()),
            (create("average", weights={Band.VIS: 0.5}),
             orig.AverageFusionBaseline(weight_vis=0.5)),
            (create("max_abs"), orig.MaxAbsoluteFusion()),
            (create("laplacian_pyramid"), orig.LaplacianPyramidFusion()),
        ]
        for ours, theirs in pairs:
            ours_img = ours.fuse(x).image
            theirs_img = theirs.fuse(vis, ir)
            diff = np.max(np.abs(ours_img - theirs_img))
            assert diff <= 1e-6, f"{ours.name}: max diff {diff}"

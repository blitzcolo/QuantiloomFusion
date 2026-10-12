"""aligned_sources and colourize on tiny synthetic arrays."""

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.fusion import FusionInput, aligned_sources, colourize
from quantiloom_fusion.geometry import Warp

from test_geometry_oracle import geometry


def _warp(g_src, g_ref=None):
    g_ref = g_ref or g_src
    h, w = g_ref.height, g_ref.width
    map_xy = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float64)
    return Warp(Band.VIS, Band.LWIR, map_xy, np.ones((h, w), bool))


def _input(images, warps, reference=Band.VIS):
    geoms = {b: geometry(a.shape[1], a.shape[0], 4.0) for b, a in images.items()}
    return FusionInput(reference=reference, images=images, geometry=geoms, warps=warps)


def test_aligned_identity_passthrough():
    vis = np.arange(48, dtype=np.float32).reshape(6, 8) / 48
    lwir = np.linspace(0, 1, 48, dtype=np.float32).reshape(6, 8)
    x = _input({Band.VIS: vis, Band.LWIR: lwir}, {Band.LWIR: _warp(geometry(8, 6, 4.0))})
    images, supports = aligned_sources(x)
    assert np.array_equal(images[Band.VIS], vis)
    assert np.allclose(images[Band.LWIR], lwir)
    assert supports[Band.VIS].all() and supports[Band.LWIR].all()


def test_aligned_shift_and_fill():
    vis = np.zeros((6, 8), np.float32)
    lwir = np.arange(48, dtype=np.float32).reshape(6, 8)
    w = _warp(geometry(8, 6, 4.0))
    # map_xy shifted +1 in x: column j samples source j+1 -> col 7 out of range.
    map_xy = w.map_xy.copy()
    map_xy[..., 0] += 1.0
    valid = np.ones((6, 8), bool)
    valid[:, 7] = False
    w = Warp(Band.VIS, Band.LWIR, map_xy, valid)
    x = _input({Band.VIS: vis, Band.LWIR: lwir}, {Band.LWIR: w})
    images, supports = aligned_sources(x)
    assert np.allclose(images[Band.LWIR][:, :7], lwir[:, 1:8])
    assert np.array_equal(images[Band.LWIR][:, 7], vis[:, 7])  # ref fill
    assert not supports[Band.LWIR][:, 7].any() and supports[Band.LWIR][:, :7].all()


def test_missing_warp_raises():
    x = _input({Band.VIS: np.zeros((6, 8)), Band.LWIR: np.zeros((6, 8))}, {})
    with pytest.raises(ValueError, match="warp"):
        aligned_sources(x)


def test_colourize_restores_vis_when_y_matches():
    rng = np.random.default_rng(0)
    vis_rgb = rng.random((6, 8, 3))
    y = vis_rgb @ np.array([0.299, 0.587, 0.114])
    out = colourize(y, vis_rgb)
    assert out.dtype == np.float32
    assert np.allclose(out, vis_rgb, atol=1e-6)


def test_colourize_fused_grey_keeps_chroma_direction():
    vis_rgb = np.zeros((4, 4, 3))
    vis_rgb[..., 0], vis_rgb[..., 1] = 0.8, 0.2  # reddish
    out = colourize(np.full((4, 4), 0.5), vis_rgb)
    assert np.all(out[..., 0] > out[..., 2])  # still reddish
    assert 0.0 <= out.min() and out.max() <= 1.0

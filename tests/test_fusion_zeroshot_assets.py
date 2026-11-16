"""Torch-free input preparation of the zero-shot models (no weights needed)."""

import numpy as np
import pytest

from quantiloom_fusion.dataset.types import CameraGeometry
from quantiloom_fusion.fusion.zeroshot_assets import (
    canvas_to_native,
    file_sha256,
    lwir_canvas_8bit,
    pure_scale_map,
    quantise_8bit,
    umf_cmgr_inputs,
)


def _geometry(w: int, h: int, f: float) -> CameraGeometry:
    eye = (1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0)
    return CameraGeometry(w, h, (f, 0, w / 2, 0, f, h / 2, 0, 0, 1), (0,) * 5, eye, eye)


VIS, LWIR = _geometry(128, 96, 100.0), _geometry(64, 48, 50.0)


def test_umf_cmgr_inputs_shapes_dtypes_and_ranges():
    rng = np.random.default_rng(0)
    vis = rng.random((96, 128, 3), dtype=np.float32) * 1.2 - 0.1  # out-of-range values are clipped
    out = umf_cmgr_inputs(vis, rng.random((48, 64), dtype=np.float32), VIS, LWIR)
    assert out["vis_8bit"].shape == (96, 128, 3) and out["vis_8bit"].dtype == np.uint8
    assert out["cpstn_rgb"].shape == (3, 96, 128) and out["cpstn_rgb"].dtype == np.float32
    assert -1.0 <= out["cpstn_rgb"].min() and out["cpstn_rgb"].max() <= 1.0
    for key in ("vi", "ir"):
        assert out[key].shape == (96, 128) and out[key].dtype == np.float32
        assert 0.0 <= out[key].min() and out[key].max() <= 1.0
        np.testing.assert_allclose(out[key] * 255.0, np.round(out[key] * 255.0), atol=1e-4)  # 8-bit levels


def test_umf_cmgr_grey_is_bt601_of_the_8bit_display_image():
    rng = np.random.default_rng(1)
    vis = rng.random((96, 128, 3), dtype=np.float32)
    out = umf_cmgr_inputs(vis, np.zeros((48, 64), np.float32), VIS, LWIR)
    v8 = out["vis_8bit"].astype(np.float64)
    grey = 0.299 * v8[..., 0] + 0.587 * v8[..., 1] + 0.114 * v8[..., 2]
    assert np.abs(out["vi"] * 255.0 - grey).max() <= 0.51
    np.testing.assert_allclose(out["cpstn_rgb"], v8.transpose(2, 0, 1) / 127.5 - 1.0, atol=1e-6)


def test_umf_cmgr_inputs_reject_bad_frames():
    with pytest.raises(ValueError, match="multiples of 4"):
        umf_cmgr_inputs(np.zeros((90, 126, 3)), np.zeros((48, 64)), _geometry(126, 90, 100.0), LWIR)
    with pytest.raises(ValueError, match="does not match"):
        umf_cmgr_inputs(np.zeros((96, 124, 3)), np.zeros((48, 64)), VIS, LWIR)
    with pytest.raises(ValueError, match="display image"):
        umf_cmgr_inputs(np.zeros((96, 128)), np.zeros((48, 64)), VIS, LWIR)


def test_lwir_canvas_follows_the_pure_scale_map_and_is_zero_outside_the_footprint():
    lwir = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (48, 1))
    canvas = lwir_canvas_8bit(lwir, VIS, LWIR)
    rmap = pure_scale_map(VIS, LWIR)
    inside = (rmap[..., 0] >= 0) & (rmap[..., 0] <= 63) & (rmap[..., 1] >= 0) & (rmap[..., 1] <= 47)
    assert inside.any() and (~inside).any()
    assert (canvas[(rmap[..., 0] < -1) | (rmap[..., 0] > 64)] == 0).all()
    expected = quantise_8bit(np.interp(rmap[..., 0], np.arange(64), lwir[0]))
    assert np.abs(canvas[inside].astype(int) - expected[inside].astype(int)).max() <= 1
    # Equal cameras: the map is the identity.
    nx, ny = canvas_to_native(np.array([3.0]), np.array([5.0]), VIS, VIS)
    assert (nx[0], ny[0]) == (3.0, 5.0)


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "w.pth"
    p.write_bytes(b"x" * 3_000_000)
    assert file_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()

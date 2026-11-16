import numpy as np
import pytest

pytest.importorskip("torch")

from quantiloom_fusion.core.bands import Band  # noqa: E402
from quantiloom_fusion.dataset.types import CameraGeometry  # noqa: E402
from quantiloom_fusion.fusion import zeroshot  # noqa: E402
from quantiloom_fusion.fusion.base import FusionInput  # noqa: E402
from quantiloom_fusion.fusion.registry import create  # noqa: E402

needs_superfusion = pytest.mark.skipif(not zeroshot.assets_available("MSRS"),
                                       reason="SuperFusion clone/weights under build/models or CUDA missing")
needs_umf_cmgr = pytest.mark.skipif(not zeroshot.umf_cmgr_assets_available(),
                                    reason="UMF-CMGR clone/weights under build/models missing")


def _geometry(w: int, h: int, f: float) -> CameraGeometry:
    eye = (1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0)
    return CameraGeometry(w, h, (f, 0, w / 2, 0, f, h / 2, 0, 0, 1), (0,) * 5, eye, eye)


@needs_superfusion
def test_fuse_runs_unregistered_pair_and_reports_a_warp_on_the_native_lwir_grid():
    rng = np.random.default_rng(0)
    x = FusionInput(
        Band.VIS,
        {Band.VIS: rng.random((96, 128, 3), dtype=np.float32), Band.LWIR: rng.random((48, 64), dtype=np.float32)},
        {Band.VIS: _geometry(128, 96, 100.0), Band.LWIR: _geometry(64, 48, 50.0)},
    )
    out = create("superfusion_msrs").fuse(x)
    assert out.image.shape == (96, 128, 3)
    assert 0.0 <= out.image.min() and out.image.max() <= 1.0
    warp = out.extras["warps"][Band.LWIR]
    assert warp.map_xy.shape == (96, 128, 2)
    assert warp.valid.any()


def _pair(seed: int = 0) -> FusionInput:
    rng = np.random.default_rng(seed)
    return FusionInput(
        Band.VIS,
        {Band.VIS: rng.random((96, 128, 3), dtype=np.float32), Band.LWIR: rng.random((48, 64), dtype=np.float32)},
        {Band.VIS: _geometry(128, 96, 100.0), Band.LWIR: _geometry(64, 48, 50.0)},
    )


@needs_umf_cmgr
def test_umf_cmgr_loads_every_weight_file_and_fuses_on_the_vis_grid_deterministically():
    method = create("umf_cmgr", device="cpu")  # CPU: the tiny frame needs no GPU
    assert set(method.weights_sha256) == {"CPSTN/checkpoints/pretrained/latest_net_G.pth",
                                          "weights/reg_0280.pth", "weights/fus_0280.pth"}
    x = _pair()
    a, b = method.fuse(x), method.fuse(x)
    assert a.image.shape == (96, 128) and a.image.dtype == np.float32
    assert 0.0 <= a.image.min() and a.image.max() <= 1.0
    np.testing.assert_array_equal(a.image, b.image)
    warp = a.extras["warps"][Band.LWIR]
    assert warp.map_xy.shape == (96, 128, 2) and warp.valid.any()
    # The pseudo-infrared image went through an 8-bit file in the authors' pipeline.
    np.testing.assert_allclose(a.extras["pseudo_ir"] * 255.0, np.round(a.extras["pseudo_ir"] * 255.0), atol=1e-4)


@needs_umf_cmgr
def test_umf_cmgr_rejects_a_frame_that_is_not_a_multiple_of_four():
    method = create("umf_cmgr", device="cpu")
    rng = np.random.default_rng(1)
    x = FusionInput(Band.VIS, {Band.VIS: rng.random((90, 126, 3), dtype=np.float32),
                               Band.LWIR: rng.random((48, 64), dtype=np.float32)},
                    {Band.VIS: _geometry(126, 90, 100.0), Band.LWIR: _geometry(64, 48, 50.0)})
    with pytest.raises(ValueError, match="multiples of 4"):
        method.fuse(x)


def test_umf_cmgr_without_weights_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTILOOM_ZEROSHOT_MODELS", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="UMF-CMGR weights missing"):
        create("umf_cmgr", device="cpu")

import numpy as np
import pytest

pytest.importorskip("torch")

from quantiloom_fusion.core.bands import Band  # noqa: E402
from quantiloom_fusion.dataset.types import CameraGeometry  # noqa: E402
from quantiloom_fusion.fusion import zeroshot  # noqa: E402
from quantiloom_fusion.fusion.base import FusionInput  # noqa: E402
from quantiloom_fusion.fusion.registry import create  # noqa: E402

pytestmark = pytest.mark.skipif(not zeroshot.assets_available("MSRS"),
                                reason="SuperFusion clone/weights under build/models or CUDA missing")


def _geometry(w: int, h: int, f: float) -> CameraGeometry:
    eye = (1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0)
    return CameraGeometry(w, h, (f, 0, w / 2, 0, f, h / 2, 0, 0, 1), (0,) * 5, eye, eye)


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

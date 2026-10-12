"""Interface tests for the camera model and the Warp container."""

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.dataset import CameraGeometry, Observation
from quantiloom_fusion.geometry import BrownConradyCamera, Warp, apply_warp, oracle_warp

IDENTITY16 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
GEO_KWARGS = dict(
    width=256,
    height=192,
    intrinsics=(250.088550210605, 0.0, 128.0, 0.0, 250.088550210605, 96.0, 0.0, 0.0, 1.0),
    distortion=(-0.055, 0.005, 0.0008, -0.0006, 0.0),
    camera_to_world=IDENTITY16,
    world_to_camera=IDENTITY16,
)


def make_observation() -> Observation:
    return Observation(
        band=Band.VIS,
        geometry=CameraGeometry(**GEO_KWARGS),
        spectral_mode="vis_hero",
        products={},
        record="samples/s/records/vis.json",
    )


def make_warp(h=4, w=5) -> Warp:
    return Warp(
        ref_band=Band.VIS,
        src_band=Band.LWIR,
        map_xy=np.zeros((h, w, 2), dtype=np.float32),
        valid=np.ones((h, w), dtype=bool),
    )


def test_camera_from_geometry_stores_it():
    g = CameraGeometry(**GEO_KWARGS)
    cam = BrownConradyCamera.from_geometry(g)
    assert cam.geometry is g


def test_warp_valid_construction():
    warp = make_warp()
    assert warp.map_xy.shape == (4, 5, 2)
    assert warp.valid.shape == (4, 5)
    assert warp.provenance == {}


@pytest.mark.parametrize(
    "map_xy, valid",
    [
        (np.zeros((4, 5)), np.ones((4, 5), bool)),  # 2-D map
        (np.zeros((4, 5, 3)), np.ones((4, 5), bool)),  # wrong last dim
        (np.zeros((4, 5, 2)), np.ones((3, 5), bool)),  # valid shape mismatch
        (np.zeros((4, 5, 2)), np.ones((4, 5), np.float32)),  # valid not bool
    ],
)
def test_warp_rejects_bad_shapes(map_xy, valid):
    with pytest.raises(ValueError):
        Warp(Band.VIS, Band.LWIR, map_xy, valid)


def test_oracle_warp_and_apply_warp_stubs():
    obs = make_observation()
    warp = make_warp()
    with pytest.raises(NotImplementedError):
        oracle_warp(obs, obs, edge_px=2, z_rel_tol=0.05)
    with pytest.raises(NotImplementedError):
        apply_warp(np.zeros((4, 5)), warp)

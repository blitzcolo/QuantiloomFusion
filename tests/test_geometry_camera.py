"""Camera model tests: synthetic round-trips plus gated real-data checks."""

import math
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest

from quantiloom_fusion.dataset import CameraGeometry, ThreeBandDataset
from quantiloom_fusion.geometry import (
    BrownConradyCamera,
    pixel_grid,
    to_camera,
    to_world,
)

REPO = Path(__file__).resolve().parents[1]
IDENTITY16 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
CAMERAS = tomllib.loads((REPO / "configs" / "env_dataset.toml").read_text())["cameras"]
DATASET = os.environ.get("QUANTILOOM_TEST_DATASET")


def geometry_for(band: str) -> CameraGeometry:
    """Pinhole+distortion geometry matching the formal recipe's cameras.*."""
    c = CAMERAS[band]
    w, h = c["resolution"]
    f = h / (2 * math.tan(math.radians(c["fov_y_deg"]) / 2))
    return CameraGeometry(
        width=w,
        height=h,
        intrinsics=(f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0),
        distortion=tuple(c["distortion"]),
        camera_to_world=IDENTITY16,
        world_to_camera=IDENTITY16,
    )


@pytest.mark.parametrize("band", ["vis", "mwir", "lwir"])
def test_project_unproject_roundtrip_full_grid(band):
    cam = BrownConradyCamera(geometry_for(band))
    grid = pixel_grid(cam.geometry).reshape(-1, 2)
    z = np.full(len(grid), 7.5)
    err = np.abs(cam.project(cam.unproject(grid, z)) - grid).max()
    assert err < 1e-6


@pytest.mark.parametrize("band", ["vis", "mwir", "lwir"])
def test_unproject_recovers_points(band):
    cam = BrownConradyCamera(geometry_for(band))
    rng = np.random.default_rng(0)
    # points inside a modest frustum keep the distortion map injective
    pts = np.empty((2000, 3))
    pts[:, :2] = rng.uniform(-0.6, 0.6, (2000, 2))
    pts[:, 2] = rng.uniform(0.5, 50.0, 2000)
    back = cam.unproject(cam.project(pts), pts[:, 2])
    rel = np.linalg.norm(back - pts, axis=1) / np.linalg.norm(pts, axis=1)
    assert rel.max() < 1e-9


def test_project_nan_behind_camera():
    cam = BrownConradyCamera(geometry_for("vis"))
    px = cam.project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.1, 0.2, 3.0]]))
    assert np.isnan(px[:2]).all()
    assert np.isfinite(px[2]).all()


def test_to_camera_to_world_inverse():
    angle, t = 0.37, np.array([0.035, -0.002, 0.012])
    c, s = math.cos(angle), math.sin(angle)
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    c2w[:3, 3] = t
    w2c = np.linalg.inv(c2w)
    g = CameraGeometry(
        width=8,
        height=6,
        intrinsics=(3.0, 0.0, 4.0, 0.0, 3.0, 3.0, 0.0, 0.0, 1.0),
        distortion=(0.0,) * 5,
        camera_to_world=tuple(c2w.ravel()),
        world_to_camera=tuple(w2c.ravel()),
    )
    pts = np.random.default_rng(1).normal(size=(500, 3)) + [0.0, 0.0, 5.0]
    err = np.abs(to_world(g, to_camera(g, pts)) - pts)
    assert err.max() < 1e-9


@pytest.mark.skipif(not DATASET, reason="QUANTILOOM_TEST_DATASET unset")
def test_real_projection_matches_truth_products():
    ds = ThreeBandDataset(DATASET)
    sample = ds.sample(ds.sample_ids()[0])
    for band, obs in sample.observations.items():
        g = obs.geometry
        world = ds.load(obs, "truth_world_position", ("X", "Y", "Z")).astype(np.float64)
        validity = ds.load(obs, "truth_validity").ravel() == 1
        camera_z = ds.load(obs, "truth_camera_z").ravel()
        cam_pts = to_camera(g, world.reshape(-1, 3)[validity])
        projected = BrownConradyCamera(g).project(cam_pts)
        centres = pixel_grid(g).reshape(-1, 2)[validity]
        assert np.abs(projected - centres).max() <= 0.005, band
        assert np.abs(cam_pts[:, 2] - camera_z[validity]).max() <= 1e-4, band

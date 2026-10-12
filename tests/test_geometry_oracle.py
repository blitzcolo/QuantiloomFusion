"""Oracle warp tests on analytic scenes plus gated real-data checks."""

import os

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.dataset import CameraGeometry, ThreeBandDataset
from quantiloom_fusion.geometry import (
    BrownConradyCamera,
    DepthView,
    Warp,
    apply_warp,
    load_depth_view,
    oracle_diagnostics,
    oracle_warp,
    pixel_grid,
    to_camera,
)

DATASET = os.environ.get("QUANTILOOM_TEST_DATASET")


def geometry(w, h, f, distortion=(0.0,) * 5, t=(0.0, 0.0, 0.0), pitch=0.0, yaw=0.0):
    """Pinhole/distorted camera: pitch down about right (x), yaw about down (y)."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    ) @ np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
    c2w[:3, 3] = t
    return CameraGeometry(
        width=w,
        height=h,
        intrinsics=(f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0),
        distortion=distortion,
        camera_to_world=tuple(c2w.ravel()),
        world_to_camera=tuple(np.linalg.inv(c2w).ravel()),
    )


def raycast(g, normal=(0.0, 0.0, 1.0), offset=10.0, box=None):
    """DepthView of a world plane (n . P = offset) plus an optional box slab.

    box is (x0, x1, y0, y1, z_front); where the slab wins the ray the surface
    is z_front, giving true occlusion between differently-posed cameras.
    """
    n = np.asarray(normal, np.float64)
    grid = pixel_grid(g).reshape(-1, 2)
    rays = BrownConradyCamera(g).unproject(grid, np.ones(len(grid)))
    c2w = np.asarray(g.camera_to_world).reshape(4, 4)
    origin, rot = c2w[:3, 3], c2w[:3, :3]
    d = rays @ rot.T
    t = (offset - origin @ n) / (d @ n)  # n . (o + t d) = offset
    cam_z = (rays * t[:, None])[:, 2]
    pos = origin + t[:, None] * d
    if box is not None:
        x0, x1, y0, y1, zb = box
        tb = (zb - origin[2]) / d[:, 2]
        pb = origin + tb[:, None] * d
        hit = (tb > 0) & (tb < t) & (pb[:, 0] >= x0) & (pb[:, 0] <= x1) & (pb[:, 1] >= y0) & (pb[:, 1] <= y1)
        pos[hit] = pb[hit]
        cam_z[hit] = tb[hit]
    h, w = g.height, g.width
    return DepthView(
        geometry=g,
        camera_z=cam_z.reshape(h, w),
        world_position=pos.reshape(h, w, 3),
        validity=((t > 0) & np.isfinite(t)).astype(np.int32).reshape(h, w),
    )


def test_identity_warp_all_valid():
    view = raycast(geometry(16, 12, 24.0))
    warp = oracle_warp(view, view, ref_band=Band.VIS, src_band=Band.LWIR)
    assert warp.valid.all()
    assert np.abs(warp.map_xy - pixel_grid(view.geometry)).max() < 1e-9
    assert warp.provenance == {"method": "oracle", "edge_px": 1, "edge_rel_tol": 0.02, "pos_rel_tol": 0.005}
    diag = oracle_diagnostics(view, view)
    assert diag["ref_valid"] == diag["final_valid"] == view.validity.size
    assert diag["outside"] == diag["src_support"] == diag["correspondence"] == 0


def test_plane_two_views_warps_world_positions():
    ref = raycast(geometry(32, 24, 96.0))
    src = raycast(geometry(32, 24, 96.0, t=(0.5, 0.0, 0.0)))
    warp = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    # the +0.5 x shift moves every src coord ~4.8 px left: the left border
    # columns land outside the src image and are invalid, nothing else is
    assert warp.valid[:, 7:].all()
    assert not warp.valid[:, :4].any()
    warped = apply_warp(src.world_position, warp)
    rel = np.linalg.norm(warped - ref.world_position, axis=2) / ref.camera_z
    assert rel[warp.valid].max() < 1e-3


def test_box_occlusion_marks_invalid_strip():
    ref = raycast(geometry(64, 48, 48.0), box=(-2.0, 2.0, -1.5, 1.5, 5.0))
    src = raycast(geometry(64, 48, 48.0, t=(1.0, 0.0, 0.0)), box=(-2.0, 2.0, -1.5, 1.5, 5.0))
    warp = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    # ref sees the plane for |x| > 4; the +x-shifted src camera covers plane
    # points in x in [-5, 3] with the box, so a ref plane pixel at world
    # (-4.8, 0, 10) fails the position test: src's surface there is the box
    pyx = np.rint(
        BrownConradyCamera(ref.geometry).project(
            to_camera(ref.geometry, np.array([[-4.8, 0.0, 10.0]]))
        )
    ).astype(int)
    assert not warp.valid[pyx[0, 1], pyx[0, 0]]
    assert np.isfinite(warp.map_xy[pyx[0, 1], pyx[0, 0]]).all()  # invalid but projectable
    # plane at (4.6, 0, 10) is seen by both views: still valid
    far = np.rint(
        BrownConradyCamera(ref.geometry).project(
            to_camera(ref.geometry, np.array([[4.6, 0.0, 10.0]]))
        )
    ).astype(int)
    assert warp.valid[far[0, 1], far[0, 0]]
    assert 0 < (~warp.valid).sum() < warp.valid.size


def test_slanted_ground_no_false_edges():
    # ground plane Y=+3 (down-positive) seen by two cameras pitched down 20 deg
    # a few cm apart: continuous surface, grazing at the top of the frame
    pitch = np.radians(20.0)
    ref = raycast(geometry(64, 48, 96.0, pitch=pitch), normal=(0.0, 1.0, 0.0), offset=3.0)
    src = raycast(
        geometry(64, 48, 96.0, pitch=pitch, t=(0.03, 0.0, 0.02)),
        normal=(0.0, 1.0, 0.0),
        offset=3.0,
    )
    warp = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    diag = oracle_diagnostics(ref, src)
    assert diag["ref_valid"] == ref.validity.size  # whole frame hits the ground
    edge = diag["ref_edge"] + diag["src_edge"]
    assert edge < 0.01 * diag["ref_valid"], diag
    overlap = diag["ref_valid"] - diag["outside"]
    assert diag["final_valid"] > 0.95 * overlap, diag


def test_apply_warp_modes_and_nans():
    g = geometry(8, 6, 12.0)
    img = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    grid = pixel_grid(g)
    identity = _warp(grid, np.ones((6, 8), bool))
    np.testing.assert_array_equal(apply_warp(img, identity, interp="nearest"), img)
    np.testing.assert_allclose(apply_warp(img, identity), img)
    # half-pixel shift: bilinear averages the row neighbours
    shifted = _warp(grid + [0.5, 0.0], np.ones((6, 8), bool))
    out = apply_warp(img, shifted)
    assert np.isnan(out[:, -1]).all()  # shift pushes last column out of range
    np.testing.assert_allclose(out[0, 0], 0.5)
    # invalid and non-finite map entries -> NaN
    mask = np.ones((6, 8), bool)
    mask[0, 0] = False
    bad = _warp(grid, mask)
    assert np.isnan(apply_warp(img, bad)[0, 0])
    assert apply_warp(img.astype(np.int32), identity).dtype == np.float32
    assert apply_warp(img.astype(np.float64), identity).dtype == np.float64
    with pytest.raises(ValueError):
        apply_warp(img, identity, interp="cubic")


def _warp(map_xy, valid):
    return Warp(Band.VIS, Band.LWIR, np.asarray(map_xy, np.float64), np.asarray(valid, bool))


@pytest.mark.skipif(not DATASET, reason="QUANTILOOM_TEST_DATASET unset")
def test_oracle_warp_real_sample():
    ds = ThreeBandDataset(DATASET)
    sample = ds.sample(ds.sample_ids()[0])
    views = {b: load_depth_view(ds, sample.observations[b]) for b in sample.bands}
    for ref_band, src_band in ((Band.VIS, Band.LWIR), (Band.LWIR, Band.VIS)):
        warp = oracle_warp(
            views[ref_band], views[src_band], ref_band=ref_band, src_band=src_band
        )
        ref_valid = views[ref_band].validity == 1
        frac = warp.valid[ref_valid].mean()
        print(f"\n{ref_band}->{src_band}: valid {frac:.3f} of ref validity==1")
        assert frac > 0

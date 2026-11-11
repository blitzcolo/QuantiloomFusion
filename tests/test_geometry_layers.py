"""Five-layer stratification of the reference grid on analytic scenes."""

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.geometry import (
    DepthView,
    Layer,
    layer_fractions,
    layer_fractions_by_depth,
    layer_map,
    oracle_warp,
)

from test_geometry_oracle import geometry, raycast

DIST = (-0.055, 0.005, 0.0008, -0.0006, 0.0)
BOX = (-1.0, 1.0, -1.0, 1.0, 8.0)


def scene(box=None, distortion=(0.0,) * 5):
    ref = raycast(geometry(64, 48, 96.0, distortion=distortion), offset=10.0, box=box)
    src = raycast(geometry(64, 48, 96.0, distortion=distortion, t=(1.0, 0.0, 0.0), yaw=0.05),
                  offset=10.0, box=box)
    return ref, src


def with_validity(view: DepthView, rows, cols, code) -> DepthView:
    validity = view.validity.copy()
    validity[rows, cols] = code
    return DepthView(view.geometry, view.camera_z, view.world_position, validity)


SCENES = {
    "plane": lambda: scene(),
    "plane_distorted": lambda: scene(distortion=DIST),
    "box": lambda: scene(box=BOX),
}


@pytest.mark.parametrize("name", SCENES)
def test_layers_cover_grid_and_correspondable_equals_oracle_valid(name):
    ref, src = SCENES[name]()
    layers = layer_map(ref, src)
    assert layers.dtype == np.int8 and layers.shape == ref.camera_z.shape
    assert set(np.unique(layers)) <= {int(layer) for layer in Layer}
    assert sum(layer_fractions(layers).values()) == pytest.approx(1.0)
    warp = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    np.testing.assert_array_equal(layers == Layer.CORRESPONDABLE, warp.valid)
    assert warp.valid.any()


def test_invalid_precedes_every_other_layer():
    ref, src = scene()
    for code in (0, 2, 3, 4):
        layers = layer_map(with_validity(ref, slice(20, 24), slice(20, 24), code), src)
        assert (layers[20:24, 20:24] == Layer.INVALID).all(), code


def test_out_of_view_where_projection_leaves_source_image():
    ref, src = scene()
    layers = layer_map(ref, src)
    out = layers == Layer.OUT_OF_VIEW
    xy = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR).map_xy
    leaves = ~((xy[..., 0] >= 0) & (xy[..., 0] <= 63) & (xy[..., 1] >= 0) & (xy[..., 1] <= 47))
    np.testing.assert_array_equal(out, leaves)
    assert out.any() and not out.all()


def test_box_shadow_is_occluded_by_world_position_not_depth_order():
    ref, src = scene(box=BOX)
    layers = layer_map(ref, src)
    occluded = layers == Layer.OCCLUDED
    assert occluded.any()
    assert np.isclose(ref.world_position[occluded][:, 2], 10.0).any()  # plane points hidden behind the box in src
    assert (layers == Layer.EDGE).any()


def test_src_support_failure_is_occluded_not_correspondable():
    ref, src = scene()
    clean = layer_map(ref, src)
    hole = with_validity(src, slice(22, 26), slice(25, 29), 0)
    holed = layer_map(ref, hole)
    changed = (holed != clean) & (clean == Layer.CORRESPONDABLE)
    assert changed.any()
    assert np.isin(holed[changed], [Layer.OCCLUDED, Layer.EDGE]).all()
    assert (holed[changed] == Layer.OCCLUDED).any()
    assert (holed == Layer.CORRESPONDABLE).sum() < (clean == Layer.CORRESPONDABLE).sum()


def test_slanted_continuous_surface_is_not_edge():
    kw = dict(normal=(0.0, 0.6, 0.8), offset=10.0)
    ref = raycast(geometry(64, 48, 96.0, pitch=0.9), **kw)
    src = raycast(geometry(64, 48, 96.0, pitch=0.9, t=(0.05, 0.0, 0.0)), **kw)
    assert (layer_map(ref, src) == Layer.EDGE).mean() < 0.05


def test_fractions_by_depth_partition_each_bin():
    ref, src = scene(box=BOX)
    by_depth = layer_fractions_by_depth(layer_map(ref, src), ref.camera_z)
    assert by_depth[(5.0, 10.0)]["n"] > 0 and by_depth[(10.0, 20.0)]["n"] > 0
    assert by_depth[(0.0, 5.0)]["n"] == 0
    for stats in by_depth.values():
        if stats["n"]:
            assert sum(v for k, v in stats.items() if k != "n") == pytest.approx(1.0)

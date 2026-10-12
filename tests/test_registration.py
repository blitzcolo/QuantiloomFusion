"""Registrar baselines and registration_error on synthetic scenes."""

import os

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.dataset import Observation, Readout, ThreeBandDataset
from quantiloom_fusion.geometry import Warp, oracle_warp, pixel_grid
from quantiloom_fusion.registration import (
    InfinityHomographyRegistrar,
    OracleRegistrar,
    ResizeRegistrar,
    registration_error,
)

from dataset_fixture import build_fixture
from test_geometry_oracle import geometry, raycast

DATASET = os.environ.get("QUANTILOOM_TEST_DATASET")
DIST = (-0.055, 0.005, 0.0008, -0.0006, 0.0)  # the recipe's lwir lens


def make_obs(band, g, record="r"):
    readout = Readout(cfa="mono", adc_bits=14, black_level_dn=0.0, electrons_per_dn=1.0,
                      exposure_s=0.03, white_balance=(1.0, 1.0, 1.0),
                      device_to_linear_srgb=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return Observation(band=band, geometry=g, readout=readout,
                       spectral_mode="m", products={}, record=record)


def make_warp(map_xy, valid, ref=Band.VIS, src=Band.LWIR):
    return Warp(ref, src, np.asarray(map_xy, np.float64), np.asarray(valid, bool))


def test_registration_error_identical_and_offset():
    grid = pixel_grid(geometry(8, 6, 12.0))
    valid = np.ones((6, 8), bool)
    err = registration_error(make_warp(grid, valid), make_warp(grid, valid))
    assert err["epe_max"] == 0.0 and err["coverage"] == 1.0 and err["n"] == 48.0
    shifted = make_warp(grid + [0.5, 0.0], valid)
    err = registration_error(shifted, make_warp(grid, valid))
    assert err["epe_mean"] == pytest.approx(0.5) and err["coverage"] == 1.0


def test_registration_error_mismatches():
    a = make_warp(np.zeros((4, 4, 2)), np.ones((4, 4), bool))
    with pytest.raises(ValueError):
        registration_error(a, make_warp(np.zeros((4, 4, 2)), np.ones((4, 4), bool), src=Band.MWIR))
    with pytest.raises(ValueError):
        registration_error(a, make_warp(np.zeros((2, 2, 2)), np.ones((2, 2), bool)))


def test_infinity_homography_zero_baseline_exact():
    ref = raycast(geometry(64, 48, 48.0, distortion=DIST))
    src = raycast(geometry(64, 48, 48.0, distortion=DIST, yaw=np.radians(2.0)))
    oracle = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    est = InfinityHomographyRegistrar().estimate(
        make_obs(Band.VIS, ref.geometry), make_obs(Band.LWIR, src.geometry), None, None
    )
    err = registration_error(est, oracle)
    assert err["coverage"] > 0.9
    assert err["epe_max"] < 1e-3


def test_resize_pure_scaling_exact():
    ref = raycast(geometry(64, 48, 48.0))
    src = raycast(geometry(32, 24, 24.0))  # same pose, half focal and resolution
    oracle = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    est = ResizeRegistrar().estimate(
        make_obs(Band.VIS, ref.geometry), make_obs(Band.LWIR, src.geometry), None, None
    )
    err = registration_error(est, oracle)
    assert err["epe_max"] < 1e-3


def test_homography_parallax_grows_as_fb_over_z():
    f, baseline, depth = 96.0, 0.05, 2.0
    ref = raycast(geometry(64, 48, f), offset=depth)
    src = raycast(geometry(64, 48, f, t=(baseline, 0.0, 0.0)), offset=depth)
    oracle = oracle_warp(ref, src, ref_band=Band.VIS, src_band=Band.LWIR)
    est = InfinityHomographyRegistrar().estimate(
        make_obs(Band.VIS, ref.geometry), make_obs(Band.LWIR, src.geometry), None, None
    )
    err = registration_error(est, oracle)
    assert err["epe_mean"] == pytest.approx(f * baseline / depth, rel=0.1)


def test_oracle_registrar_fixture_shapes(tmp_path):
    ds = ThreeBandDataset(build_fixture(tmp_path / "ds"))
    reg = OracleRegistrar(ds)
    sample = ds.sample(ds.sample_ids()[0])
    ref, src = sample.observations[Band.VIS], sample.observations[Band.LWIR]
    warp = reg.estimate(ref, src, None, None)
    assert warp.map_xy.shape == (ref.geometry.height, ref.geometry.width, 2)
    assert warp.valid.dtype == bool
    assert reg.estimate(ref, src, None, None) is not None  # second call uses cache
    assert len(reg._views) == 2


@pytest.mark.skipif(not DATASET, reason="QUANTILOOM_TEST_DATASET unset")
def test_oracle_registrar_vs_oracle_zero():
    ds = ThreeBandDataset(DATASET)
    sample = ds.sample(ds.sample_ids()[0])
    reg = OracleRegistrar(ds)
    warp = reg.estimate(
        sample.observations[Band.VIS], sample.observations[Band.LWIR], None, None
    )
    err = registration_error(warp, warp)
    assert err["epe_max"] == 0.0 and err["coverage"] == 1.0

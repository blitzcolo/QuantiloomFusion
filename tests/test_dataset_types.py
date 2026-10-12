"""Shape tests for the dataset contract types (no JSON parsing here)."""

import dataclasses

import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.dataset import (
    CameraGeometry,
    Condition,
    Observation,
    Product,
    Sample,
    SampleParameters,
    SignalSemantics,
    condition_of,
)

# Literal values copied from the real pilot LWIR observation
# (build/env-pilot-0.7.4/samples/concrete_a000_e00_r00_n00/sample.json).
IDENTITY16 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
GEO_KWARGS = dict(
    width=256,
    height=192,
    intrinsics=(250.088550210605, 0.0, 128.0, 0.0, 250.088550210605, 96.0, 0.0, 0.0, 1.0),
    distortion=(-0.055, 0.005, 0.0008, -0.0006, 0.0),
    camera_to_world=IDENTITY16,
    world_to_camera=IDENTITY16,
)


def make_observation(band: Band, mode: str) -> Observation:
    signal = SignalSemantics(
        kind="raw_dn",
        unit="dn",
        transfer="linear",
        cfa="0",
        channels=("Y",),
        colour_space="mono",
        storage="float32",
    )
    products = {
        "rawdn": Product("rawdn", "samples/s/lwir_rawdn.exr", signal),
        "truth_camera_z": Product("truth_camera_z", "samples/s/lwir_truth_camera_z.exr"),
    }
    return Observation(
        band=band,
        geometry=CameraGeometry(**GEO_KWARGS),
        spectral_mode=mode,
        products=products,
        record="samples/s/records/lwir.json",
    )


def make_sample(scene_id: str = "concrete") -> Sample:
    observations = {
        Band.LWIR: make_observation(Band.LWIR, "lwir_fused"),
        Band.VIS: make_observation(Band.VIS, "vis_hero"),
    }
    return Sample(
        dataset_id="env-pilot-0.7.4",
        sample_id="concrete_a000_e00_r00_n00",
        scene_id=scene_id,
        split_group="sg0",
        parameters=SampleParameters(azimuth_deg=0.0, elevation_deg=0.0, radius_scale=1.0, repeat=0),
        rig_to_world=IDENTITY16,
        observations=observations,
    )


def test_intrinsics_properties():
    g = CameraGeometry(**GEO_KWARGS)
    assert g.fx == pytest.approx(250.088550210605)
    assert g.fy == pytest.approx(250.088550210605)
    assert g.cx == 128.0
    assert g.cy == 96.0


def test_geometry_is_frozen():
    g = CameraGeometry(**GEO_KWARGS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.width = 512


@pytest.mark.parametrize(
    "field, value",
    [
        ("intrinsics", (0.0,) * 8),
        ("distortion", (0.0,) * 4),
        ("camera_to_world", (0.0,) * 15),
        ("world_to_camera", (0.0,) * 17),
        ("width", 0),
    ],
)
def test_geometry_bad_lengths_rejected(field, value):
    with pytest.raises(ValueError):
        CameraGeometry(**{**GEO_KWARGS, field: value})


def test_product_lookup_and_keyerror():
    obs = make_observation(Band.LWIR, "lwir_fused")
    assert obs.product("rawdn").path == "samples/s/lwir_rawdn.exr"
    assert obs.product("truth_camera_z").signal is None
    with pytest.raises(KeyError, match="rawdn"):
        obs.product("nope")


def test_condition_of_suffixes():
    assert condition_of("concrete") is Condition.DAY
    assert condition_of("concrete_night") is Condition.NIGHT
    assert condition_of("concrete_moonlight") is Condition.MOONLIGHT


def test_sample_condition_and_band_order():
    assert make_sample("concrete_night").condition is Condition.NIGHT
    sample = make_sample()
    # observations were inserted LWIR-first; bands must follow BANDS order.
    assert sample.bands == (Band.VIS, Band.LWIR)


def test_signal_semantics_is_linear():
    assert SignalSemantics("raw_dn", "dn", "linear", "0", ("Y",), "mono", "float32").is_linear
    assert not SignalSemantics("display", "dn", "srgb", "0", ("R", "G", "B"), "srgb", "uint8").is_linear

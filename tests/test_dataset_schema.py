"""Schema parsing: real dataset (env-gated), producer cross-check, negatives."""

import argparse
import copy
import json
import os
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from quantiloom_fusion.core import jsonio, locking, tomlio
from quantiloom_fusion.core.bands import BANDS, Band
from quantiloom_fusion.datagen import planner, runner, settings
from quantiloom_fusion.dataset import schema

from fake_renderer import FakeRenderer

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("QUANTILOOM_TEST_DATASET")


@pytest.mark.skipif(not DATASET, reason="set QUANTILOOM_TEST_DATASET for real dataset tests")
def test_real_dataset_parses():
    root = Path(DATASET).resolve()
    plan_doc = jsonio.load_json(root / "dataset.plan.json")
    plan = schema.parse_plan_summary(plan_doc)
    index = schema.parse_index(jsonio.load_json(root / "dataset.index.json"))
    assert index.dataset_id == plan.dataset_id
    assert len(index.entries) == index.complete_triplets
    truth = plan_doc["geometry_ground_truth"]
    for entry in index.entries:
        s = schema.parse_sample(jsonio.load_json(root / entry.manifest))
        assert s.dataset_id == plan.dataset_id and s.sample_id == entry.sample_id
        assert set(s.observations) == set(BANDS)
        for band, obs in s.observations.items():
            for name in ("measurement", "rawdn", "corrected", "display", "rendered_reference"):
                assert name in obs.products
            if band in (Band.MWIR, Band.LWIR):
                assert "tapp" in obs.products
            if truth:
                assert "truth_surface_temperature" in obs.products
                assert "truth_material_id" in obs.products


def _fake_dataset(root: Path) -> Path:
    """Plan + run a minimal dataset through FakeRenderer (same fixture as test_datagen_run)."""
    cli = root / "Quantiloom.exe"
    cli.write_bytes(b"fixture cli")
    (root / "assets").mkdir()
    (root / "src/shaders").mkdir(parents=True)
    model = root / "scene.glb"
    payload = json.dumps({"asset": {"version": "2.0"}, "materials": [{"name": "Object"}]}).encode()
    payload += b" " * ((-len(payload)) % 4)
    model.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(payload)) + struct.pack("<II", len(payload), 0x4E4F534A) + payload)
    (root / "solar.csv").write_text("nm,direct,global\n400,1,2\n4000,0.01,0.02\n")
    scene_toml = root / "scene.toml"
    scene_toml.write_text(tomlio.toml_text({"renderer": {}, "scene": {"gltf": model.as_posix(), "world_units_to_meters": 2.0},
        "camera": {"position": [0, 1, 6], "look_at": [0, 1, 0]}, "lighting": {"solar_lut": "solar.csv"},
        "materials": [{"name": "Object", "ir_emissivity": 0.9}]}))
    s = settings.read_settings(ROOT / "configs/env_dataset.toml")
    s["dataset"].update(cli=cli.as_posix(), qt_root=root.as_posix(), verification="hashes",
        thermal_mode="authored", ir_materials="constant", geometry_truth=False,
        daylight_solar_scale="none")
    s["sampling"].update(azimuths_deg=[0], elevations_deg=[5], radius_scales=[1.0], repeats=1)
    s["scenes"] = [{"id": "test", "config": scene_toml.as_posix()}]
    config = root / "experiment.toml"
    config.write_text(tomlio.toml_text(s))
    out = root / "dataset"
    fake = FakeRenderer()
    plan = planner.plan(config, out, renderer=fake)
    args = argparse.Namespace(scene=None, max_captures=None, keep_going=False)
    # run_dataset refuses non-drive output dirs on POSIX; the fake needs no
    # such path, so the check is off.
    with patch.object(runner, "POSIX", False), locking.runner_lock(out) as (lease, lock):
        assert runner.run_dataset(out, plan, args, lease, lock, renderer=fake) == 0
    return out


def test_producer_output_parses(tmp_path):
    out = _fake_dataset(tmp_path)
    plan = schema.parse_plan_summary(jsonio.load_json(out / "dataset.plan.json"))
    index = schema.parse_index(jsonio.load_json(out / "dataset.index.json"))
    assert index.dataset_id == plan.dataset_id and index.complete_triplets == 1
    s = schema.parse_sample(jsonio.load_json(out / index.entries[0].manifest))
    assert set(s.observations) == set(BANDS)
    lwir = s.observations[Band.LWIR]
    for name in ("measurement", "rawdn", "corrected", "display", "rendered_reference", "truth_camera_z"):
        assert name in lwir.products
    assert len(lwir.geometry.intrinsics) == 9
    assert lwir.products["rawdn"].signal.channels == ("Y",)


def _geometry_doc(w=8, h=6):
    return {
        "width": w, "height": h,
        "intrinsics": [1.0] * 9,
        "camera_to_world": [0.0] * 16,
        "world_to_camera": [0.0] * 16,
        "camera_axes": "right_down_forward",
        "distortion": {"model": "brown_conrady", "coefficient_order": "k1_k2_p1_p2_k3",
                       "coefficients": [0.0] * 5, "max_theta_radians": 1.0},
        "pixel_center_offset": [0.5, 0.5],
        "pixel_origin": "top_left",
        "world_units_to_meters": 1.0,
    }


CONFIG_TOML = (
    "[sensor]\ncfa = \"mono\"\n"
    "[sensor.exposure]\ntime_s = 0.03\n"
    "[sensor.readout]\nadc_bits = 14\nblack_level_dn = 0\nelectrons_per_dn = 1\n"
    "[isp]\nwhite_balance = [1, 1, 1]\n"
    "device_to_linear_srgb = [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
)


def _observation(w=8, h=6):
    sensor = {"path": "f.exr",
              "signal": {"kind": "k", "unit": "u", "transfer": "linear", "camera_cfa": "0",
                         "channels": ["Y"], "colour_space": "none", "storage": "float32"},
              "provenance": {"geometry": _geometry_doc(w, h), "camera_config_toml": CONFIG_TOML},
              "image_metadata": {}}
    prods = {k: copy.deepcopy(sensor) for k in ("measurement", "rawdn", "corrected", "display")}
    prods["rendered_reference"] = {"path": "frame.exr", "role": "r", "description": {}}
    return {"record": "r.json", "camera_to_world": [0.0] * 16, "spectral_mode": "m",
            "products": prods, "extra_products": {}, "pixel_qa": {}}


def _minimal_sample():
    return {"schema": "quantiloomfusion.sample", "schema_version": 1, "dataset_id": "d",
            "sample_id": "s0", "scene_id": "test", "split_group": "g", "state": "complete",
            "parameters": {"azimuth_deg": 0, "elevation_deg": 5, "radius_scale": 1.0, "repeat": 0},
            "rig_to_world": [0.0] * 16,
            "observations": {str(b): _observation() for b in BANDS}}


def test_product_name_mapping():
    assert schema.product_name("frame.exr") == "rendered_reference"
    assert schema.product_name("frame.png") == "preview"
    assert schema.product_name("frame_tapp.exr") == "tapp"
    assert schema.product_name("frame_truth_materials.json") == "truth_materials"
    for bad in ("frame_mystery.bin", "weird.exr", "frame_.exr"):
        with pytest.raises(ValueError):
            schema.product_name(bad)


def test_negative_cases():
    doc = _minimal_sample()
    doc["schema_version"] = 2
    with pytest.raises(ValueError, match="schema"):
        schema.parse_sample(doc)

    doc = _minimal_sample()
    doc["observations"]["vis"]["extra_products"] = {"samples/s0/vis/v0001/frame_mystery.bin": {}}
    with pytest.raises(ValueError, match="unknown product"):
        schema.parse_sample(doc)

    doc = _minimal_sample()
    doc["observations"]["vis"]["products"]["display"]["provenance"]["geometry"]["width"] = 99
    with pytest.raises(ValueError, match="geometry"):
        schema.parse_sample(doc)

    doc = _minimal_sample()
    doc["observations"]["vis"]["products"]["measurement"]["provenance"]["geometry"]["distortion"]["coefficient_order"] = "k3_k2"
    with pytest.raises(ValueError, match="distortion"):
        schema.parse_sample(doc)

    doc = _minimal_sample()
    doc["observations"]["vis"]["extra_products"] = {"samples/s0/vis/v0001/frame_measurement.exr": {}}
    with pytest.raises(ValueError, match="duplicate"):
        schema.parse_sample(doc)


def test_readout_parsed():
    s = schema.parse_sample(_minimal_sample())
    r = s.observations[Band.VIS].readout
    assert r.cfa == "mono" and r.adc_bits == 14
    assert r.exposure_s == pytest.approx(0.03) and r.electrons_per_dn == 1.0
    assert r.white_balance == (1.0, 1.0, 1.0)
    assert r.device_to_linear_srgb == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_readout_negative_cases():
    doc = _minimal_sample()
    del doc["observations"]["vis"]["products"]["rawdn"]["provenance"]["camera_config_toml"]
    with pytest.raises(ValueError, match="camera_config_toml"):
        schema.parse_sample(doc)

    doc = _minimal_sample()
    prov = doc["observations"]["vis"]["products"]["display"]["provenance"]
    prov["camera_config_toml"] = prov["camera_config_toml"].replace('cfa = "mono"', 'cfa = "rggb"')
    with pytest.raises(ValueError, match="camera_config_toml"):
        schema.parse_sample(doc)

    doc = _minimal_sample()
    for name in ("measurement", "rawdn", "corrected", "display"):
        doc["observations"]["vis"]["products"][name]["provenance"]["camera_config_toml"] = (
            "[sensor]\ncfa = \"mono\"\n"
        )
    with pytest.raises(ValueError, match="readout"):
        schema.parse_sample(doc)


@pytest.mark.skipif(not DATASET, reason="set QUANTILOOM_TEST_DATASET for real dataset tests")
def test_real_dataset_readout():
    root = Path(DATASET).resolve()
    index = schema.parse_index(jsonio.load_json(root / "dataset.index.json"))
    s = schema.parse_sample(jsonio.load_json(root / index.entries[0].manifest))
    vis = s.observations[Band.VIS].readout
    assert vis.cfa == "rggb" and vis.adc_bits == 12
    lwir = s.observations[Band.LWIR].readout
    assert lwir.cfa == "mono" and lwir.adc_bits == 14

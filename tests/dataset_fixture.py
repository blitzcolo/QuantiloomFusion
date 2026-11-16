"""Tiny consumer-side dataset fixture: writes real EXRs and contract JSONs.

Mirrors the published contract shape of build/env-pilot-0.7.4: sample.json,
dataset.index.json, dataset.plan.json, and per-band EXR products including
both uint32 id products that share the channel name "instance_id".
"""

import json
from pathlib import Path

import numpy as np

from quantiloom_fusion.core import jsonio, tomlio
from quantiloom_fusion.dataset import exr

IDENTITY16 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _geometry(w: int, h: int) -> dict:
    f = h / 2.0
    return {
        "camera_model": "brown_conrady",
        "camera_axes": "right_down_forward",
        "projection": "perspective",
        "width": w,
        "height": h,
        "intrinsics": [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0],
        "camera_to_world": list(IDENTITY16),
        "world_to_camera": list(IDENTITY16),
        "distortion": {"model": "brown_conrady", "coefficient_order": "k1_k2_p1_p2_k3",
                       "coefficients": [0.0] * 5, "max_theta_radians": 1.5},
        "pixel_center_offset": [0.5, 0.5],
        "pixel_origin": "top_left",
        "world_units_to_meters": 1.0,
    }


def _camera_config_toml(band: str, black_level_dn: float = 0.0) -> str:
    """camera_config_toml the renderer would echo for this band's sensor."""
    vis = band == "vis"
    return tomlio.toml_text({
        "sensor": {
            "cfa": "rggb" if vis else "mono",
            "exposure": {"time_s": 5e-5 if vis else 0.03},
            "readout": {"adc_bits": 12 if vis else 14, "black_level_dn": black_level_dn,
                        "electrons_per_dn": 5.0 if vis else 1.0},
        },
        "isp": {"white_balance": [1.0, 1.0, 1.0],
                "device_to_linear_srgb": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
    })


def _signal(kind: str, unit: str, channels: list, cfa: str = "0", transfer: str = "linear",
            colour_space: str = "device_native") -> dict:
    return {"kind": kind, "unit": unit, "transfer": transfer, "camera_cfa": cfa,
            "channels": channels, "colour_space": colour_space, "storage": "float32"}


def _write_observation(root: Path, sid: str, folder: str, band: str, size, black_level_dn: float = 0.0) -> dict:
    """Write one observation's products under samples/<sid>/<folder>/v0001; return its manifest entry."""
    w, h = size
    vdir = root / "samples" / sid / folder / "v0001"
    vdir.mkdir(parents=True, exist_ok=True)
    rel = f"samples/{sid}/{folder}/v0001"
    geom = _geometry(w, h)

    def write(name, channels, dtype=np.float32, fill=1.0):
        a = {c: np.full((h, w), fill, dtype=dtype) for c in channels}
        exr.write_image(vdir / name, a)

    products = {}
    cfa = "1" if band == "vis" else "0"
    signals = {
        "measurement": ("band_measurement", "e-/s", "linear", ["Measurement"], "device_native"),
        "rawdn": ("raw_dn", "DN", "linear", ["Raw"], "device_native"),
        "corrected": ("device_linear", "e-", "linear", ["Signal"], "device_native"),
        "display": ("display_srgb", "encoded sRGB", "sRGB", ["R", "G", "B"], "sRGB"),
    }
    config = _camera_config_toml(band, black_level_dn)
    for name, (kind, unit, transfer, chans, cs) in signals.items():
        fname = f"frame_{name}.exr"
        if name == "display":
            exr.write_image(vdir / fname, {c: np.full((h, w), v, np.float32)
                                           for c, v in zip(chans, (0.2, 0.4, 0.6))})
        elif name == "rawdn" and band == "vis":
            raw = np.zeros((h, w), np.float32)  # constant-colour RGGB mosaic
            raw[0::2, 0::2], raw[0::2, 1::2] = 400.0, 600.0
            raw[1::2, 0::2], raw[1::2, 1::2] = 600.0, 300.0
            exr.write_image(vdir / fname, {"Raw": raw + np.float32(black_level_dn)})
        elif name == "corrected":
            ramp = np.repeat(np.linspace(0.0, 100.0, w, dtype=np.float32)[None, :], h, 0)
            exr.write_image(vdir / fname, {chans[0]: ramp})
        else:
            exr.write_image(vdir / fname, {c: np.full((h, w), 1.0, np.float32)
                                           for c in chans})
        products[name] = {"path": f"{rel}/{fname}",
                          "signal": _signal(kind, unit, chans, cfa, transfer, cs),
                          "provenance": {"geometry": geom,
                                         "camera_config_toml": config},
                          "image_metadata": {}}
    write("frame.exr", ["R", "G", "B", "A"])
    products["rendered_reference"] = {"path": f"{rel}/frame.exr",
                                      "role": "separate_ideal_pinhole_render_without_sensor_degradation",
                                      "description": {"channel_names": ["R", "G", "B", "A"],
                                                      "channels": 4, "height": h, "width": w}}
    extras = {}

    def extra(fname, channels, dtype=np.float32, fill=1.0, **desc):
        write(fname, channels, dtype, fill)
        extras[f"{rel}/{fname}"] = {"channel_names": list(channels), "channels": len(channels),
                                    "height": h, "width": w, **desc}

    extra("frame_truth_camera_z.exr", ["camera_z_m"], fill=10.0)
    extra("frame_truth_validity.exr", ["validity"], fill=1.0)
    extra("frame_truth_world_position.exr", ["X", "Y", "Z"])
    extra("frame_truth_surface_temperature.exr", ["surface_temperature_K"], fill=300.0)
    extra("frame_truth_instance_id.exr", ["instance_id"], dtype=np.uint32, fill=1)
    extra("frame_truth_material_id.exr", ["instance_id"], dtype=np.uint32, fill=2)
    if band != "vis":
        extra("frame_tapp.exr", ["Channel_0"], fill=295.0,
              signal=_signal("apparent_temperature", "K", ["Channel_0"]))
    for fname, body in (
        ("frame_truth_instances.json", [{"instance_id": 1, "name": "obj"}]),
        ("frame_truth_materials.json", [{"material_id": 2, "name": "mat"}]),
    ):
        (vdir / fname).write_text(json.dumps(body), encoding="utf-8", newline="\n")
        extras[f"{rel}/{fname}"] = {"kind": fname.removeprefix("frame_").removesuffix(".json")}
    (vdir / "frame.png").write_bytes(b"PNG")
    extras[f"{rel}/frame.png"] = {"channel_names": ["R", "G", "B"], "channels": 3,
                                  "height": h, "width": w}
    (vdir / "frame.metadata.json").write_text(json.dumps({"record_id": "f" * 64}),
                                              encoding="utf-8", newline="\n")
    return {"record": f"{rel}/frame.metadata.json", "record_sha256": "0" * 64,
            "camera_to_world": list(IDENTITY16),
            "spectral_mode": f"{band}_fused",
            "products": products,
            "extra_products": extras,
            "pixel_qa": {}}


def build_fixture(root: Path, samples: int = 2, size=(8, 6), groups=None, references: bool = False,
                  vis_black_level_dn: float = 0.0) -> Path:
    """Write a tiny dataset under root and return its path.

    groups, if given, cycles sample split_groups (e.g. ["g0", "g1"] puts even
    samples in g0, odd in g1); default is one shared "fixture-group".
    references=False writes the v1 sample contract (no references, as older
    datasets have); True writes v2 with all three co-view references on the
    VIS geometry and size, flagged in the index. vis_black_level_dn puts a
    pedestal under the VIS RAW mosaic and records it in camera_config_toml,
    as the renderer does (64 DN for generic_cmos since Quantiloom 0.10.1).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset_id = "fixture-dataset-id"
    sizes = {"vis": size, "mwir": (size[0] - 2, size[1] - 1), "lwir": (size[0] - 3, size[1] - 2)}
    entries = []
    for i in range(samples):
        sid = f"fixture_s{i:02d}"
        group = groups[i % len(groups)] if groups else "fixture-group"
        observations = {band: _write_observation(root, sid, band, band, wh,
                                                 vis_black_level_dn if band == "vis" else 0.0)
                        for band, wh in sizes.items()}
        manifest = {"schema": "quantiloomfusion.sample", "schema_version": 2 if references else 1,
                    "dataset_id": dataset_id, "sample_id": sid, "scene_id": "fixture",
                    "split_group": group, "state": "complete",
                    "parameters": {"azimuth_deg": 0, "elevation_deg": 5,
                                   "radius_scale": 1.0, "repeat": i},
                    "rig_to_world": IDENTITY16,
                    "registration": "unregistered", "geometry_ground_truth": True,
                    "observations": observations}
        entry = {"sample_id": sid, "manifest": f"samples/{sid}/sample.json", "split_group": group}
        if references:
            manifest["references"] = {
                band: {**_write_observation(root, sid, f"ref_{band}", band, size,
                                            vis_black_level_dn if band == "vis" else 0.0),
                       "geometry_from": "vis", "psf_sigma_px": 0.45}
                for band in sizes}
            entry["references"] = True
        sdir = root / "samples" / sid
        jsonio.atomic_json(sdir / "sample.json", manifest)
        entries.append(entry)
    jsonio.atomic_json(root / "dataset.index.json",
                       {"schema": "quantiloomfusion.env_dataset", "schema_version": 1,
                        "dataset_id": dataset_id, "complete_triplets": len(entries),
                        "planned_triplets": len(entries), "samples": entries})
    jsonio.atomic_json(root / "dataset.plan.json",
                       {"schema": "quantiloomfusion.env_dataset", "schema_version": 1,
                        "dataset_id": dataset_id, "name": "fixture",
                        "recipe": {"cli_version": "Quantiloom 0.7.4"},
                        "scenes": [{"id": "fixture", "split_group": "fixture-group"}],
                        "geometry_ground_truth": True})
    return root

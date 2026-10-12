"""Strict parsing of the published dataset contract into dataset.types.

Standard library only. Every accessor raises ValueError naming the offending
key path, and unknown product filenames fail deliberately: a new renderer
product is added to the contract on purpose, not by accident.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from ..core.bands import BANDS, Band
from .types import (
    CameraGeometry,
    Observation,
    Product,
    Readout,
    Sample,
    SampleParameters,
    SignalSemantics,
)

SCHEMA_DATASET = "quantiloomfusion.env_dataset"
SCHEMA_SAMPLE = "quantiloomfusion.sample"

_SENSOR_PRODUCTS = ("measurement", "rawdn", "corrected", "display")
_PRODUCTS = _SENSOR_PRODUCTS + ("rendered_reference",)


def _req(doc: dict, key: str, where: str) -> Any:
    """doc[key] or a ValueError naming the missing key path."""
    if key not in doc:
        raise ValueError(f"{where}: missing key {key!r}")
    return doc[key]


def _check_schema(doc: dict, name: str, where: str) -> None:
    if _req(doc, "schema", where) != name or _req(doc, "schema_version", where) != 1:
        raise ValueError(f"{where}: unsupported schema {doc.get('schema')!r} v{doc.get('schema_version')!r}")


@dataclass(frozen=True)
class IndexEntry:
    """One row of dataset.index.json's samples list."""

    sample_id: str
    manifest: str  # dataset-relative path of the sample.json
    split_group: str


@dataclass(frozen=True)
class DatasetIndex:
    """Parsed dataset.index.json: completion accounting and manifest list."""

    dataset_id: str
    complete_triplets: int
    planned_triplets: int
    entries: tuple[IndexEntry, ...]


@dataclass(frozen=True)
class PlanSummary:
    """The subset of dataset.plan.json the consumer side needs."""

    dataset_id: str
    name: str
    cli_version: str
    scenes: tuple[tuple[str, str], ...]  # (scene id, split_group)


def parse_index(doc: dict) -> DatasetIndex:
    """Parse dataset.index.json; entries must number complete_triplets."""
    _check_schema(doc, SCHEMA_DATASET, "index")
    entries = tuple(
        IndexEntry(
            _req(e, "sample_id", f"samples[{i}]"),
            _req(e, "manifest", f"samples[{i}]"),
            _req(e, "split_group", f"samples[{i}]"),
        )
        for i, e in enumerate(_req(doc, "samples", "index"))
    )
    complete = _req(doc, "complete_triplets", "index")
    if len(entries) != complete:
        raise ValueError(f"index: {len(entries)} entries but complete_triplets={complete}")
    return DatasetIndex(
        _req(doc, "dataset_id", "index"), complete, _req(doc, "planned_triplets", "index"), entries
    )


def parse_plan_summary(doc: dict) -> PlanSummary:
    """Parse the consumer subset of dataset.plan.json."""
    _check_schema(doc, SCHEMA_DATASET, "plan")
    recipe = _req(doc, "recipe", "plan")
    scenes = tuple(
        (_req(s, "id", f"scenes[{i}]"), _req(s, "split_group", f"scenes[{i}]"))
        for i, s in enumerate(_req(doc, "scenes", "plan"))
    )
    return PlanSummary(
        _req(doc, "dataset_id", "plan"),
        _req(doc, "name", "plan"),
        _req(recipe, "cli_version", "recipe"),
        scenes,
    )


def product_name(filename: str) -> str:
    """Manifest product key for a product filename (strict mapping).

    frame.exr -> rendered_reference, frame.png -> preview,
    frame_<x>.exr/json -> <x>; anything else is an error.
    """
    if filename == "frame.exr":
        return "rendered_reference"
    if filename == "frame.png":
        return "preview"
    if filename.startswith("frame_") and filename.endswith((".exr", ".json")):
        name = PurePosixPath(filename).stem.removeprefix("frame_")
        if name:
            return name
    raise ValueError(f"unknown product filename: {filename!r}")


def _signal(doc: dict, where: str) -> SignalSemantics:
    return SignalSemantics(
        kind=_req(doc, "kind", where),
        unit=_req(doc, "unit", where),
        transfer=_req(doc, "transfer", where),
        cfa=str(_req(doc, "camera_cfa", where)),
        channels=tuple(_req(doc, "channels", where)),
        colour_space=_req(doc, "colour_space", where),
        storage=_req(doc, "storage", where),
    )


def _geometry(doc: dict, where: str) -> CameraGeometry:
    dist = _req(doc, "distortion", where)
    order = _req(dist, "coefficient_order", f"{where}.distortion")
    model = _req(dist, "model", f"{where}.distortion")
    if order != "k1_k2_p1_p2_k3" or model != "brown_conrady":
        raise ValueError(f"{where}.distortion: unsupported model {model!r} order {order!r}")
    return CameraGeometry(
        width=_req(doc, "width", where),
        height=_req(doc, "height", where),
        intrinsics=tuple(_req(doc, "intrinsics", where)),
        distortion=tuple(_req(dist, "coefficients", f"{where}.distortion")),
        camera_to_world=tuple(_req(doc, "camera_to_world", where)),
        world_to_camera=tuple(_req(doc, "world_to_camera", where)),
        max_theta_radians=dist.get("max_theta_radians"),
        camera_axes=_req(doc, "camera_axes", where),
        pixel_center_offset=tuple(_req(doc, "pixel_center_offset", where)),
        pixel_origin=_req(doc, "pixel_origin", where),
        world_units_to_meters=_req(doc, "world_units_to_meters", where),
    )


def _readout(toml: str, where: str) -> Readout:
    """Parse sensor readout fields out of a product's camera_config_toml."""
    try:
        cfg = tomllib.loads(toml)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{where}: invalid TOML: {e}") from None
    sensor = _req(cfg, "sensor", where)
    readout = _req(sensor, "readout", f"{where}.sensor")
    exposure = _req(sensor, "exposure", f"{where}.sensor")
    isp = _req(cfg, "isp", where)
    return Readout(
        cfa=_req(sensor, "cfa", f"{where}.sensor"),
        adc_bits=int(_req(readout, "adc_bits", f"{where}.sensor.readout")),
        black_level_dn=float(_req(readout, "black_level_dn", f"{where}.sensor.readout")),
        electrons_per_dn=float(_req(readout, "electrons_per_dn", f"{where}.sensor.readout")),
        exposure_s=float(_req(exposure, "time_s", f"{where}.sensor.exposure")),
        white_balance=tuple(_req(isp, "white_balance", f"{where}.isp")),
        device_to_linear_srgb=tuple(_req(isp, "device_to_linear_srgb", f"{where}.isp")),
    )


def _observation(band: Band, doc: dict, where: str) -> Observation:
    prods = _req(doc, "products", where)
    pw = f"{where}.products"
    g = _req(_req(_req(prods, "measurement", pw), "provenance", f"{pw}.measurement"), "geometry", f"{pw}.measurement.provenance")
    geometry = _geometry(g, f"{pw}.measurement.provenance.geometry")
    config = None
    for name in _SENSOR_PRODUCTS:
        prov = _req(_req(prods, name, pw), "provenance", f"{pw}.{name}")
        if _req(prov, "geometry", f"{pw}.{name}.provenance") != g:
            raise ValueError(f"{pw}.{name}.provenance.geometry differs from measurement's")
        cfg = _req(prov, "camera_config_toml", f"{pw}.{name}.provenance")
        if config is None:
            config = cfg
        elif cfg != config:
            raise ValueError(f"{pw}.{name}.provenance.camera_config_toml differs from measurement's")
    readout = _readout(config, f"{pw}.measurement.provenance.camera_config_toml")
    products: dict[str, Product] = {}

    def add(p: Product) -> None:
        if p.name in products:
            raise ValueError(f"{where}: duplicate product {p.name!r}")
        products[p.name] = p

    for name in _PRODUCTS:
        e = _req(prods, name, pw)
        ew = f"{pw}.{name}"
        sig = _signal(e["signal"], ew) if "signal" in e else None
        if sig:
            channels = tuple(sig.channels)
        else:
            channels = tuple(e.get("description", {}).get("channel_names", ()))
        add(Product(name, _req(e, "path", ew), sig, channels))
    for path, entry in _req(doc, "extra_products", where).items():
        name = product_name(PurePosixPath(path).name)
        sig = _signal(entry["signal"], f"{where}.extra_products[{name!r}]") if "signal" in entry else None
        add(Product(name, path, sig, tuple(entry.get("channel_names", ()))))
    return Observation(
        band=band,
        geometry=geometry,
        readout=readout,
        spectral_mode=_req(doc, "spectral_mode", where),
        products=products,
        record=_req(doc, "record", where),
    )


def parse_sample(doc: dict) -> Sample:
    """Parse one sample.json document into a typed Sample."""
    _check_schema(doc, SCHEMA_SAMPLE, "sample")
    if _req(doc, "state", "sample") != "complete":
        raise ValueError("sample.state: only 'complete' observations are published")
    obs = _req(doc, "observations", "sample")
    if set(obs) != {str(b) for b in BANDS}:
        raise ValueError(f"observations: expected bands {[str(b) for b in BANDS]}, got {sorted(obs)}")
    observations = {Band(k): _observation(Band(k), v, f"observations.{k}") for k, v in obs.items()}
    p = _req(doc, "parameters", "sample")
    return Sample(
        dataset_id=_req(doc, "dataset_id", "sample"),
        sample_id=_req(doc, "sample_id", "sample"),
        scene_id=_req(doc, "scene_id", "sample"),
        split_group=_req(doc, "split_group", "sample"),
        parameters=SampleParameters(
            azimuth_deg=_req(p, "azimuth_deg", "parameters"),
            elevation_deg=_req(p, "elevation_deg", "parameters"),
            radius_scale=_req(p, "radius_scale", "parameters"),
            repeat=_req(p, "repeat", "parameters"),
        ),
        rig_to_world=tuple(_req(doc, "rig_to_world", "sample")),
        observations=observations,
    )

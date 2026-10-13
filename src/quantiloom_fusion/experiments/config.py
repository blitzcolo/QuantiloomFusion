"""Experiment configuration: TOML parsed into frozen dataclasses, strict.

Unknown keys and invalid values raise ValueError naming the key.  The TOML
shape is::

    name = "pilot-baselines"
    dataset = "build/env-pilot-0.7.4"
    reference = "vis"
    band_sets = [["vis"], ["vis","lwir"], ...]
    vis_input = ["display", "raw_percentile"]
    ir_input = ["linear"]
    registration = ["oracle", "infinity_homography", "resize"]
    outputs = ["grey", "colour"]
    metrics = ["EN","SD","SF","AG","SCD","Qabf","MS_SSIM"]
    [[fusion]]
    name = "laplacian_pyramid"
    levels = 4          # extra keys are passed to fusion.create as config
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..core.bands import BANDS, Band
from ..fusion.registry import available as fusion_available

VIS_INPUTS = ("display", "raw_adc", "raw_percentile")
IR_INPUTS = ("linear", "temperature")
REGISTRATIONS = ("oracle", "infinity_homography", "resize")
OUTPUTS = ("grey", "colour")
METRIC_NAMES = ("EN", "SD", "SF", "AG", "SCD", "Qabf", "MS_SSIM")

_TOP_LEVEL = {
    "name", "dataset", "reference", "band_sets", "vis_input", "ir_input",
    "registration", "outputs", "metrics", "fusion",
}


@dataclass(frozen=True)
class FusionSpec:
    """One [[fusion]] table: a registry name plus constructor kwargs."""

    name: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str          # path as written; resolved via core.paths.native_path
    reference: Band
    band_sets: tuple[tuple[Band, ...], ...]
    vis_input: tuple[str, ...]
    ir_input: tuple[str, ...]
    registration: tuple[str, ...]
    outputs: tuple[str, ...]
    metrics: tuple[str, ...]
    fusion: tuple[FusionSpec, ...]


def _bad(key: str, msg: str) -> ValueError:
    return ValueError(f"{key}: {msg}")


def _strings(data: dict, key: str, allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise _bad(key, f"expected a non-empty list of strings, got {value!r}")
    if allowed is not None:
        for v in value:
            if v not in allowed:
                raise _bad(key, f"{v!r} not one of {list(allowed)}")
    return tuple(value)


def _band(name: str, key: str) -> Band:
    try:
        return Band(name)
    except ValueError:
        raise _bad(key, f"{name!r} not a band; known: {[str(b) for b in BANDS]}") from None


def load_config(path: Path) -> ExperimentConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for key in data:
        if key not in _TOP_LEVEL:
            raise _bad(key, "unknown config key")

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise _bad("name", f"expected a non-empty string, got {name!r}")
    dataset = data.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise _bad("dataset", f"expected a non-empty string, got {dataset!r}")
    reference = _band(data.get("reference", ""), "reference")

    raw_sets = data.get("band_sets")
    if not isinstance(raw_sets, list) or not raw_sets:
        raise _bad("band_sets", "expected a non-empty list of band lists")
    band_sets = []
    for i, s in enumerate(raw_sets):
        if not isinstance(s, list) or not s or not all(isinstance(b, str) for b in s):
            raise _bad(f"band_sets[{i}]", f"expected a non-empty list of band names, got {s!r}")
        bands = tuple(_band(b, f"band_sets[{i}]") for b in s)
        if len(set(bands)) != len(bands):
            raise _bad(f"band_sets[{i}]", f"duplicate band in {s!r}")
        if reference not in bands:
            raise _bad(f"band_sets[{i}]", f"does not contain the reference band {reference!r}")
        band_sets.append(bands)

    fusion_raw = data.get("fusion")
    if not isinstance(fusion_raw, list) or not fusion_raw:
        raise _bad("fusion", "expected a non-empty list of [[fusion]] tables")
    fusion = []
    for i, f in enumerate(fusion_raw):
        if not isinstance(f, dict) or not isinstance(f.get("name"), str):
            raise _bad(f"fusion[{i}]", f"expected a table with a string 'name', got {f!r}")
        if f["name"] not in fusion_available():
            raise _bad(f"fusion[{i}].name",
                       f"{f['name']!r} not a registered method; known: {fusion_available()}")
        fusion.append(FusionSpec(f["name"], {k: v for k, v in f.items() if k != "name"}))

    return ExperimentConfig(
        name=name,
        dataset=dataset,
        reference=reference,
        band_sets=tuple(band_sets),
        vis_input=_strings(data, "vis_input", VIS_INPUTS),
        ir_input=_strings(data, "ir_input", IR_INPUTS),
        registration=_strings(data, "registration", REGISTRATIONS),
        outputs=_strings(data, "outputs", OUTPUTS),
        metrics=_strings(data, "metrics", METRIC_NAMES),
        fusion=tuple(fusion),
    )

"""Variant expansion: the experiment's cross product, collapsed where it
degenerates, each variant carrying a provenance spec and fingerprint.

The fingerprint is the first 16 hex of the sha256 of the variant spec's
canonical JSON (sorted keys, compact separators).  Everything that affects
results is in the spec — band set, inputs with normalizer settings,
registrar, fusion method with its params, output kind — so the same spec
hashes identically regardless of config ordering.  The experiment name and
the metric list are deliberately not in the fingerprint: they describe the
run, not the method.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..core.bands import BANDS, Band
from ..signal.normalize import (
    IrLinear,
    IrTemperature,
    Normalizer,
    VisDisplay,
    VisRawLinear,
)
from .config import ExperimentConfig, FusionSpec

IR_BANDS = (Band.MWIR, Band.LWIR)


def vis_normalizer(kind: str) -> Normalizer:
    if kind == "display":
        return VisDisplay()
    if kind == "raw_adc":
        return VisRawLinear(scale="adc")
    if kind == "raw_percentile":
        return VisRawLinear(scale="percentile")
    raise ValueError(f"unknown vis_input {kind!r}")


def ir_normalizer(kind: str) -> Normalizer:
    if kind == "linear":
        return IrLinear()
    if kind == "temperature":
        return IrTemperature()
    raise ValueError(f"unknown ir_input {kind!r}")


def fingerprint(spec: dict) -> str:
    """First 16 hex of sha256 of the spec's canonical JSON."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"),
                           allow_nan=False, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]


@dataclass(frozen=True)
class Variant:
    """One fusion-method instance to score per sample."""

    label: str
    spec: dict
    band_set: tuple[Band, ...]
    vis_input: str            # name, or the only input when vis is absent
    ir_input: str             # name or "none"
    registration: str         # registrar name or "none"
    fusion_name: str
    fusion_params: dict
    output: str               # "grey" | "colour"

    @property
    def method_fingerprint(self) -> str:
        return fingerprint(self.spec)


def _spec(band_set, reference, vis_input, vis_params, ir_input, ir_params,
          registration, fusion_name, fusion_params, output) -> dict:
    return {
        "band_set": [str(b) for b in band_set],
        "reference": str(reference),
        "vis_input": vis_input,
        "vis_params": vis_params,
        "ir_input": ir_input,
        "ir_params": ir_params,
        "registration": registration,
        "fusion": fusion_name,
        "fusion_params": dict(fusion_params),
        "output": output,
    }


def _label(band_set, vis_input, ir_input, registration, fusion_name, output) -> str:
    bands = "+".join(str(b) for b in band_set)
    return (f"{bands}|vis={vis_input}|ir={ir_input}|reg={registration}"
            f"|{fusion_name}|{output}")


def expand(cfg: ExperimentConfig) -> list[Variant]:
    """band_set x vis_input x ir_input x registration x fusion x output.

    A single-band set (only the reference) collapses registration to "none"
    and fusion to "reference_only" — one variant, not a duplicate per
    registrar/method.  ir_input collapses to "none" when the set has no IR
    band.  Variants carry pre-call normalizer params (settings only — no
    per-image applied values), so specs are stable across runs.
    """
    variants: list[Variant] = []
    for band_set in cfg.band_sets:
        ordered = tuple(b for b in BANDS if b in band_set)
        single = len(ordered) == 1
        registrations = ("none",) if single else cfg.registration
        fusions = (FusionSpec("reference_only", {}),) if single else cfg.fusion
        ir_inputs = ("none",) if not any(b in IR_BANDS for b in ordered) else cfg.ir_input
        for vis_in in cfg.vis_input:
            vis_params = vis_normalizer(vis_in).params()
            for ir_in in ir_inputs:
                ir_params = None if ir_in == "none" else ir_normalizer(ir_in).params()
                for reg in registrations:
                    for f in fusions:
                        for out in cfg.outputs:
                            spec = _spec(ordered, cfg.reference, vis_in, vis_params,
                                         ir_in, ir_params, reg, f.name, f.params, out)
                            variants.append(Variant(
                                label=_label(ordered, vis_in, ir_in, reg, f.name, out),
                                spec=spec,
                                band_set=ordered,
                                vis_input=vis_in,
                                ir_input=ir_in,
                                registration=reg,
                                fusion_name=f.name,
                                fusion_params=dict(f.params),
                                output=out,
                            ))
    return variants

"""Evaluation scenes: normalised luminance inputs, truth geometry, layers.

Normalisation is chosen explicitly through :class:`InputSpec` and recorded in
every result row.  VIS is either the display image or linear RGB from RAW
(black level, demosaic, white balance); IR is either the corrected linear
signal or the apparent-temperature window.  The IR display image is rejected
(its stretch is scene dependent).  Fusion runs on the luminance channel.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import numpy as np

from ..core.bands import BANDS, Band
from ..dataset.reader import ThreeBandDataset
from ..dataset.types import CameraGeometry
from ..geometry.layers import layer_map
from ..geometry.warp import DepthView, Warp, load_depth_view, oracle_warp
from ..signal.colour import luminance
from ..signal.normalize import Normalizer, normalized_input
from .layer_report import combine_layers

VIS_PRODUCTS = ("display", "rawdn")
IR_PRODUCTS = ("corrected", "tapp")


@dataclass(frozen=True)
class InputSpec:
    """The explicit normalisation choice: one normaliser for VIS, one for IR."""

    vis: Normalizer
    ir: Normalizer

    def __post_init__(self) -> None:
        if self.vis.product not in VIS_PRODUCTS:
            raise ValueError(f"VIS normaliser must consume one of {VIS_PRODUCTS}, got {self.vis.product!r}")
        if self.ir.product not in IR_PRODUCTS:
            raise ValueError(f"IR normaliser must consume one of {IR_PRODUCTS} (never the display "
                             f"image), got {self.ir.product!r}")

    def for_band(self, band: Band) -> Normalizer:
        return copy.copy(self.vis if band == Band.VIS else self.ir)

    @property
    def scene_independent(self) -> bool:
        """True when no normaliser fits its window to the image it normalises."""
        return all(getattr(n, "mode", None) != "percentile" and getattr(n, "scale", None) != "percentile"
                   for n in (self.vis, self.ir))


@dataclass(frozen=True)
class EvalScene:
    """One sample ready for evaluation, all images luminance on native grids."""

    sample_id: str
    reference: Band
    images: dict[Band, np.ndarray]
    geometry: dict[Band, CameraGeometry]
    views: dict[Band, DepthView]
    oracle_warps: dict[Band, Warp]
    layers: np.ndarray
    norm: dict = field(default_factory=dict)
    clean: dict[Band, np.ndarray] | None = None

    def norm_columns(self) -> dict[str, str]:
        return {"norm_vis": self.norm["vis"], "norm_ir": self.norm["ir"],
                "norm_params": json.dumps(self.norm["params"], sort_keys=True, default=str)}


def _luminance(image: np.ndarray) -> np.ndarray:
    return luminance(image).astype(np.float32) if image.ndim == 3 else image


def build_scene(sample_id: str, images: dict[Band, np.ndarray], views: dict[Band, DepthView],
                norm: dict, reference: Band = Band.VIS) -> EvalScene:
    """Oracle warps and the combined layer map from images and truth views."""
    sources = [b for b in BANDS if b in images and b != reference]
    warps = {b: oracle_warp(views[reference], views[b], ref_band=reference, src_band=b) for b in sources}
    layers = combine_layers([layer_map(views[reference], views[b]) for b in sources])
    return EvalScene(sample_id, reference, images, {b: v.geometry for b, v in views.items()},
                     views, warps, layers, norm)


def load_scene(dataset: ThreeBandDataset, sample_id: str, spec: InputSpec,
               bands: tuple[Band, ...] = BANDS) -> EvalScene:
    """Load a sample, its truth geometry and (when present) its co-view clean images."""
    sample = dataset.sample(sample_id)
    images, params, views = {}, {}, {}
    for band in bands:
        obs = sample.observations[band]
        norm = spec.for_band(band)
        images[band] = _luminance(normalized_input(dataset, obs, norm))
        params[band.value] = norm.params()
        views[band] = load_depth_view(dataset, obs)
    record = {"vis": spec.vis.name, "ir": spec.ir.name, "params": params,
              "scene_independent": spec.scene_independent}
    scene = build_scene(sample_id, images, views, record)
    if not sample.references:
        return scene
    clean = {b: _luminance(normalized_input(dataset, sample.references[b], spec.for_band(b)))
             for b in bands}
    return EvalScene(**{**scene.__dict__, "clean": clean})

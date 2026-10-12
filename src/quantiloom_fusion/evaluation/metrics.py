"""Metric wrappers: the fusion metrics behind the :class:`Metric` interface.

Each wrapper scores a :class:`FusionOutput` against the aligned sources of
its :class:`FusionInput`.  The fused image is taken as its luminance: a
greyscale output is used directly, an RGB output is reduced to its BT.601 Y.

The evaluation region is the intersection of every band's support — the
pixels where all bands genuinely exist on the reference grid.  The
source-referenced metrics additionally receive each band's own support, so a
band is only compared where its footprint reaches; see the ``*_n``
generalisations in :mod:`.source_ref` for what that means per metric.
"""

from __future__ import annotations

import numpy as np

from ..core.bands import BANDS
from ..fusion.align import aligned_sources
from ..fusion.base import FusionInput, FusionOutput
from ..dataset.types import Sample
from ..signal.colour import luminance
from .base import Metric
from .no_reference import (
    average_gradient,
    entropy,
    spatial_frequency,
    standard_deviation,
)
from .source_ref import ms_ssim_fusion_n, qabf_n, scd_n


def _fused_y(image: np.ndarray) -> np.ndarray:
    """The fused image's luminance: itself when greyscale, Y when RGB."""
    a = np.asarray(image, dtype=np.float64)
    return luminance(a) if a.ndim == 3 else a


def _eval_mask(supports) -> np.ndarray | None:
    """Intersection of every band's support, as a float mask (None if all-True)."""
    region = None
    for support in supports.values() if isinstance(supports, dict) else supports:
        inside = np.asarray(support) > 0.5
        region = inside if region is None else region & inside
    if region is None or region.all():
        return None
    return region.astype(np.float64)


def _sources(x: FusionInput):
    """Aligned sources and their supports, ordered by BANDS."""
    images, supports = aligned_sources(x)
    bands = [b for b in BANDS if b in images]
    return [images[b] for b in bands], [supports[b] for b in bands]


def _grey_sources(images: list[np.ndarray]) -> list[np.ndarray]:
    return [luminance(np.asarray(i, np.float64)) if i.ndim == 3 else np.asarray(i, np.float64)
            for i in images]


class Entropy(Metric):
    """EN: histogram entropy of the fused luminance, on the 0-255 scale."""

    name = "EN"
    version = 1

    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]:
        _, supports = aligned_sources(x)
        return {self.name: entropy(_fused_y(out.image), mask=_eval_mask(supports))}


class StandardDeviation(Metric):
    """SD: standard deviation of the fused luminance, on the 0-255 scale."""

    name = "SD"
    version = 1

    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]:
        _, supports = aligned_sources(x)
        return {self.name: standard_deviation(_fused_y(out.image), mask=_eval_mask(supports))}


class SpatialFrequency(Metric):
    """SF: spatial frequency of the fused luminance, on the 0-255 scale."""

    name = "SF"
    version = 1

    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]:
        _, supports = aligned_sources(x)
        return {self.name: spatial_frequency(_fused_y(out.image), mask=_eval_mask(supports))}


class AverageGradient(Metric):
    """AG: average gradient of the fused luminance, on the 0-255 scale."""

    name = "AG"
    version = 1

    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]:
        _, supports = aligned_sources(x)
        return {self.name: average_gradient(_fused_y(out.image), mask=_eval_mask(supports))}


class SCD(Metric):
    """SCD: sum of correlations of differences, over the N aligned sources."""

    name = "SCD"
    version = 1

    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]:
        images, supports = _sources(x)
        return {self.name: scd_n(
            _fused_y(out.image), _grey_sources(images),
            mask=_eval_mask(supports), supports=list(supports))}


class Qabf(Metric):
    """Qabf: edge-preservation measure, weighted over the N aligned sources."""

    name = "Qabf"
    version = 1

    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]:
        images, supports = _sources(x)
        return {self.name: qabf_n(
            _fused_y(out.image), _grey_sources(images),
            mask=_eval_mask(supports), supports=list(supports))}


class MsSsim(Metric):
    """MS_SSIM: mean multi-scale SSIM of the fused image against each source."""

    name = "MS_SSIM"
    version = 1

    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]:
        images, supports = _sources(x)
        return {self.name: ms_ssim_fusion_n(
            _fused_y(out.image), _grey_sources(images),
            mask=_eval_mask(supports), supports=list(supports))}

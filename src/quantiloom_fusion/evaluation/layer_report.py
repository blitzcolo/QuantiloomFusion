"""Layer-wise reporting of one fused image.

Layers come from ``geometry.layers`` (invalid / out_of_view / occluded / edge /
correspondable).  With several sources the per-source maps are combined by the
worst layer (lowest code), so a pixel is "correspondable" only when every
source corresponds there.  Reported groups:

* ``correspondable`` and ``edge``: context (no-reference) metrics and the
  source-referenced metrics, each band counted only inside its own support.
* ``occluded_oov`` (occluded + out_of_view): no source has information, so the
  correct behaviour is to fall back to the reference band.  Source metrics are
  not defined there; the deviation from the reference band is the score.

No-reference metrics carry the ``ctx_`` prefix: they are context only and are
never a dependent variable of registration sensitivity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..core.bands import BANDS, Band
from ..geometry.layers import Layer
from .full_ref import psnr
from .no_reference import average_gradient, entropy, spatial_frequency, standard_deviation
from .source_ref import ms_ssim_fusion_n, qabf_n, scd_n

LAYER_GROUPS: dict[str, tuple[Layer, ...]] = {
    "correspondable": (Layer.CORRESPONDABLE,),
    "edge": (Layer.EDGE,),
    "occluded_oov": (Layer.OCCLUDED, Layer.OUT_OF_VIEW),
}
SOURCE_METRIC_KEYS = ("SCD", "Qabf", "MS_SSIM")
MS_SSIM_MARGIN_PX = 32
_NAN = float("nan")


@dataclass(frozen=True)
class FusedView:
    """A fused luminance image with the aligned sources it was fused from."""

    fused: np.ndarray
    sources: dict[Band, np.ndarray]
    supports: dict[Band, np.ndarray]
    reference: Band


def combine_layers(maps: Sequence[np.ndarray]) -> np.ndarray:
    """Worst layer over the per-source maps (INVALID < OUT_OF_VIEW < OCCLUDED < EDGE < CORRESPONDABLE)."""
    return np.minimum.reduce(list(maps))


def layer_masks(layers: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.isin(layers, [int(layer) for layer in members])
            for name, members in LAYER_GROUPS.items()}


def context_metrics(view: FusedView, mask: np.ndarray) -> dict[str, float]:
    m = mask.astype(np.float64)
    return {"ctx_EN": entropy(view.fused, mask=m), "ctx_SD": standard_deviation(view.fused, mask=m),
            "ctx_SF": spatial_frequency(view.fused, mask=m), "ctx_AG": average_gradient(view.fused, mask=m)}


def source_ref_metrics(view: FusedView, mask: np.ndarray) -> dict[str, float]:
    """SCD / Qabf / MS-SSIM on ``mask`` with each band restricted to its own support.

    ``ms_ssim_fusion_n`` only discards the finest-scale window around a footprint edge, so fill
    outside the footprint still leaks through the coarser scales (about 5e-5 on a 0.8 score,
    zero beyond 32 px).  Here every non-trivial support is eroded by ``MS_SSIM_MARGIN_PX``
    for the MS-SSIM term only; SCD and Qabf already confine each band exactly.
    """
    if not mask.any():
        return dict.fromkeys(SOURCE_METRIC_KEYS, _NAN)
    bands = [b for b in BANDS if b in view.sources]
    srcs = [np.asarray(view.sources[b], np.float64) for b in bands]
    sups = [view.supports[b] for b in bands]
    m = mask.astype(np.float64)
    eroded = [s if s.all() else ndimage.binary_erosion(s, iterations=MS_SSIM_MARGIN_PX, border_value=1)
              for s in sups]
    return {"SCD": scd_n(view.fused, srcs, mask=m, supports=sups),
            "Qabf": qabf_n(view.fused, srcs, mask=m, supports=sups),
            "MS_SSIM": ms_ssim_fusion_n(view.fused, srcs, mask=m, supports=eroded)}


def reference_deviation(view: FusedView, mask: np.ndarray) -> dict[str, float]:
    """Deviation of the fused image from the reference band on ``mask``."""
    if not mask.any():
        return {"dev_RMSE": _NAN, "dev_MAE": _NAN, "dev_PSNR": _NAN}
    diff = (view.fused - view.sources[view.reference])[mask].astype(np.float64)
    return {"dev_RMSE": float(np.sqrt(np.mean(diff**2))), "dev_MAE": float(np.mean(np.abs(diff))),
            "dev_PSNR": psnr(view.fused[mask], view.sources[view.reference][mask])}


def layer_metrics(view: FusedView, group: str, mask: np.ndarray) -> dict[str, float]:
    """All metrics of one layer group; source metrics are NaN in ``occluded_oov``."""
    row = {"layer": group, "n_pixels": int(mask.sum()), "layer_fraction": float(mask.mean())}
    row |= context_metrics(view, mask) if mask.any() else {
        k: _NAN for k in ("ctx_EN", "ctx_SD", "ctx_SF", "ctx_AG")}
    row |= (dict.fromkeys(SOURCE_METRIC_KEYS, _NAN) if group == "occluded_oov"
            else source_ref_metrics(view, mask))
    return row | reference_deviation(view, mask)

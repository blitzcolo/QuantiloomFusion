"""Layer-wise reporting of one fused image.

Layers come from ``geometry.layers`` (invalid / out_of_view / occluded / edge /
correspondable).  With several sources the per-source maps are combined by the
worst layer (lowest code), so a pixel is "correspondable" only when every
source corresponds there.  Reported groups:

* ``correspondable`` and ``edge``: context (no-reference) metrics and the
  source-referenced metrics, each band counted only inside its own support.
* ``occluded``, ``out_of_view`` and their union ``occluded_oov``: no source has
  information, so the correct behaviour is to fall back to the reference band.  Source metrics are
  not defined there.  The score is the deviation from the reference band in
  the high-pass component (``hp_dev_*``), by distance to the source
  footprint: a multiresolution spline deliberately carries the low
  frequencies (the bands' brightness difference included) some 32 px past a
  footprint edge, and that transition is not ghosting -- ghosting is high-
  frequency structure brought in by misregistration.  The full-signal
  difference stays as a control column (``fullsig_dev_*``).  On samples with
  co-view references the truth-referenced metrics are the answer there: the
  references have real IR content in the occluded region.

No-reference metrics carry the ``ctx_`` prefix: they are context only and are
never a dependent variable of registration sensitivity.

The groups ``correspondable``, ``edge``, ``occluded`` and ``out_of_view`` are
mutually exclusive and, with the invalid layer, cover the grid, so a
truth-referenced MSE decomposes over them: MSE = sum_k a_k MSE_k with a_k the
group's share of the valid pixels (``n_pixels``).  ``occluded_oov`` is kept as
a reporting group for continuity; it is not part of that partition.
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
    "occluded": (Layer.OCCLUDED,),
    "out_of_view": (Layer.OUT_OF_VIEW,),
    "occluded_oov": (Layer.OCCLUDED, Layer.OUT_OF_VIEW),
}
# Groups where no source band has information: source-referenced metrics are undefined.
NO_SOURCE_GROUPS = frozenset({"occluded", "out_of_view", "occluded_oov"})
# The mutually exclusive groups a truth MSE decomposes over (with the invalid layer excluded).
PARTITION_GROUPS = ("correspondable", "edge", "occluded", "out_of_view")
# Correspondable pixels farther than this from every edge / occluded / out-of-view pixel. A
# multiresolution pyramid mixes content across a layer boundary over roughly its coarsest
# support (about 32 px at 4 levels), so near a boundary the correspondable score also carries
# the neighbouring layer; the core group compares registrations away from that influence.
CORE_MARGIN_PX = 32.0
REPORT_GROUPS = (*LAYER_GROUPS, "correspondable_core")
SOURCE_METRIC_KEYS = ("SCD", "Qabf", "MS_SSIM")
MS_SSIM_MARGIN_PX = 32
HP_SIGMA_PX = 2.0
FOOTPRINT_DISTANCE_BINS_PX: tuple[tuple[float, float], ...] = (
    (0.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, float("inf")),
)
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


def layer_masks(layers: np.ndarray, core_margin_px: float = CORE_MARGIN_PX) -> dict[str, np.ndarray]:
    """Masks of every reporting group, ``REPORT_GROUPS`` order.

    ``correspondable_core``: correspondable pixels whose Euclidean distance to the nearest edge,
    occluded or out-of-view pixel exceeds ``core_margin_px`` (invalid pixels and the image border
    do not count as boundaries).  A supplementary statistic, not part of ``PARTITION_GROUPS``.
    """
    masks = {name: np.isin(layers, [int(layer) for layer in members])
             for name, members in LAYER_GROUPS.items()}
    boundary = np.isin(layers, [int(Layer.EDGE), int(Layer.OCCLUDED), int(Layer.OUT_OF_VIEW)])
    corr = masks["correspondable"]
    if boundary.any():
        masks["correspondable_core"] = corr & (ndimage.distance_transform_edt(~boundary) > core_margin_px)
    else:
        masks["correspondable_core"] = corr.copy()
    return masks


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


def _bin_key(lo: float, hi: float) -> str:
    return f"d{lo:g}_{'inf' if np.isinf(hi) else f'{hi:g}'}"


def _highpass(image: np.ndarray, sigma_px: float) -> np.ndarray:
    img = np.asarray(image, np.float64)
    return img - ndimage.gaussian_filter(img, sigma_px, mode="reflect")


def reference_deviation(view: FusedView, mask: np.ndarray, hp_sigma_px: float = HP_SIGMA_PX) -> dict[str, float]:
    """Deviation of the fused image from the reference band on ``mask``.

    ``hp_dev_RMSE``: RMS of the difference of the two images' high-pass
    components (each minus the same Gaussian low-pass of sigma
    ``hp_sigma_px``), overall and per bin of distance (px) to the source
    footprint (the pixels every non-reference band supports): ``(lo, hi]``
    from ``FOOTPRINT_DISTANCE_BINS_PX``, with the pixel count beside it, plus
    ``inside`` (distance 0): pixels of ``mask`` the registration did sample
    the sources at -- in the occluded layer, where its warp put source
    content the source cannot have seen.
    ``fullsig_dev_*``: the full-signal difference, which mostly measures the
    pyramid's deliberate low-frequency transition -- a control, not ghosting.
    """
    bins = {"inside": (-1.0, 0.0)} | {_bin_key(lo, hi): (lo, hi) for lo, hi in FOOTPRINT_DISTANCE_BINS_PX}
    row = {"hp_sigma_px": float(hp_sigma_px)}
    if not mask.any():
        return row | dict.fromkeys(["fullsig_dev_RMSE", "fullsig_dev_MAE", "fullsig_dev_PSNR", "hp_dev_RMSE",
                                    *(f"hp_dev_RMSE_{k}" for k in bins)], _NAN) | {f"n_{k}": 0 for k in bins}
    fused = np.asarray(view.fused, np.float64)
    ref = np.asarray(view.sources[view.reference], np.float64)
    diff = (fused - ref)[mask]
    row |= {"fullsig_dev_RMSE": float(np.sqrt(np.mean(diff**2))), "fullsig_dev_MAE": float(np.mean(np.abs(diff))),
            "fullsig_dev_PSNR": psnr(fused[mask], ref[mask])}
    hp = _highpass(fused, hp_sigma_px) - _highpass(ref, hp_sigma_px)
    row["hp_dev_RMSE"] = float(np.sqrt(np.mean(hp[mask] ** 2)))
    others = [s for b, s in view.supports.items() if b != view.reference]
    footprint = np.logical_and.reduce(others) if others else np.ones(mask.shape, bool)
    dist = ndimage.distance_transform_edt(~footprint) if footprint.any() else np.full(mask.shape, np.inf)
    for key, (lo, hi) in bins.items():
        sel = mask & (dist > lo) & (dist <= hi)
        row[f"hp_dev_RMSE_{key}"] = float(np.sqrt(np.mean(hp[sel] ** 2))) if sel.any() else _NAN
        row[f"n_{key}"] = int(sel.sum())
    return row


def layer_metrics(view: FusedView, group: str, mask: np.ndarray) -> dict[str, float]:
    """All metrics of one layer group; source metrics are NaN in the groups without source information."""
    row = {"layer": group, "n_pixels": int(mask.sum()), "layer_fraction": float(mask.mean())}
    row |= context_metrics(view, mask) if mask.any() else {
        k: _NAN for k in ("ctx_EN", "ctx_SD", "ctx_SF", "ctx_AG")}
    row |= (dict.fromkeys(SOURCE_METRIC_KEYS, _NAN) if group in NO_SOURCE_GROUPS
            else source_ref_metrics(view, mask))
    return row | reference_deviation(view, mask)

"""Truth references from the co-view noise-free images.

The co-view references are rendered on the VIS camera's grid, so they need
no warp.  Two modes that must never be mixed:

* ``REGISTRATION``: compare registration methods with the fusion operator
  fixed.  Reference = that same operator applied to the clean images; the
  score is the loss from parallax misregistration and sensor degradation.
* ``FUSION``: compare fusion operators.  Reference = one common baseline rule
  (``laplacian_pyramid_masked`` on full masks, which equals the plain
  ``laplacian_pyramid``) applied to the clean images, identical for every
  operator.  An operator's own clean output would be a bad reference (a
  constant-output operator would score perfectly).  Additionally scored
  against the clean *source* images, which is rule-agnostic.
"""

from __future__ import annotations

import enum

import numpy as np
from skimage.metrics import structural_similarity

from ..core.bands import BANDS, Band
from ..dataset.types import CameraGeometry
from ..fusion import FusionInput, FusionMethod, create
from ..geometry.warp import Warp
from .full_ref import psnr
from .source_ref import ms_ssim_fusion_n, qabf_n, scd_n

BASELINE_RULE = "laplacian_pyramid_masked"


class TruthMode(enum.StrEnum):
    REGISTRATION = "registration_method"
    FUSION = "fusion_operator"


def identity_warp(shape: tuple[int, int], ref: Band, src: Band) -> Warp:
    h, w = shape
    map_xy = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float64)
    return Warp(ref, src, map_xy, np.ones((h, w), bool), {"method": "identity"})


def clean_input(clean: dict[Band, np.ndarray], geometry: dict[Band, CameraGeometry],
                reference: Band) -> FusionInput:
    """Clean co-view images as a fusion input (identity warps: already aligned)."""
    shape = clean[reference].shape[:2]
    warps = {b: identity_warp(shape, reference, b) for b in clean if b != reference}
    return FusionInput(reference=reference, images=dict(clean), geometry=dict(geometry), warps=warps)


def truth_reference(mode: TruthMode, operator: FusionMethod, clean: FusionInput) -> tuple[np.ndarray, str]:
    """The reference image for ``mode`` and the name of the rule that made it."""
    match mode:
        case TruthMode.REGISTRATION:
            return operator.fuse(clean).image, operator.name
        case TruthMode.FUSION:
            return create(BASELINE_RULE).fuse(clean).image, BASELINE_RULE


def _ssim_map(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM with the same window rule as ``full_ref.ssim``."""
    a, b = img.astype(np.float64), ref.astype(np.float64)
    side = min(a.shape)
    if side < 11:
        win = max(3, side - (side + 1) % 2)
        return structural_similarity(a, b, data_range=1.0, win_size=win, full=True)[1]
    return structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False, full=True)[1]


def truth_metrics(fused: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """PSNR / SSIM of the fused luminance against the truth reference on ``mask``."""
    if not mask.any():
        return {"truth_PSNR": float("nan"), "truth_SSIM": float("nan")}
    return {"truth_PSNR": psnr(fused[mask], reference[mask]),
            "truth_SSIM": float(_ssim_map(fused, reference)[mask].mean())}


def clean_source_metrics(fused: np.ndarray, clean: dict[Band, np.ndarray],
                         mask: np.ndarray) -> dict[str, float]:
    """Source-referenced metrics against the clean sources (rule-agnostic), on ``mask``."""
    nan = float("nan")
    if not mask.any():
        return {"cleansrc_SCD": nan, "cleansrc_Qabf": nan, "cleansrc_MS_SSIM": nan}
    srcs = [np.asarray(clean[b], np.float64) for b in BANDS if b in clean]
    m = mask.astype(np.float64)
    return {"cleansrc_SCD": scd_n(fused, srcs, mask=m),
            "cleansrc_Qabf": qabf_n(fused, srcs, mask=m),
            "cleansrc_MS_SSIM": ms_ssim_fusion_n(fused, srcs, mask=m)}

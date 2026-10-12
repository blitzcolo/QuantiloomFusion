"""Aligning sources onto the reference grid and putting VIS chroma back.

Every fusion input is converted to per-band images on the reference grid
plus per-band support masks. A pixel where a band's warp is invalid is
filled with the reference image value, so fusion degenerates to the
reference there; the support masks are kept for the metrics.
"""

from __future__ import annotations

import numpy as np

from ..core.bands import Band
from ..geometry.warp import apply_warp
from ..signal.colour import rgb_to_ycbcr, ycbcr_to_rgb
from .base import FusionInput


def aligned_sources(x: FusionInput) -> tuple[dict[Band, np.ndarray], dict[Band, np.ndarray]]:
    """All band images on the reference grid, and where each really exists.

    Returns (images, supports): the reference passes through with an
    all-True support; every other band is sampled through its warp with
    "linear" interpolation, support = warp.valid & finite output, and
    unsupported pixels filled with the reference value. Raises ValueError
    when a non-reference band has no warp.
    """
    ref = x.reference
    ref_img = np.asarray(x.images[ref])
    images = {ref: ref_img}
    supports = {ref: np.ones(ref_img.shape[:2], bool)}
    for band, img in x.images.items():
        if band == ref:
            continue
        if x.warps is None or band not in x.warps:
            raise ValueError(f"missing warp for {band}")
        warped = apply_warp(img, x.warps[band], "linear")
        if warped.shape[:2] != ref_img.shape[:2]:
            raise ValueError(
                f"warp for {band} yields grid {warped.shape[:2]}, reference is {ref_img.shape[:2]}"
            )
        if warped.ndim == 3:
            support = x.warps[band].valid & np.isfinite(warped).all(axis=-1)
            fill = ref_img if ref_img.ndim == 3 else ref_img[..., None]
            images[band] = np.where(support[..., None], warped, fill)
        else:
            support = x.warps[band].valid & np.isfinite(warped)
            if ref_img.ndim != 2:
                raise ValueError("cannot fill a greyscale band with an RGB reference")
            images[band] = np.where(support, warped, ref_img)
        supports[band] = support
    return images, supports


def colourize(fused_y: np.ndarray, vis_rgb: np.ndarray) -> np.ndarray:
    """RGB output: fused luminance Y with the VIS Cb/Cr put back.

    vis_rgb must already be on the reference grid. Output is clipped to
    [0, 1] float32.
    """
    ycc = rgb_to_ycbcr(np.asarray(vis_rgb, np.float64))
    fused = np.dstack([np.asarray(fused_y, np.float64), ycc[..., 1:3]])
    return np.clip(ycbcr_to_rgb(fused), 0.0, 1.0).astype(np.float32)

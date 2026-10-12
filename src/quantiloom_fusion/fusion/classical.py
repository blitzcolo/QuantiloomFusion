"""Training-free classical fusion baselines, generalised to N bands.

Ported from paper-exp paper_exp/baselines/classical.py (ab70a3d, 2026-08-15);
on a two-band input at the same resolution the outputs match the original's
fuse(vis, ir). The originals fuse two registered greyscale sources; these
fuse the aligned greys of every band in the input, in BANDS order.

These need no external repository and no checkpoint, so they can populate the
comparison tables from the first run.  They are not the deep baselines the
paper compares against — they are the floor those comparisons sit on, and
having them registered means the comparison table is never a table with one
method in it.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.bands import BANDS, Band
from .align import aligned_sources
from .base import Capabilities, FusionInput, FusionMethod, FusionOutput
from .registry import register

_CAPS = Capabilities(
    bands=frozenset(BANDS),
    needs_registration=True,
    heterogeneous_resolution=False,
    color_output=False,
    trainable=False,
)


def _aligned_greys(x: FusionInput) -> tuple[np.ndarray, dict[Band, np.ndarray]]:
    """Aligned grey images stacked in BANDS order, plus the support masks."""
    images, supports = aligned_sources(x)
    bands = [b for b in BANDS if b in images]
    greys = np.stack([np.asarray(images[b], np.float64) for b in bands])
    return greys, supports


@register("average")
class Average(FusionMethod):
    """Weighted average of the sources (equal weights by default)."""

    capabilities = _CAPS

    def __init__(self, weights: dict[Band, float] | None = None):
        """weights maps a subset of bands to their share; the remaining mass
        splits equally over the rest, so {vis: 0.5} on two bands is the
        original's weight_vis=0.5."""
        self.weights = dict(weights) if weights else None

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        greys, supports = _aligned_greys(x)
        if self.weights is None:
            out = greys.mean(axis=0)
        else:
            bands = [b for b in BANDS if b in x.images]
            rest_bands = [b for b in bands if b not in self.weights]
            w = np.array([self.weights.get(b, (1.0 - sum(self.weights.values())) / len(rest_bands)
                                             if rest_bands else 0.0)
                          for b in bands])
            out = np.tensordot(w, greys, axes=(0, 0))
        return FusionOutput(np.clip(out, 0.0, 1.0).astype(np.float32),
                            {"support": supports})


@register("max_abs")
class MaxAbs(FusionMethod):
    """Per-pixel choose-max, the simplest saliency-preserving rule."""

    capabilities = _CAPS

    def __init__(self):
        pass

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        greys, supports = _aligned_greys(x)
        return FusionOutput(np.clip(greys.max(axis=0), 0.0, 1.0).astype(np.float32),
                            {"support": supports})


@register("laplacian_pyramid")
class LaplacianPyramid(FusionMethod):
    """Classical multi-scale fusion: max-magnitude Laplacian coefficients.

    Per level the coefficient with the largest magnitude across bands wins;
    ties go to the first band in BANDS order (the original's ``>=`` prefers
    the first source the same way). The base level averages the bands.
    """

    capabilities = _CAPS

    def __init__(self, levels: int = 4):
        self.levels = levels

    @staticmethod
    def _pyramid(img: np.ndarray, levels: int):
        gaussian = [img]
        for _ in range(levels):
            gaussian.append(cv2.pyrDown(gaussian[-1]))
        laplacian = []
        for i in range(levels):
            size = (gaussian[i].shape[1], gaussian[i].shape[0])
            laplacian.append(gaussian[i] - cv2.pyrUp(gaussian[i + 1], dstsize=size))
        laplacian.append(gaussian[-1])
        return laplacian

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        greys, supports = _aligned_greys(x)
        n = greys.shape[0]

        # Each pyrDown halves the side, so the level count is bounded by the
        # smaller dimension rather than assumed.
        max_levels = int(np.floor(np.log2(max(min(greys.shape[1:3]), 2)))) - 1
        levels = max(min(self.levels, max_levels), 1)

        laps = [self._pyramid(greys[i], levels) for i in range(n)]

        fused_levels = []
        for level in range(levels):
            stack = np.stack([lap[level] for lap in laps])
            idx = np.argmax(np.abs(stack), axis=0)  # first max wins: BANDS order
            fused_levels.append(np.take_along_axis(stack, idx[None], axis=0)[0])
        fused_levels.append(np.mean([lap[-1] for lap in laps], axis=0))

        out = fused_levels[-1]
        for level in reversed(fused_levels[:-1]):
            size = (level.shape[1], level.shape[0])
            out = cv2.pyrUp(out, dstsize=size) + level
        return FusionOutput(np.clip(out, 0.0, 1.0).astype(np.float32),
                            {"support": supports})

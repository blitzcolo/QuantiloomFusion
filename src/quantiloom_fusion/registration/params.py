"""Parameters shared by the epipolar sweep and the free-search control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .similarity import Similarity

Sigma = float | np.ndarray | None


@dataclass(frozen=True)
class SweepParams:
    """Knobs of a sweep; ref_sigma/src_sigma are optional noise stds in the normalised units of the
    image passed to ``estimate`` (scalar, or a map of that image's shape, as from signal.noise.band_sigma)."""

    similarity: Similarity = Similarity.GRAD_NCC
    n_hypotheses: int = 16
    z_min_m: float = 3.0
    window: int = 41  # box side in ref pixels
    bins: int = 8  # mutual-information histogram bins per axis
    min_saliency: float = 0.1  # (best - second competing peak) / (best - worst)
    min_texture: float = 0.01  # RMS ref gradient (per pixel, stretched images in [0, 1])
    noise_factor: float = 3.0  # window RMS gradient of ref AND src must exceed this x own noise floor
    noise_quantile: float = 0.1  # self-estimated floor = this quantile of the frame's windowed RMS gradient
    separation_px: float = 1.0  # a competing peak must land this far (src px) from the best one
    lr_tol_px: float = 2.0  # left-right disagreement allowed, as src-pixel displacement of d
    left_right: bool = True
    ref_sigma: Sigma = field(default=None, compare=False)
    src_sigma: Sigma = field(default=None, compare=False)

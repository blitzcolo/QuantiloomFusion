"""Full-reference distance measures, reused as self-referenced Delta measures.

Ported verbatim from paper_exp/metrics/full_ref.py (paper-exp ab70a3d,
2026-08-15); only the provenance line was added to this docstring.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as _skimage_ssim

# A pair of identical images has infinite PSNR, which is true and useless: it
# propagates through every mean, sum and LaTeX cell downstream and is not
# valid JSON either. Identical outputs do occur here — the nominal cell of the
# no-noise tier compares a method's output against itself — so the value is
# capped at the level a 16-bit-exact match would give.
PSNR_CEILING_DB = 96.3


def psnr(img: np.ndarray, ref: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for ``[0, 1]`` images."""
    if img.shape != ref.shape:
        raise ValueError(
            f"PSNR needs matching shapes, got {img.shape} and {ref.shape}"
        )
    mse = float(np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2))
    if mse <= 0:
        return PSNR_CEILING_DB
    return float(min(10.0 * np.log10(1.0 / mse), PSNR_CEILING_DB))


def ssim(img: np.ndarray, ref: np.ndarray) -> float:
    """Structural similarity for ``[0, 1]`` images.

    Uses the Gaussian-weighted window, matching the SSIM inside the
    multi-scale metric, so the two never disagree about what SSIM means.
    """
    if img.shape != ref.shape:
        raise ValueError(
            f"SSIM needs matching shapes, got {img.shape} and {ref.shape}"
        )
    min_side = min(img.shape[0], img.shape[1])
    if min_side < 11:
        # Below the Gaussian window a windowed estimate is mostly padding.
        win = max(3, min_side - (min_side + 1) % 2)
        return float(_skimage_ssim(img, ref, data_range=1.0, win_size=win))
    return float(
        _skimage_ssim(img, ref, data_range=1.0, gaussian_weights=True,
                      sigma=1.5, use_sample_covariance=False)
    )

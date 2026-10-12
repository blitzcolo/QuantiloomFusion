"""No-reference fusion quality metrics (EN, SD, SF, AG).

Ported verbatim from paper_exp/metrics/no_reference.py (paper-exp 42024cd,
2026-08-29); only the provenance line was added to this docstring.


All four are reported on the 0-255 intensity scale that the fusion literature
tabulates, so the numbers here are comparable to published tables.  Inputs are
the pipeline's ``[0, 1]`` floats; ``DISPLAY_SCALE`` is the single place that
conversion happens.

Every metric takes an optional region mask.  The evaluation protocol scores
inside the overlap and reports the non-overlap region separately, and the
non-overlap region is the complement of a rectangle — so it cannot be
expressed as a crop, and a metric that only knows how to crop silently
reports the whole frame instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

DISPLAY_SCALE = 255.0
HIST_BINS = 256


def _as_display(img: np.ndarray) -> np.ndarray:
    """Bring a ``[0, 1]`` image onto the 0-255 scale used for reporting.

    Out-of-range values are clipped rather than trusted: the fused output of
    an unbounded network can leave ``[0, 1]``, and an unclipped value would
    otherwise land outside the histogram or inflate the gradient statistics.
    """
    return np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * DISPLAY_SCALE


def _as_mask(mask: Optional[np.ndarray], shape) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape[:2] != tuple(shape[:2]):
        raise ValueError(
            f"Region mask {mask.shape[:2]} does not match image {tuple(shape[:2])}"
        )
    return mask > 0.5


def entropy(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Shannon entropy (EN) of the 256-bin intensity histogram, in bits."""
    display = _as_display(img)
    sel = _as_mask(mask, display.shape)
    values = display[sel] if sel is not None else display
    if values.size == 0:
        return float("nan")
    hist, _ = np.histogram(values, bins=HIST_BINS, range=(0.0, DISPLAY_SCALE))
    total = hist.sum()
    if total == 0:
        return float("nan")
    p = hist / total
    p = p[p > 0]
    # The trailing + 0.0 turns the -0.0 of a single-valued region into 0.0.
    return -float(np.sum(p * np.log2(p))) + 0.0


def standard_deviation(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Standard deviation (SD) of intensity, on the 0-255 scale."""
    display = _as_display(img)
    sel = _as_mask(mask, display.shape)
    values = display[sel] if sel is not None else display
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def spatial_frequency(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Spatial frequency (SF), the classical row/column-frequency RMS.

    .. math::

        RF^2 = \\frac{1}{MN}\\sum_{i}\\sum_{j>0} (I_{i,j} - I_{i,j-1})^2,
        \\qquad SF = \\sqrt{RF^2 + CF^2}

    The normalizer is the image area MN, not the number of differences: with
    ``M(N-1)`` the value drifts with image size, which matters here because
    cells are scored on regions of different sizes.  Under a mask, only
    differences whose *both* endpoints are inside the region contribute, so
    the region boundary is not read as an edge.
    """
    display = _as_display(img)
    sel = _as_mask(mask, display.shape)
    if display.shape[0] < 1 or display.shape[1] < 1:
        return float("nan")

    row_diff = np.diff(display, axis=1) ** 2
    col_diff = np.diff(display, axis=0) ** 2

    if sel is None:
        area = float(display.size)
        rf_sq = float(row_diff.sum() / area)
        cf_sq = float(col_diff.sum() / area)
    else:
        area = float(sel.sum())
        if area == 0:
            return float("nan")
        row_valid = sel[:, 1:] & sel[:, :-1]
        col_valid = sel[1:, :] & sel[:-1, :]
        rf_sq = float(row_diff[row_valid].sum() / area)
        cf_sq = float(col_diff[col_valid].sum() / area)

    return float(np.sqrt(rf_sq + cf_sq))


def average_gradient(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Average gradient (AG) from forward differences, on the 0-255 scale.

    .. math::

        AG = \\frac{1}{(M-1)(N-1)} \\sum \\sqrt{\\frac{\\Delta_x^2 + \\Delta_y^2}{2}}

    Forward differences, not a Sobel response: a 3x3 Sobel sums to eight times
    the per-pixel step, so a Sobel-based AG is ~8x the tabulated value and not
    comparable to any published number.
    """
    display = _as_display(img)
    if display.shape[0] < 2 or display.shape[1] < 2:
        return float("nan")

    dx = np.diff(display, axis=1)[:-1, :]
    dy = np.diff(display, axis=0)[:, :-1]
    grad = np.sqrt((dx ** 2 + dy ** 2) / 2.0)

    sel = _as_mask(mask, display.shape)
    if sel is None:
        return float(np.mean(grad))

    valid = sel[:-1, :-1] & sel[:-1, 1:] & sel[1:, :-1]
    if not valid.any():
        return float("nan")
    return float(np.mean(grad[valid]))

"""Source-referenced fusion metrics (SCD, Qabf, MS-SSIM).

Ported from paper_exp/metrics/source_ref.py (paper-exp 42024cd, 2026-08-29).
The two-source ``scd``/``qabf``/``ms_ssim_fusion`` are verbatim; the ``*_n``
functions at the bottom are our N-source generalisations, each stated to
reduce to its original on two sources.

These score a fused image against the two sources it came from, so they need
all three images on one grid and one intensity scale.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import cv2
from scipy.ndimage import convolve1d
from skimage.metrics import structural_similarity

from .no_reference import DISPLAY_SCALE

# Xydeas & Petrovic (2000), "Objective image fusion performance measure":
# sigmoid constants for the edge-strength and orientation preservation models,
# and the strength exponent used to weight each pixel's contribution.
QABF_GAMMA_G, QABF_KAPPA_G, QABF_SIGMA_G = 0.9994, -15.0, 0.5
QABF_GAMMA_A, QABF_KAPPA_A, QABF_SIGMA_A = 0.9879, -22.0, 0.8
QABF_L = 1.5

MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _pearson(a: np.ndarray, b: np.ndarray,
             mask: Optional[np.ndarray] = None) -> float:
    a_flat = a.astype(np.float64)
    b_flat = b.astype(np.float64)
    if mask is not None:
        sel = np.asarray(mask) > 0.5
        a_flat, b_flat = a_flat[sel], b_flat[sel]
    else:
        a_flat, b_flat = a_flat.ravel(), b_flat.ravel()
    if a_flat.size == 0:
        return float("nan")
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
    if denom == 0:
        # Undefined, not zero. A correlation needs both series to vary, and a
        # constant one has no direction to agree or disagree with -- returning
        # 0.0 reports "no agreement", which is a *score*, and a bad one.
        #
        # This fires on real data rather than in theory. SCD's second term
        # correlates the fused image against the infrared, and outside the
        # overlap the infrared is filled with a constant, so every
        # out-of-overlap SCD took this branch and reported a number that looked
        # like a measurement of poor fusion. Pandas skips NaN when it averages,
        # so an undefined cell now drops out of the mean instead of dragging it
        # toward zero.
        return float("nan")
    return float(np.sum(a_centered * b_centered) / denom)


def _check_shapes(fused: np.ndarray, vis: np.ndarray, ir: np.ndarray) -> None:
    if not (fused.shape == vis.shape == ir.shape):
        raise ValueError(
            "Source-referenced metrics need the fused image and both sources "
            f"on one grid, got fused {fused.shape}, vis {vis.shape}, "
            f"ir {ir.shape}. Resample the sources onto the evaluation frame "
            "before scoring."
        )


def scd(fused: np.ndarray, vis: np.ndarray, ir: np.ndarray,
        mask: Optional[np.ndarray] = None,
        ir_support: Optional[np.ndarray] = None) -> float:
    """Sum of the correlations of differences.

    ``corr(F - B, A) + corr(F - A, B)``: what the fused image adds over one
    source should correlate with the *other* source.  Correlating the
    difference with the source it was taken from instead is the classic
    sign error, and it produces plausible values of the opposite meaning.
    """
    _check_shapes(fused, vis, ir)
    # Both correlations involve the infrared, so both are confined to where
    # it exists: outside its footprint the "infrared" is the harness's fill,
    # and a correlation with a fill is a property of the fill convention.
    # Measured on three rho < 1 cells, the full-frame value moved between
    # 0.89 and 1.50 under a change of fill alone.
    mask = _within(mask, ir_support)
    return (_pearson(fused - ir, vis, mask)
            + _pearson(fused - vis, ir, mask))


def _within(mask: Optional[np.ndarray],
            support: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """The region *and* the support, or whichever one is given."""
    if support is None:
        return mask
    inside = np.asarray(support) > 0.5
    if mask is None:
        return inside.astype(np.float32)
    return ((np.asarray(mask) > 0.5) & inside).astype(np.float32)


def _interior(support: np.ndarray, radius: int) -> np.ndarray:
    """Pixels whose whole (2r+1)^2 neighbourhood lies inside ``support``.

    The frame edge is not a footprint edge: it is padded by replication,
    the way the filters themselves pad, so a support covering the whole
    frame is returned unchanged.
    """
    inside = np.asarray(support) > 0.5
    padded = np.pad(inside, radius, mode="edge")
    out = np.ones_like(inside)
    h, w = inside.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out &= padded[dy:dy + h, dx:dx + w]
    return out


def _sobel_edge(img: np.ndarray):
    """Sobel edge strength and *undirected* orientation.

    Orientation is taken modulo pi.  An edge and its reverse-polarity twin
    are the same edge, and thermal fusion is full of structures whose VIS and
    IR contrast have opposite sign; scoring those as maximally misoriented
    would drive the IR term of Qabf toward zero for exactly the content the
    metric exists to reward.
    """
    gx = cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    strength = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) % np.pi
    return strength, orientation


def _angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest angle between two undirected orientations, in [0, pi/2]."""
    diff = np.abs(a - b) % np.pi
    return np.minimum(diff, np.pi - diff)


def _sobel_interior(support: np.ndarray) -> np.ndarray:
    """Pixels whose whole 3x3 Sobel neighbourhood lies inside ``support``."""
    return _interior(support, 1)


def qabf(fused: np.ndarray, vis: np.ndarray, ir: np.ndarray,
         mask: Optional[np.ndarray] = None,
         ir_support: Optional[np.ndarray] = None) -> float:
    """Xydeas-Petrovic gradient-based fusion performance measure Q^{AB/F}.

    Edge strength and orientation preservation are each passed through the
    published sigmoid, multiplied, and weighted by source edge strength
    raised to ``L``.  Dropping the sigmoids and the exponent — using the raw
    ratios instead — yields a number in the same range that is not the
    published metric and cannot be compared with anyone else's table.

    ``ir_support`` is the infrared footprint on the frame, for a source that
    does not cover it.  The IR reaches this metric placed on the evaluation
    frame with a constant fill outside its footprint, and the step from
    image to fill is an edge the Sobel sees from both sides: one pixel in
    and one pixel out of the footprint carry a gradient of the order of the
    IR level itself, against natural gradients a tenth of that, raised to
    ``L``.  Measured on the rho < 1 test cells, that two-pixel ring held
    **14–42% of the in-region weight and 21–92% of the out-region weight**,
    and a perfect VIS pass-through outside the footprint -- the behaviour an
    overlap-aware method exists to produce -- scored 0.08 to 0.77 there
    instead of 0.97.  The ring is the harness's fill convention, not either
    source, so the IR weight is zero wherever its Sobel neighbourhood leaves
    the footprint.  A support covering the frame changes nothing.
    """
    _check_shapes(fused, vis, ir)
    eps = np.finfo(np.float64).eps

    g_f, alpha_f = _sobel_edge(fused)
    g_v, alpha_v = _sobel_edge(vis)
    g_i, alpha_i = _sobel_edge(ir)

    def _preservation(g_source: np.ndarray, alpha_source: np.ndarray) -> np.ndarray:
        # Relative strength: the ratio is inverted where the fused edge is
        # stronger, so both over- and under-representation are penalized.
        g_max = np.maximum(g_f, g_source)
        g_min = np.minimum(g_f, g_source)
        g_ratio = np.where(g_max > eps, g_min / (g_max + eps), 0.0)

        alpha_ratio = 1.0 - _angular_distance(alpha_f, alpha_source) / (np.pi / 2)

        q_g = QABF_GAMMA_G / (1.0 + np.exp(QABF_KAPPA_G * (g_ratio - QABF_SIGMA_G)))
        q_a = QABF_GAMMA_A / (1.0 + np.exp(QABF_KAPPA_A * (alpha_ratio - QABF_SIGMA_A)))
        return q_g * q_a

    q_vf = _preservation(g_v, alpha_v)
    q_if = _preservation(g_i, alpha_i)

    w_v = g_v ** QABF_L
    w_i = g_i ** QABF_L

    if ir_support is not None:
        w_i = w_i * _sobel_interior(ir_support).astype(np.float64)

    if mask is not None:
        sel = (np.asarray(mask) > 0.5).astype(np.float64)
        w_v = w_v * sel
        w_i = w_i * sel

    denom = float(np.sum(w_v) + np.sum(w_i))
    if denom <= eps:
        # Qabf is a gradient-weighted average of how well each source's edges
        # survived. With no gradient anywhere in either source there is no
        # weight to average under, and the answer is undefined rather than
        # zero: 0.0 says "every edge was lost" about an image that had none.
        #
        # The band this matters in is the one where a night frame's visible
        # channel is nearly empty. Reporting zero there would make "the visible
        # image held nothing" indistinguishable from "the fusion destroyed what
        # it held".
        return float("nan")
    return float((np.sum(w_v * q_vf) + np.sum(w_i * q_if)) / denom)


# ---------------------------------------------------------------------------
# MS-SSIM
# ---------------------------------------------------------------------------

_MS_SSIM_MIN_SIDE = 11  # the Gaussian window skimage uses at every scale


def _gaussian_kernel(sigma: float = 1.5, radius: int = 5) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _downsample(img: np.ndarray) -> np.ndarray:
    """Low-pass, then decimate by two.

    The prefilter is not optional: decimating without it aliases the detail
    that the coarser scales are supposed to be measuring.
    """
    kernel = _gaussian_kernel()
    smoothed = convolve1d(img, kernel, axis=0, mode="reflect")
    smoothed = convolve1d(smoothed, kernel, axis=1, mode="reflect")
    return smoothed[::2, ::2]


def _masked_mean(values: np.ndarray, mask: Optional[np.ndarray]) -> float:
    if mask is None:
        return float(np.mean(values))
    sel = np.asarray(mask) > 0.5
    if not sel.any():
        return float("nan")
    return float(np.mean(values[sel]))


def _ssim_components(img1: np.ndarray, img2: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> tuple[float, float]:
    """``(ssim, contrast_structure)`` at one scale, on the 0-255 scale.

    Under a mask both are averages of the per-pixel maps over the region.
    The windows themselves still straddle the region boundary — a windowed
    statistic cannot do otherwise — so region scores near a boundary carry a
    window's worth of neighbouring content by construction.
    """
    _, ssim_map = structural_similarity(
        img1, img2, data_range=DISPLAY_SCALE,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        full=True,
    )
    ssim_val = _masked_mean(ssim_map, mask)

    # cs = (2 sigma_xy + C2) / (sigma_x^2 + sigma_y^2 + C2), computed with the
    # same Gaussian window skimage uses so the two agree scale for scale.
    c2 = (0.03 * DISPLAY_SCALE) ** 2
    kernel = _gaussian_kernel()

    def _filter(x):
        return convolve1d(convolve1d(x, kernel, axis=0, mode="reflect"),
                          kernel, axis=1, mode="reflect")

    mu1, mu2 = _filter(img1), _filter(img2)
    sigma1 = _filter(img1 * img1) - mu1 * mu1
    sigma2 = _filter(img2 * img2) - mu2 * mu2
    sigma12 = _filter(img1 * img2) - mu1 * mu2
    cs_map = (2 * sigma12 + c2) / (sigma1 + sigma2 + c2)
    return ssim_val, _masked_mean(cs_map, mask)


def ms_ssim(img1: np.ndarray, img2: np.ndarray,
            weights: tuple[float, ...] = MS_SSIM_WEIGHTS,
            mask: Optional[np.ndarray] = None) -> float:
    """Multi-scale SSIM (Wang, Simoncelli & Bovik, 2003).

    Contrast-structure only at the finer scales and the full SSIM at the
    coarsest, as published — multiplying the full SSIM at every scale
    double-counts the luminance term once per scale.

    Scales that would leave an image smaller than the Gaussian window are
    dropped and the remaining weights renormalized, rather than repeating the
    last usable value: a repeated value is a real measurement in the product
    and quietly biases the result.

    The same rule applies to a scale at which the *region* has emptied.  The
    mask is eroded as it is downsampled, so a thin region -- the 34-pixel
    ring outside the overlap at rho = 0.75 -- is gone by the fifth scale,
    and an empty region used to contribute 0.0, clipped to 1e-8 and raised
    to the scale's weight: a factor of 0.085 on the product, for identical
    images.  Measured: two identical images on a 20-pixel strip scored
    0.0011.  A scale with nothing in it is not a measurement of anything.
    An empty region at the finest scale is undefined, and says so.
    """
    a = np.clip(np.asarray(img1, dtype=np.float64), 0.0, 1.0) * DISPLAY_SCALE
    b = np.clip(np.asarray(img2, dtype=np.float64), 0.0, 1.0) * DISPLAY_SCALE

    m = None if mask is None else (np.asarray(mask) > 0.5).astype(np.float64)

    # Measure every scale that is large enough for the window, then combine.
    scales: list[tuple[float, float]] = []
    while len(scales) < len(weights) and min(a.shape[:2]) >= _MS_SSIM_MIN_SIDE:
        if m is not None and not (m > 0.5).any():
            break
        scales.append(_ssim_components(a, b, m))
        if len(scales) == len(weights):
            break
        a, b = _downsample(a), _downsample(b)
        if m is not None:
            # Erode as it shrinks: a coarse-scale pixel counts as inside the
            # region only if the fine-scale pixels it came from all were.
            m = _downsample(m)
            m = (m > 0.99).astype(np.float64)

    if not scales and m is not None and not (m > 0.5).any():
        return float("nan")
    if not scales:
        # Smaller than the window at full resolution: one small-window SSIM.
        side = min(img1.shape[0], img1.shape[1])
        win = max(3, side - (side + 1) % 2)
        return float(structural_similarity(
            np.clip(img1, 0.0, 1.0), np.clip(img2, 0.0, 1.0),
            data_range=1.0, win_size=win,
        ))

    # Contrast-structure at every scale but the coarsest reached, full SSIM there.
    values = [cs for _, cs in scales[:-1]] + [scales[-1][0]]
    w = np.asarray(weights[:len(scales)], dtype=np.float64)
    w = w / w.sum()
    vals = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0)
    return float(np.prod(vals ** w))


# Radius of skimage's Gaussian SSIM window at sigma = 1.5, truncate = 3.5.
_SSIM_WINDOW_RADIUS = int(3.5 * 1.5 + 0.5)


def ms_ssim_fusion(fused: np.ndarray, vis: np.ndarray, ir: np.ndarray,
                   mask: Optional[np.ndarray] = None,
                   ir_support: Optional[np.ndarray] = None) -> float:
    """Mean MS-SSIM of the fused image against each source that is present.

    ``ir_support`` is the infrared footprint when the IR does not cover the
    frame.  The IR term is then confined to the pixels whose finest-scale
    window lies inside the footprint, so the fill outside it is never
    compared against; where no such pixel is in the region -- outside the
    overlap -- the fused image is scored against the visible image alone,
    which is the only source there.  That mirrors :func:`qabf`, whose edge
    weight is zero for an absent source, rather than :func:`scd`, whose
    correlation has no such natural weighting and is undefined instead.

    Inside the overlap this is a partial remedy: the coarser scales are built
    from a blurred, downsampled pyramid of the placed image, and the fill
    leaks into that pyramid a window's width past the boundary at every
    scale.  The protocol scores the overlap on a *crop* for that reason (see
    :func:`paper_exp.metrics.protocol.score_cell`); the support handles the
    full-frame and out-of-overlap cases, where a crop is not available.
    """
    _check_shapes(fused, vis, ir)
    vis_term = ms_ssim(fused, vis, mask=mask)
    if ir_support is None:
        return 0.5 * (vis_term + ms_ssim(fused, ir, mask=mask))
    ir_mask = _within(mask, _interior(ir_support, _SSIM_WINDOW_RADIUS))
    if not (ir_mask > 0.5).any():
        return vis_term
    return 0.5 * (vis_term + ms_ssim(fused, ir, mask=ir_mask))


# ---------------------------------------------------------------------------
# N-source generalisations
#
# Everything below is ours, not a published definition: the three functions
# extend the two-source metrics above to a source list, and each reduces to
# its original on two sources (verified against paper-exp in
# tests/test_metrics_source_ref.py). ``supports`` is one footprint per
# source, in source order; ``None`` means the source covers the frame.  A
# support restricts where its source participates, with the same rule the
# original applies to the infrared: for ``scd_n`` the correlation region is
# the mask intersected with every support (a correlation has no natural
# weighting, so fill outside any footprint must not enter it); for
# ``qabf_n`` a source's edge weight is zero wherever its 3x3 Sobel
# neighbourhood leaves its footprint; for ``ms_ssim_fusion_n`` a source's
# term is confined to pixels whose finest-scale window lies inside its
# footprint and drops out of the mean where no such pixel exists.
# ---------------------------------------------------------------------------


def _check_shapes_n(fused: np.ndarray, sources: Sequence[np.ndarray]) -> None:
    if not sources:
        raise ValueError("Source-referenced metrics need at least one source.")
    for i, src in enumerate(sources):
        if src.shape != fused.shape:
            raise ValueError(
                "Source-referenced metrics need the fused image and every "
                f"source on one grid, got fused {fused.shape} and source "
                f"{i} {src.shape}. Resample the sources onto the evaluation "
                "frame before scoring."
            )


def _normalize_supports(
        supports: Optional[Sequence[Optional[np.ndarray]]],
        n: int) -> list[Optional[np.ndarray]]:
    if supports is None:
        return [None] * n
    if len(supports) != n:
        raise ValueError(
            f"supports must give one footprint per source, got "
            f"{len(supports)} for {n} sources."
        )
    return list(supports)


def scd_n(fused: np.ndarray, sources: Sequence[np.ndarray],
          mask: Optional[np.ndarray] = None,
          supports: Optional[Sequence[Optional[np.ndarray]]] = None) -> float:
    """Sum of correlations of differences, generalised to N sources.

    ``(2/N) * sum_i corr(F - mean_{j != i} S_j, S_i)``: what the fused image
    adds over the mean of the *other* sources should correlate with each
    source in turn.  For N = 2 this is exactly ``corr(F - B, A) +
    corr(F - A, B)``, so ``scd_n(f, [a, b], mask, [None, s]) == scd`` with
    ``ir_support=s``.  Every correlation involves every source through the
    mean, so the region is the mask intersected with *all* supports.
    """
    _check_shapes_n(fused, sources)
    if len(sources) < 2:
        raise ValueError("scd_n needs at least two sources.")
    m = mask
    for sup in _normalize_supports(supports, len(sources)):
        m = _within(m, sup)
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in sources])
    n = len(sources)
    acc = 0.0
    stacked_sum = stack.sum(axis=0)
    for i in range(n):
        others = (stacked_sum - stack[i]) / (n - 1)
        acc += _pearson(np.asarray(fused, dtype=np.float64) - others,
                        sources[i], m)
    return (2.0 / n) * acc


def _qabf_preservation(g_f: np.ndarray, alpha_f: np.ndarray,
                       g_source: np.ndarray, alpha_source: np.ndarray,
                       eps: float) -> np.ndarray:
    """The fused-vs-source edge preservation map from :func:`qabf`."""
    # Relative strength: the ratio is inverted where the fused edge is
    # stronger, so both over- and under-representation are penalized.
    g_max = np.maximum(g_f, g_source)
    g_min = np.minimum(g_f, g_source)
    g_ratio = np.where(g_max > eps, g_min / (g_max + eps), 0.0)

    alpha_ratio = 1.0 - _angular_distance(alpha_f, alpha_source) / (np.pi / 2)

    q_g = QABF_GAMMA_G / (1.0 + np.exp(QABF_KAPPA_G * (g_ratio - QABF_SIGMA_G)))
    q_a = QABF_GAMMA_A / (1.0 + np.exp(QABF_KAPPA_A * (alpha_ratio - QABF_SIGMA_A)))
    return q_g * q_a


def qabf_n(fused: np.ndarray, sources: Sequence[np.ndarray],
           mask: Optional[np.ndarray] = None,
           supports: Optional[Sequence[Optional[np.ndarray]]] = None) -> float:
    """Xydeas-Petrovic Q^{AB/F}, generalised to N sources.

    The weighted mean ``sum_s sum_px Q^{sF} * w^s / sum_s sum_px w^s`` runs
    over every source — exactly how the two-source :func:`qabf` combines its
    pair — with ``w^s = g_s ** L`` zeroed wherever the source's 3x3 Sobel
    neighbourhood leaves its support.  ``qabf_n(f, [a, b], mask,
    [None, s]) == qabf(f, a, b, mask, ir_support=s)``.
    """
    _check_shapes_n(fused, sources)
    eps = np.finfo(np.float64).eps
    sups = _normalize_supports(supports, len(sources))

    g_f, alpha_f = _sobel_edge(fused)
    sel = None if mask is None else (np.asarray(mask) > 0.5).astype(np.float64)

    num = 0.0
    den = 0.0
    for src, sup in zip(sources, sups):
        g_s, alpha_s = _sobel_edge(src)
        q = _qabf_preservation(g_f, alpha_f, g_s, alpha_s, eps)
        w = g_s ** QABF_L
        if sup is not None:
            w = w * _sobel_interior(sup).astype(np.float64)
        if sel is not None:
            w = w * sel
        num += float(np.sum(w * q))
        den += float(np.sum(w))
    if den <= eps:
        return float("nan")
    return float(num / den)


def ms_ssim_fusion_n(fused: np.ndarray, sources: Sequence[np.ndarray],
                     mask: Optional[np.ndarray] = None,
                     supports: Optional[Sequence[Optional[np.ndarray]]] = None
                     ) -> float:
    """Mean MS-SSIM against each source present, generalised to N sources.

    The original averages the fused-vs-visible and fused-vs-infrared terms
    ``0.5 * (v + i)``; here the mean runs over every source.  A source with
    a support contributes its term only on the pixels whose finest-scale
    window lies inside its footprint, and is dropped from the mean where no
    such pixel exists — the same rule :func:`ms_ssim_fusion` applies to the
    infrared.  All-empty gives NaN.  ``ms_ssim_fusion_n(f, [a, b], mask,
    [None, s]) == ms_ssim_fusion(f, a, b, mask, ir_support=s)``.
    """
    _check_shapes_n(fused, sources)
    sups = _normalize_supports(supports, len(sources))
    terms = []
    for src, sup in zip(sources, sups):
        if sup is None:
            terms.append(ms_ssim(fused, src, mask=mask))
            continue
        src_mask = _within(mask, _interior(sup, _SSIM_WINDOW_RADIUS))
        if not (src_mask > 0.5).any():
            continue
        terms.append(ms_ssim(fused, src, mask=src_mask))
    if not terms:
        return float("nan")
    return float(np.mean(terms))

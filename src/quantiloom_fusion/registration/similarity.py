"""Local similarities between a reference image and a resampled source image.

VIS vs thermal intensity is non-monotonic and edge polarity can flip, so none
of these compares intensities directly: gradient-magnitude NCC, polarity-
invariant gradient-orientation agreement, and windowed mutual information.
Higher is more similar. All windows are square boxes in reference pixels.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import cv2
import numpy as np

_EPS_NCC = 1e-10
_EPS_ORIENT = 1e-6


class Similarity(enum.StrEnum):
    GRAD_NCC = "grad_ncc"
    GRAD_ORIENT = "grad_orient"
    LOCAL_MI = "local_mi"


@dataclass(frozen=True)
class Features:
    """Per-pixel quantities one similarity needs; unused fields stay None."""

    gx: np.ndarray | None = None
    gy: np.ndarray | None = None
    mag: np.ndarray | None = None
    quantized: np.ndarray | None = None  # int32 bin index in [0, bins)


def to_gray(image: np.ndarray) -> np.ndarray:
    """(H, W) float32 from a (H, W) or (H, W, 3) image (BT.601 luma)."""
    a = np.asarray(image, np.float32)
    if a.ndim == 3:
        a = a @ np.asarray([0.299, 0.587, 0.114], np.float32)
    return a


def smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; a no-op below sigma 0.3."""
    if sigma < 0.3:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def intensity_range(image: np.ndarray, min_span: float = 0.05) -> tuple[float, float]:
    """Robust (1%, 99%) range, at least min_span wide so near-flat images are not amplified."""
    lo, hi = np.percentile(image, [1.0, 99.0])
    return float(lo), float(max(hi, lo + min_span))


def features(image: np.ndarray, sim: Similarity, bins: int, rng: tuple[float, float]) -> Features:
    """Features of an (already smoothed, float32) image for the given similarity."""
    match sim:
        case Similarity.LOCAL_MI:
            lo, hi = rng
            q = np.clip((image - lo) / (hi - lo) * bins, 0, bins - 1)
            return Features(quantized=q.astype(np.int32))
        case Similarity.GRAD_NCC | Similarity.GRAD_ORIENT:
            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE) / 8.0
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE) / 8.0
            return Features(gx=gx, gy=gy, mag=np.hypot(gx, gy))
        case unreachable:
            raise AssertionError(unreachable)


def _box(a: np.ndarray, window: int) -> np.ndarray:
    return cv2.boxFilter(a, -1, (window, window), borderType=cv2.BORDER_REPLICATE)


def texture(ref: Features, window: int) -> np.ndarray:
    """RMS gradient magnitude of the reference over the window (gradient features)."""
    return np.sqrt(np.maximum(_box(ref.mag.astype(np.float64) ** 2, window), 0.0))


def score(sim: Similarity, ref: Features, src: Features, window: int, bins: int) -> np.ndarray:
    """(H, W) float64 similarity of ref and the resampled source in each window."""
    match sim:
        case Similarity.GRAD_NCC:
            a, b = ref.mag.astype(np.float64), src.mag.astype(np.float64)
            ma, mb = _box(a, window), _box(b, window)
            cov = _box(a * b, window) - ma * mb
            va = np.maximum(_box(a * a, window) - ma * ma, 0.0)
            vb = np.maximum(_box(b * b, window) - mb * mb, 0.0)
            return cov / np.sqrt(va * vb + _EPS_NCC)
        case Similarity.GRAD_ORIENT:
            dot = (ref.gx * src.gx + ref.gy * src.gy).astype(np.float64)
            w = (ref.mag * src.mag).astype(np.float64)
            # dot^2 / w = w * cos^2(angle): invariant to gradient sign.
            return _box(dot * dot / (w + 1e-12), window) / (_box(w, window) + _EPS_ORIENT)
        case Similarity.LOCAL_MI:
            return _local_mi(ref.quantized, src.quantized, window, bins)
        case unreachable:
            raise AssertionError(unreachable)


def _local_mi(qr: np.ndarray, qs: np.ndarray, window: int, bins: int) -> np.ndarray:
    joint = np.empty((bins, bins) + qr.shape, np.float32)
    for a in range(bins):
        ra = qr == a
        for b in range(bins):
            joint[a, b] = _box((ra & (qs == b)).astype(np.float32), window)
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    log_term = np.log(joint + 1e-9) - np.log(pa[:, None] + 1e-9) - np.log(pb[None, :] + 1e-9)
    return (joint * log_term).sum(axis=(0, 1)).astype(np.float64)

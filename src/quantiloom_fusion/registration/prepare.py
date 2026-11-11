"""Input preparation for the sweeps: contrast stretch, smoothing, gradient features and noise floors.

The structure-strength gate compares a window's RMS gradient with the gradient
noise floor of the same image. With a caller-supplied per-band sigma the floor
is that sigma pushed through the pipeline's smoothing and gradient filter
(white-noise gain). Without one it is a low quantile of the image's own windowed
RMS gradient (over windows that contain any signal): every frame has flat windows, and measuring the floor in the same
quantity the gate tests stays valid for spatially correlated noise (demosaicked
or sparse photon-count night frames), where Laplacian/MAD pixel-noise estimators
read an order of magnitude low.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from ..dataset.types import CameraGeometry
from .params import Sigma, SweepParams
from .similarity import Features, Similarity, features, intensity_range, smooth, texture, to_gray

_MIN_SIGMA = 1e-6
_FLAT = 1e-4  # windows below this RMS gradient are empty (e.g. zero photon counts), not noise
_GRADIENT = Similarity.GRAD_NCC


@dataclass(frozen=True)
class Prepared:
    """Smoothed source, ref features, MI range, and the ref/src structure-to-noise ratios."""

    src: np.ndarray
    ref_f: Features
    rng: tuple[float, float]
    tex: np.ndarray  # (H_ref, W_ref) window RMS ref gradient
    ref_snr: np.ndarray  # (H_ref, W_ref) tex / own gradient noise floor
    src_snr: np.ndarray  # (H_src, W_src) the same on the source grid


def stretch(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Map the 1%..99% range to [0, 1] (so gates mean the same across bands); also returns the span."""
    lo, hi = intensity_range(gray)
    return ((gray - lo) / (hi - lo)).astype(np.float32), hi - lo


@functools.cache
def _gradient_gain(blur: float) -> float:
    """RMS gradient magnitude that unit white noise yields after the pipeline's smoothing."""
    noise = np.random.default_rng(0).standard_normal((192, 192)).astype(np.float32)
    mag = features(smooth(noise, blur), _GRADIENT, 8, (0.0, 1.0)).mag
    return float(np.sqrt(np.mean(mag[8:-8, 8:-8] ** 2)))


def _snr(smoothed: np.ndarray, blur: float, window: int, given: Sigma, span: float,
         quantile: float) -> np.ndarray:
    tex = texture(features(smoothed, _GRADIENT, 8, (0.0, 1.0)), window)
    if given is None:
        active = tex[tex > _FLAT]
        floor = np.quantile(active, quantile) if active.size else _MIN_SIGMA
    else:
        sigma = np.asarray(given, np.float64) / span
        if sigma.ndim and sigma.shape != smoothed.shape:
            raise ValueError(f"sigma map {sigma.shape} does not match the image {smoothed.shape}")
        floor = _gradient_gain(round(blur, 3)) * sigma
    return tex / np.maximum(floor, _MIN_SIGMA)


def prepare(ref_g: CameraGeometry, src_g: CameraGeometry, ref_image: np.ndarray, src_image: np.ndarray,
            params: SweepParams) -> Prepared:
    """Resolution-matched smoothing, features, and noise-normalised structure maps for both images."""
    ratio = ref_g.fx / src_g.fx
    ref_blur, src_blur = 0.5 * max(1.0, ratio), 0.5 * max(1.0, 1.0 / ratio)
    ref_st, ref_span = stretch(to_gray(ref_image))
    src_st, src_span = stretch(to_gray(src_image))
    ref_s, src_s = smooth(ref_st, ref_blur), smooth(src_st, src_blur)
    mi = params.similarity == Similarity.LOCAL_MI
    rng = intensity_range(src_s) if mi else (0.0, 1.0)
    ref_rng = intensity_range(ref_s) if mi else (0.0, 1.0)
    src_window = max(5, round(params.window / ratio))
    return Prepared(
        src=src_s,
        ref_f=features(ref_s, params.similarity, params.bins, ref_rng),
        rng=rng,
        tex=texture(features(ref_s, _GRADIENT, params.bins, ref_rng), params.window),
        ref_snr=_snr(ref_s, ref_blur, params.window, params.ref_sigma, ref_span, params.noise_quantile),
        src_snr=_snr(src_s, src_blur, src_window, params.src_sigma, src_span, params.noise_quantile),
    )


def structure_ok(prep: Prepared, map_xy: np.ndarray, factor: float) -> np.ndarray:
    """Ref window and the source window at map_xy both exceed `factor` x their own noise floor."""
    hs, ws = prep.src_snr.shape
    finite = np.isfinite(map_xy).all(axis=2)
    ix = np.clip(np.rint(np.where(finite, map_xy[..., 0], 0.0)), 0, ws - 1).astype(np.int64)
    iy = np.clip(np.rint(np.where(finite, map_xy[..., 1], 0.0)), 0, hs - 1).astype(np.int64)
    return finite & (prep.ref_snr >= factor) & (prep.src_snr[iy, ix] >= factor)

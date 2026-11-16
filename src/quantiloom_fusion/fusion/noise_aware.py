"""Mask-weighted Laplacian pyramid fusion, plain and noise-aware.

Both methods extend each source band outside its validity mask by the
nearest valid pixel before decomposing (no scene-independent step at the
footprint edge), weight every band's coefficients by the Gaussian pyramid of
its mask (Burt & Adelson multiresolution spline), and select per pixel the
band with the largest WEIGHTED coefficient. The noise-aware variant first
soft-thresholds every band's coefficients (reference included) with
c * max(0, 1 - kappa * sigma / |c|), so selection compares shrunk magnitudes
and pure-noise regions inject no detail. The masked baseline is the same
code without shrinkage.

The per-level noise std is sigma times a per-level gain. Every band reaches
the pyramid through its own processing chain -- VIS through demosaic, white
balance and the luminance sum, MWIR/LWIR through the warp's resampling onto
the reference grid -- none of which leaves the noise white, so the gain is
measured per band by pushing unit white noise through that chain and the
same pyramid (``chain_level_gains``). ``level_noise_gain`` is the white
special case.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from functools import lru_cache

import cv2
import numpy as np
from scipy import ndimage

from ..core.bands import BANDS, Band
from ..geometry.warp import Warp, apply_warp
from .align import aligned_sources
from .base import Capabilities, FusionInput, FusionMethod, FusionOutput
from .classical import _CAPS, LaplacianPyramid
from .registry import register

Sigma = float | np.ndarray


def pyramid_levels(shape: tuple[int, ...], requested: int) -> int:
    """Detail levels, bounded by the smaller side exactly like the baseline."""
    max_levels = int(np.floor(np.log2(max(min(shape[:2]), 2)))) - 1
    return max(min(requested, max_levels), 1)


def gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [img]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


@lru_cache(maxsize=None)
def level_noise_gain(levels: int) -> tuple[float, ...]:
    """Std of each detail level for unit-variance white noise input.

    White-noise approximation of the band-pass kernel energy, computed by
    pushing unit impulses through the same pyramid code the fusion uses. The
    pyramid is periodic in 2**levels input pixels, so one impulse per phase
    of that period (in the image interior) covers every kernel alignment:
    gain_l^2 = mean over level-l pixels of the white-noise variance
             = 4**(l - levels) * sum over phases of ||level-l response||^2.
    Index l is detail level l.
    """
    period = 2**levels
    size = 8 * period
    energy = np.zeros(levels)
    for dy in range(period):
        for dx in range(period):
            impulse = np.zeros((size, size))
            impulse[size // 2 + dy, size // 2 + dx] = 1.0
            laps = LaplacianPyramid._pyramid(impulse, levels)
            energy += [float(np.sum(lap**2)) for lap in laps[:levels]]
    return tuple(float(np.sqrt(e * 4.0 ** (lv - levels))) for lv, e in enumerate(energy))


def chain_level_gains(chain: Callable[[np.ndarray], np.ndarray], native_shape: tuple[int, int],
                      levels: int, realisations: int = 4, seed: int = 0) -> tuple[float, ...]:
    """Std of each detail level for unit white noise on the native grid pushed through ``chain``.

    ``chain`` maps a native-grid image to the image the pyramid sees (NaN
    where it has no value). Measured over the pixels whose level-l
    coefficient lies at least 4 * 2**l px from a missing value (all finite
    pixels when none does), averaged
    over ``realisations`` seeded draws. For the identity chain this
    reproduces level_noise_gain up to sampling error.
    """
    rng = np.random.default_rng(seed)
    energy, count = np.zeros(levels), np.zeros(levels)
    for _ in range(realisations):
        out = np.asarray(chain(rng.standard_normal(native_shape)), np.float64)
        finite = np.isfinite(out)
        laps = LaplacianPyramid._pyramid(np.where(finite, out, 0.0), levels)
        # Distance to the nearest missing value; the image border is not one
        # (the pyramid reflects there).
        depth = ndimage.distance_transform_edt(finite) if not finite.all() else np.full(finite.shape, np.inf)
        for lv in range(levels):
            step = 2**lv
            crop = (slice(0, laps[lv].shape[0]), slice(0, laps[lv].shape[1]))
            inner = (depth > 4 * step)[::step, ::step][crop]
            if not inner.any():  # a footprint smaller than the margin: use all of it
                inner = finite[::step, ::step][crop]
            energy[lv] += float(np.sum(laps[lv][inner] ** 2))
            count[lv] += int(inner.sum())
    if (count == 0).any():
        raise ValueError("the chain returned no finite pixels")
    return tuple(float(g) for g in np.sqrt(energy / count))


_WARP_GAINS: dict[tuple, tuple[float, ...]] = {}


def _warp_key(band: Band, native_shape: tuple[int, int], warp: Warp, levels: int) -> tuple:
    # The resampling ratio of a fixed camera pair barely changes between
    # samples, so the gain is cached per band and ratio (to 1e-3).
    m = warp.map_xy[warp.support]
    if m.size == 0:
        return (band, native_shape, levels, hashlib.sha1(warp.map_xy.tobytes()).hexdigest())
    dx = np.nanmedian(np.diff(warp.map_xy[..., 0], axis=1)[warp.support[:, 1:]])
    dy = np.nanmedian(np.diff(warp.map_xy[..., 1], axis=0)[warp.support[1:, :]])
    return (band, native_shape, warp.map_xy.shape[:2], levels, round(float(dx), 3), round(float(dy), 3))


def band_level_gains(x: FusionInput, band: Band, levels: int,
                     chains: Mapping[Band, Callable[[np.ndarray], np.ndarray]] | None = None) -> tuple[float, ...]:
    """Per-level noise gain of one band: its native chain (if any), then its warp.

    The reference band without a native chain is white (level_noise_gain).
    A source band is pushed through ``chains[band]`` (identity if absent) and
    the same bilinear warp ``aligned_sources`` uses; without a native chain
    the result is cached per band and resampling ratio.
    """
    native = (chains or {}).get(band)
    shape = np.asarray(x.images[band]).shape[:2]
    if band == x.reference:
        return level_noise_gain(levels) if native is None else chain_level_gains(native, shape, levels)
    warp = x.warps[band]
    pre = native or (lambda n: n)
    run = lambda n: apply_warp(pre(n), warp, "linear")  # noqa: E731
    if native is not None:
        return chain_level_gains(run, shape, levels)
    key = _warp_key(band, shape, warp, levels)
    if key not in _WARP_GAINS:
        _WARP_GAINS[key] = chain_level_gains(run, shape, levels)
    return _WARP_GAINS[key]


def shrink(c: np.ndarray, sigma: Sigma, kappa: float) -> np.ndarray:
    """Soft threshold c * max(0, 1 - kappa*sigma/|c|); 0 where c == 0."""
    c = np.asarray(c, np.float64)
    mag = np.abs(c)
    nonzero = mag > 0
    factor = np.where(nonzero, np.maximum(0.0, 1.0 - kappa * np.asarray(sigma) / np.where(nonzero, mag, 1.0)), 0.0)
    return c * factor


def fuse_coefficients(coeffs: np.ndarray, sigma: np.ndarray, weights: np.ndarray,
                      kappa: float | None) -> np.ndarray:
    """One level: shrink, weight, then take the largest weighted magnitude per pixel.

    coeffs, sigma, weights are (n_bands, H, W); kappa None skips shrinkage.
    Ties go to the first band. Returns the weighted coefficient of the winner.
    """
    shrunk = coeffs if kappa is None else shrink(coeffs, sigma, kappa)
    weighted = weights * shrunk
    idx = np.argmax(np.abs(weighted), axis=0)
    return np.take_along_axis(weighted, idx[None], axis=0)[0]


def _level_sigma(var_map: np.ndarray, level: int, gain: float) -> np.ndarray:
    """Per-pixel sigma of detail level `level` from the full-resolution variance map."""
    if level == 0:
        return np.sqrt(var_map) * gain
    h, w = var_map.shape
    size = (-(-w // 2**level), -(-h // 2**level))
    return np.sqrt(cv2.resize(var_map, size, interpolation=cv2.INTER_AREA)) * gain


def _extend(img: np.ndarray, var: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace pixels outside `valid` by the value of the nearest valid pixel."""
    if valid.all() or not valid.any():
        return img, var
    idx = tuple(ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True))
    return img[idx], var[idx]


def _fuse_masked(x: FusionInput, requested_levels: int, kappa: float | None,
                 sigmas: Mapping[Band, Sigma] | None, gain_model: str = "white",
                 chains: Mapping[Band, Callable[[np.ndarray], np.ndarray]] | None = None) -> FusionOutput:
    images, supports = aligned_sources(x)
    bands = [b for b in BANDS if b in images]
    shape = np.asarray(images[bands[0]]).shape[:2]
    levels = pyramid_levels(shape, requested_levels)
    if sigmas is None or gain_model == "white":
        band_gains = {b: level_noise_gain(levels) for b in bands}
    else:
        band_gains = {b: band_level_gains(x, b, levels, chains) for b in bands}

    laps, weights, sigma_levels = [], [], []
    for b in bands:
        valid = supports[b]
        s = 0.0 if sigmas is None else sigmas[b]
        var = np.broadcast_to(np.asarray(s, np.float64) ** 2, shape).copy()
        img, var = _extend(np.asarray(images[b], np.float64), var, valid)
        laps.append(LaplacianPyramid._pyramid(img, levels))
        weights.append(gaussian_pyramid(valid.astype(np.float64), levels))
        sigma_levels.append([_level_sigma(var, lv, band_gains[b][lv]) for lv in range(levels)])

    fused = []
    for lv in range(levels):
        fused.append(fuse_coefficients(
            np.stack([lap[lv] for lap in laps]),
            np.stack([sl[lv] for sl in sigma_levels]),
            np.stack([w[lv] for w in weights]),
            kappa))
    top_w = np.stack([w[-1] for w in weights])
    top = np.sum(top_w * np.stack([lap[-1] for lap in laps]), axis=0) / np.sum(top_w, axis=0)

    out = top
    for level in reversed(fused):
        out = cv2.pyrUp(out, dstsize=(level.shape[1], level.shape[0])) + level
    extras = {"support": supports, "levels": levels, "kappa": kappa,
              "sigma_level": {b: [float(np.median(a)) for a in sl] for b, sl in zip(bands, sigma_levels)},
              "fused_detail": fused, "gain_model": gain_model if sigmas is not None else None,
              "level_gain": {b: list(g) for b, g in band_gains.items()}}
    # Clipped at 1 only: a dim VIS RAW input carries noise below 0 (VisRawLinear keeps the
    # sub-black side), and clipping the fused image at 0 would rectify it again.
    return FusionOutput(np.minimum(out, 1.0).astype(np.float32), extras)


@register("laplacian_pyramid_masked")
class MaskedLaplacianPyramid(FusionMethod):
    """Max-weighted-coefficient Laplacian fusion with nearest-valid extension and mask weights."""

    capabilities: Capabilities = _CAPS

    def __init__(self, levels: int = 4):
        self.levels = levels

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        return _fuse_masked(x, self.levels, None, None)


@register("noise_aware_pyramid")
class NoiseAwarePyramid(FusionMethod):
    """Masked Laplacian fusion with per-band, per-level soft-threshold shrinkage.

    sigma maps each band to the std of its normalised noise per native
    pixel, given on the reference grid (scalar or (H, W) array: a per-pixel
    map resampled onto the reference grid; see signal.noise.band_sigma) --
    the std before resampling, which the per-level gains then account for.
    gain_model "chain" (default) measures each band's gains through its
    processing chain (``band_level_gains``; ``chains`` gives native chains
    such as signal.noise.vis_luminance_chain), "white" uses
    level_noise_gain for every band. kappa is the shrinkage strength (2 to
    3). noise_info is echoed into extras (spans, fixed-pattern switch, ...)
    for provenance.
    """

    capabilities: Capabilities = _CAPS

    def __init__(self, sigma: Mapping[Band, Sigma], levels: int = 4, kappa: float = 2.5,
                 noise_info: dict | None = None, gain_model: str = "chain",
                 chains: Mapping[Band, Callable[[np.ndarray], np.ndarray]] | None = None):
        if gain_model not in ("chain", "white"):
            raise ValueError(f"gain_model must be 'chain' or 'white', got {gain_model!r}")
        self.gain_model = gain_model
        self.chains = dict(chains) if chains else {}
        self.sigma = dict(sigma)
        self.levels = levels
        self.kappa = float(kappa)
        self.noise_info = dict(noise_info) if noise_info else {}

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        missing = set(x.images) - set(self.sigma)
        if missing:
            raise ValueError(f"{self.name}: no sigma for bands {sorted(missing)}")
        out = _fuse_masked(x, self.levels, self.kappa, self.sigma, self.gain_model, self.chains)
        out.extras["noise_info"] = self.noise_info
        return out

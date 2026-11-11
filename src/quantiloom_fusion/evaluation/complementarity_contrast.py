"""IR radiance contrast across region boundaries: temperature vs emissivity share.

Model (per pixel, one IR band, units W m^-2 sr^-1)::

    L = eps * B(T) + (1 - eps) * L_down  =  eps * (B(T) - L_down) + L_down

B(T) is the Planck radiance integrated over the band [lo, hi] nm (rectangular
band, trapezoid quadrature over wavelength; no detector response curve).
For two regions 1 and 2, dL = L2 - L1 splits *exactly* (no remainder) as::

    dL = mean(eps) * (B2 - B1)            temperature term
       + mean(B - L_down) * (eps2 - eps1) emissivity term

because L is bilinear in (eps, B - L_down). A first-order variant about
region 1 is also reported::

    dL ~ eps1 * B'(T1) * dT + (B(T1) - L_down) * d_eps ;  interaction = remainder
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]

_PLANCK_H = 6.62607015e-34
_LIGHT_C = 299792458.0
_BOLTZMANN_K = 1.380649e-23


def band_radiance(
    temperature_k: FloatArray | float,
    band_nm: tuple[float, float],
    *,
    samples: int = 129,
) -> FloatArray:
    """Band-integrated Planck radiance B(T) in W m^-2 sr^-1 (T > 0 required)."""
    lo, hi = band_nm
    if not 0.0 < lo < hi:
        raise ValueError(f"band_nm must satisfy 0 < lo < hi, got {band_nm}")
    lam = np.linspace(lo, hi, samples) * 1e-9
    t = np.asarray(temperature_k, dtype=np.float64)
    if not np.all(t > 0):
        raise ValueError("temperature must be positive")
    flat = t.reshape(-1)
    out = np.empty_like(flat)
    for start in range(0, flat.size, 16384):
        part = flat[start : start + 16384]
        x = _PLANCK_H * _LIGHT_C / (lam[None, :] * _BOLTZMANN_K * part[:, None])
        spec = 2.0 * _PLANCK_H * _LIGHT_C**2 / lam**5 / np.expm1(x)
        out[start : start + 16384] = np.trapezoid(spec, lam, axis=1)
    return out.reshape(t.shape)


@dataclass(frozen=True)
class ContrastTerms:
    """Region-2 minus region-1 radiance difference and its decompositions."""

    d_radiance: FloatArray
    temperature: FloatArray  # exact symmetric split: temperature + emissivity == d_radiance
    emissivity: FloatArray
    linear_temperature: FloatArray  # first order about region 1
    linear_emissivity: FloatArray
    interaction: FloatArray  # d_radiance - linear_temperature - linear_emissivity


def decompose_contrast(
    t1: FloatArray,
    eps1: FloatArray,
    t2: FloatArray,
    eps2: FloatArray,
    band_nm: tuple[float, float],
    l_down: float = 0.0,
) -> ContrastTerms:
    """Split L2 - L1 into temperature and emissivity contributions (see module doc)."""
    b1 = band_radiance(t1, band_nm)
    b2 = band_radiance(t2, band_nm)
    l1 = eps1 * b1 + (1.0 - eps1) * l_down
    l2 = eps2 * b2 + (1.0 - eps2) * l_down
    d_eps = eps2 - eps1
    temperature = 0.5 * (eps1 + eps2) * (b2 - b1)
    emissivity = (0.5 * (b1 + b2) - l_down) * d_eps
    dt = np.asarray(t2, np.float64) - np.asarray(t1, np.float64)
    db_dt = (band_radiance(t1 + 0.5, band_nm) - band_radiance(t1 - 0.5, band_nm))
    lin_t = eps1 * db_dt * dt
    lin_e = (b1 - l_down) * d_eps
    d_l = l2 - l1
    return ContrastTerms(d_l, temperature, emissivity, lin_t, lin_e, d_l - lin_t - lin_e)


def boundary_pairs(label: IntArray, valid: BoolArray) -> tuple[IntArray, IntArray]:
    """Flat indices (a, b) of 4-neighbour pixel pairs, both valid, with different labels."""
    idx = np.arange(label.size, dtype=np.intp).reshape(label.shape)
    a_parts: list[IntArray] = []
    b_parts: list[IntArray] = []
    for sl_a, sl_b in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        keep = valid[sl_a] & valid[sl_b] & (label[sl_a] != label[sl_b])
        a_parts.append(idx[sl_a][keep])
        b_parts.append(idx[sl_b][keep])
    return np.concatenate(a_parts), np.concatenate(b_parts)


@dataclass(frozen=True)
class BoundaryContrast:
    """Additive sums over boundary pixel pairs (add instances to pool frames)."""

    n_pairs: int = 0
    sum_abs_d_radiance: float = 0.0
    sum_abs_temperature: float = 0.0
    sum_abs_emissivity: float = 0.0
    sum_abs_interaction: float = 0.0  # first-order remainder
    n_temperature_dominant: int = 0

    def __add__(self, other: BoundaryContrast) -> BoundaryContrast:
        return BoundaryContrast(
            self.n_pairs + other.n_pairs,
            self.sum_abs_d_radiance + other.sum_abs_d_radiance,
            self.sum_abs_temperature + other.sum_abs_temperature,
            self.sum_abs_emissivity + other.sum_abs_emissivity,
            self.sum_abs_interaction + other.sum_abs_interaction,
            self.n_temperature_dominant + other.n_temperature_dominant,
        )

    @property
    def temperature_share(self) -> float:
        """sum|temperature term| / (sum|temperature| + sum|emissivity|)."""
        total = self.sum_abs_temperature + self.sum_abs_emissivity
        return self.sum_abs_temperature / total if total > 0 else float("nan")

    @property
    def temperature_dominant_fraction(self) -> float:
        return self.n_temperature_dominant / self.n_pairs if self.n_pairs else float("nan")


def boundary_contrast(
    temperature: FloatArray,
    emissivity: FloatArray,
    label: IntArray,
    valid: BoolArray,
    band_nm: tuple[float, float],
    l_down: float = 0.0,
) -> BoundaryContrast:
    """Decompose dL over every valid cross-label 4-neighbour pair of one frame (2-D inputs)."""
    a, b = boundary_pairs(label, valid)
    if a.size == 0:
        return BoundaryContrast()
    t, e = temperature.reshape(-1), emissivity.reshape(-1)
    c = decompose_contrast(t[a], e[a], t[b], e[b], band_nm, l_down)
    abs_t, abs_e = np.abs(c.temperature), np.abs(c.emissivity)
    return BoundaryContrast(
        int(a.size),
        float(np.abs(c.d_radiance).sum()),
        float(abs_t.sum()),
        float(abs_e.sum()),
        float(np.abs(c.interaction).sum()),
        int((abs_t > abs_e).sum()),
    )

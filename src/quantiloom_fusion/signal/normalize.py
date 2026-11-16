"""Normalizers: explicit, named mappings from product signals to fusion inputs.

Each normalizer names the product it consumes, validates that product's
SignalSemantics (and the observation's Readout where the mapping needs it),
and returns a float32 array. ``params()`` reports the parameters actually
applied on the last call, for provenance in experiment records.

VIS has two switchable inputs: ``VisDisplay`` keeps the sRGB-encoded display
image, ``VisRawLinear`` reconstructs linear RGB from the Bayer mosaic.
IR never uses ``display`` (its stretch is scene-dependent): ``IrLinear``
normalizes the device-linear corrected product, ``IrTemperature`` windows
the apparent-temperature product in Kelvin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..dataset.types import Observation, SignalSemantics
from .colour import demosaic_bilinear, luminance

if TYPE_CHECKING:
    from ..dataset.reader import ThreeBandDataset


@runtime_checkable
class Normalizer(Protocol):
    """Explicit, named mapping from a product image to a fusion input."""

    name: str
    product: str  # "display" | "rawdn" | "corrected" | "tapp" | ...

    def params(self) -> dict: ...

    def __call__(self, image: np.ndarray, observation: Observation) -> np.ndarray: ...


def _sig(observation: Observation, product: str) -> SignalSemantics:
    sig = observation.product(product).signal
    if sig is None:
        raise ValueError(f"{product!r} of {observation.band} carries no signal semantics")
    return sig


def _scale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    denom = hi - lo
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    return np.clip((np.asarray(x, np.float64) - lo) / denom, 0.0, 1.0).astype(np.float32)


class VisDisplay:
    """VIS display product: sRGB-encoded RGB, clipped to [0, 1]."""

    name = "vis_display"
    product = "display"

    def __init__(self) -> None:
        self._applied: dict = {}

    def params(self) -> dict:
        return {"name": self.name, "product": self.product, **self._applied}

    def __call__(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        sig = _sig(observation, self.product)
        if sig.kind != "display_srgb":
            raise ValueError(f"{self.product}: expected kind 'display_srgb', got {sig.kind!r}")
        a = np.asarray(image, np.float32)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"{self.product}: expected a (H, W, 3) sRGB image, got {a.shape}")
        return np.clip(a, 0.0, 1.0)


class VisRawLinear:
    """VIS rawdn -> linear RGB: black level, demosaic, WB, colour matrix, scale.

    scale "adc" divides by the fixed full range (2**adc_bits - 1 - black),
    comparable across samples and conditions; "percentile" divides by the
    given luminance percentile of this image alone.

    The result is clipped at 1 only. Values below the black level stay
    negative: the renderer's pedestal (64 DN for generic_cmos since Quantiloom
    0.10.1) exists so that a dark pixel's negative read-noise excursions are
    kept, and clipping them at 0 here would bias a dim pixel's mean up and its
    variance down exactly where the noise model (signal.noise) is asked about.
    """

    name = "vis_raw_linear"
    product = "rawdn"

    def __init__(self, scale: str = "adc", percentile: float = 99.5, demosaic: str = "bilinear"):
        if scale not in ("adc", "percentile"):
            raise ValueError(f"scale must be 'adc' or 'percentile', got {scale!r}")
        if demosaic != "bilinear":
            raise ValueError(f"only bilinear demosaic is implemented, got {demosaic!r}")
        self.scale = scale
        self.percentile = float(percentile)
        self.demosaic = demosaic
        self._applied: dict = {}

    def params(self) -> dict:
        return {"name": self.name, "product": self.product, "scale": self.scale,
                "percentile": self.percentile, "demosaic": self.demosaic, **self._applied}

    def __call__(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        sig = _sig(observation, self.product)
        if sig.kind != "raw_dn":
            raise ValueError(f"{self.product}: expected kind 'raw_dn', got {sig.kind!r}")
        r = observation.readout
        if r.cfa != "rggb":
            raise ValueError(f"{self.name} expects an rggb mosaic, got cfa {r.cfa!r}")
        rgb = demosaic_bilinear(np.asarray(image, np.float64) - r.black_level_dn, r.cfa)
        rgb = rgb * np.asarray(r.white_balance, np.float64)
        rgb = rgb @ np.asarray(r.device_to_linear_srgb, np.float64).reshape(3, 3).T
        if self.scale == "adc":
            denom = float(2 ** r.adc_bits - 1 - r.black_level_dn)
        else:
            denom = float(np.percentile(luminance(np.clip(rgb, 0.0, None)), self.percentile))
            if not np.isfinite(denom) or denom <= 0:
                denom = 1.0
        self._applied = {"black_level_dn": r.black_level_dn, "scale_value": denom}
        return np.minimum(rgb / denom, 1.0).astype(np.float32)


class IrLinear:
    """IR corrected product: device-linear signal windowed to [0, 1].

    mode "percentile" computes lo/hi per image over finite pixels; mode
    "fixed" uses the given fixed=(lo, hi) window.
    """

    name = "ir_linear"
    product = "corrected"

    def __init__(self, mode: str = "percentile", low: float = 1.0, high: float = 99.0,
                 fixed: tuple[float, float] | None = None):
        if mode not in ("percentile", "fixed"):
            raise ValueError(f"mode must be 'percentile' or 'fixed', got {mode!r}")
        if mode == "fixed" and fixed is None:
            raise ValueError("mode 'fixed' needs fixed=(lo, hi)")
        self.mode = mode
        self.low = float(low)
        self.high = float(high)
        self.fixed = fixed
        self._applied: dict = {}

    def params(self) -> dict:
        return {"name": self.name, "product": self.product, "mode": self.mode,
                "low": self.low, "high": self.high, "fixed": self.fixed, **self._applied}

    def __call__(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        sig = _sig(observation, self.product)
        if sig.kind != "device_linear":
            raise ValueError(f"{self.product}: expected kind 'device_linear', got {sig.kind!r}")
        if observation.readout.cfa != "mono":
            raise ValueError(f"{self.name} expects a mono sensor, got cfa {observation.readout.cfa!r}")
        x = np.asarray(image, np.float64)
        if self.mode == "fixed":
            lo, hi = self.fixed
        else:
            finite = x[np.isfinite(x)]
            lo, hi = np.percentile(finite, [self.low, self.high]) if finite.size else (0.0, 1.0)
        self._applied = {"lo": float(lo), "hi": float(hi)}
        return _scale(x, float(lo), float(hi))


class IrTemperature:
    """IR apparent-temperature product: linear Kelvin window to [0, 1]."""

    name = "ir_temperature"
    product = "tapp"

    def __init__(self, window_k: tuple[float, float] = (260.0, 340.0)):
        if window_k[1] <= window_k[0]:
            raise ValueError(f"window_k must be increasing, got {window_k}")
        self.window_k = (float(window_k[0]), float(window_k[1]))
        self._applied: dict = {}

    def params(self) -> dict:
        return {"name": self.name, "product": self.product, "window_k": self.window_k,
                **self._applied}

    def __call__(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        sig = _sig(observation, self.product)
        if sig.unit != "K":
            raise ValueError(f"{self.product}: expected unit 'K', got {sig.unit!r}")
        self._applied = {"lo": self.window_k[0], "hi": self.window_k[1]}
        return _scale(image, *self.window_k)


def normalized_input(dataset: ThreeBandDataset, observation: Observation,
                     normalizer: Normalizer) -> np.ndarray:
    """Load the normalizer's product for this observation and normalize it.

    The product's declared channel order is requested explicitly: OpenEXR
    stores channels lexicographically, so an unadorned read would return
    B,G,R for an RGB image.
    """
    channels = observation.product(normalizer.product).channels or None
    return normalizer(dataset.load(observation, normalizer.product, channels), observation)

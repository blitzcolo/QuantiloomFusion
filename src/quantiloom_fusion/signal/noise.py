"""Per-band noise std in normalised units, from the imaging-chain model.

The noise parameters are read from the sample's ``camera_config_toml``
provenance text (the same text ``dataset/schema.py`` parses for the readout);
``Readout`` carries only part of them, so this module parses the text itself.

Photon detectors: sigma_e = sqrt(total electrons + read_noise_e_rms^2) with
total electrons = signal + background + dark (NOT the background-subtracted
net signal). Thermal detectors: NETD (K) converted to signal units through
dS/dT, or used directly when the normaliser works in Kelvin. The result is
divided by the span of the normaliser's window, so it is the std of the
normalised image (what a pyramid coefficient is measured in).

Fixed pattern (fpn=True) follows the renderer's camera pipeline
(Quantiloom CpuCameraPipeline): photon detectors have a gain deviation (PRNU
plus column and row gain) acting on the light that reached the pixel, and an
offset (DSNU scaled by exposure / dsnu_reference_exposure_s, plus column and
row offsets); thermal detectors have offsets stated in K (per pixel, column,
row) and a gain deviation acting on the incident power. In the corrected
products (corrected, tapp) a two-point NUC leaves nuc_residual_fraction of
the pattern; rawdn keeps all of it. The column and row parts are counted as
per-pixel variance: exact for the marginal std of one pixel, but a stripe is
fully correlated along its line, so it is not white across pixels and the
per-level pyramid gains (computed for white noise pushed through the
processing chain) are an approximation for it.

Simplifications: the enable_* switches of the config are ignored except
enable_fpn; for VIS the sigma is the per-mosaic-site value -- the fusion's
per-level gains account for demosaic, white balance and the luminance sum
(see vis_luminance_chain).
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..dataset.types import Readout
from .colour import demosaic_bilinear, luminance
from .normalize import IrLinear, IrTemperature, Normalizer, VisRawLinear


@dataclass(frozen=True)
class NoiseConfig:
    """The camera_config_toml fields the noise model reads."""

    detector: str  # "photon" | "thermal"
    exposure_s: float
    electrons_per_dn: float
    black_level_dn: float
    read_noise_e_rms: float
    dark_current_e_s: float
    prnu_sigma: float
    dsnu_e_rms: float
    netd_k: float
    netd_reference_temperature_k: float
    # Fixed pattern of the detector kind in use (0 / off when a dataset's
    # config predates the keys).
    enable_fpn: bool = False
    apply_nuc: bool = False
    nuc_residual_fraction: float = 1.0
    dsnu_reference_exposure_s: float = 0.0  # 0: DSNU taken at the exposure itself
    column_gain_sigma: float = 0.0
    column_offset_e_rms: float = 0.0
    row_gain_sigma: float = 0.0
    row_offset_e_rms: float = 0.0
    fpn_gain_sigma: float = 0.0
    fpn_offset_k: float = 0.0
    fpn_column_gain_sigma: float = 0.0
    fpn_column_offset_k: float = 0.0
    fpn_row_gain_sigma: float = 0.0
    fpn_row_offset_k: float = 0.0


def parse_noise_config(toml_text: str) -> NoiseConfig:
    """Read the noise fields of a camera_config_toml text.

    Reads sensor.detector, sensor.exposure.time_s, sensor.readout.{electrons_per_dn,
    black_level_dn}, sensor.photon.{read_noise_e_rms, dark_current_e_s, prnu_sigma,
    dsnu_e_rms} and sensor.thermal.{netd_k, netd_reference_temperature_k}.
    """
    s = tomllib.loads(toml_text)["sensor"]
    ph, th = s["photon"], s["thermal"]
    own = ph if str(s["detector"]) == "photon" else th
    return NoiseConfig(
        detector=str(s["detector"]),
        exposure_s=float(s["exposure"]["time_s"]),
        electrons_per_dn=float(s["readout"]["electrons_per_dn"]),
        black_level_dn=float(s["readout"]["black_level_dn"]),
        read_noise_e_rms=float(s["photon"]["read_noise_e_rms"]),
        dark_current_e_s=float(s["photon"]["dark_current_e_s"]),
        prnu_sigma=float(s["photon"]["prnu_sigma"]),
        dsnu_e_rms=float(s["photon"]["dsnu_e_rms"]),
        netd_k=float(s["thermal"]["netd_k"]),
        netd_reference_temperature_k=float(s["thermal"]["netd_reference_temperature_k"]),
        enable_fpn=bool(own.get("enable_fpn", False)),
        apply_nuc=bool(own.get("apply_nuc", False)),
        nuc_residual_fraction=float(own.get("nuc_residual_fraction", 1.0)),
        dsnu_reference_exposure_s=float(ph.get("dsnu_reference_exposure_s", 0.0)),
        **{k: float(ph.get(k, 0.0)) for k in ("column_gain_sigma", "column_offset_e_rms",
                                              "row_gain_sigma", "row_offset_e_rms")},
        **{k: float(th.get(k, 0.0)) for k in ("fpn_gain_sigma", "fpn_offset_k", "fpn_column_gain_sigma",
                                              "fpn_column_offset_k", "fpn_row_gain_sigma", "fpn_row_offset_k")},
    )


def sample_noise_config(sample_doc: dict, band: str) -> NoiseConfig:
    """NoiseConfig of one band of a parsed sample.json document."""
    prov = sample_doc["observations"][band]["products"]["measurement"]["provenance"]
    return parse_noise_config(prov["camera_config_toml"])


def total_electrons(signal_e, background_e, cfg: NoiseConfig):
    """signal + background + dark electrons (all collected charge), >= 0."""
    dark = cfg.dark_current_e_s * cfg.exposure_s
    return np.maximum(np.asarray(signal_e, np.float64) + background_e + dark, 0.0)


def electrons_from_raw(raw_dn, cfg: NoiseConfig):
    """Total collected electrons of a raw DN image (it already holds background + dark)."""
    return np.maximum(np.asarray(raw_dn, np.float64) - cfg.black_level_dn, 0.0) * cfg.electrons_per_dn


CORRECTED_PRODUCTS = ("corrected", "tapp")


def pattern_residual(cfg: NoiseConfig, product: str) -> float:
    """Fraction of the fixed pattern left in ``product``: the NUC residual in a corrected product."""
    return cfg.nuc_residual_fraction if (cfg.apply_nuc and product in CORRECTED_PRODUCTS) else 1.0


def photon_sigma_e(total_e, cfg: NoiseConfig, fpn: bool = False, product: str = "rawdn"):
    """Photon-detector noise std in electrons: sqrt(total + read^2 [+ fixed pattern]).

    fpn=True (and enable_fpn in the config) adds, scaled by the residual of
    ``product``, the gain deviation sqrt(prnu^2 + column^2 + row^2) times the
    light electrons (total minus the mean dark charge) and the offsets
    sqrt(dsnu_eff^2 + column^2 + row^2), dsnu_eff = dsnu_e_rms * exposure /
    dsnu_reference_exposure_s.
    """
    total = np.maximum(np.asarray(total_e, np.float64), 0.0)
    var = total + cfg.read_noise_e_rms**2
    if fpn and cfg.enable_fpn:
        k = pattern_residual(cfg, product)
        light = np.maximum(total - cfg.dark_current_e_s * cfg.exposure_s, 0.0)
        ref_t = cfg.dsnu_reference_exposure_s if cfg.dsnu_reference_exposure_s > 0 else cfg.exposure_s
        dsnu = cfg.dsnu_e_rms * cfg.exposure_s / ref_t
        gain2 = cfg.prnu_sigma**2 + cfg.column_gain_sigma**2 + cfg.row_gain_sigma**2
        offset2 = dsnu**2 + cfg.column_offset_e_rms**2 + cfg.row_offset_e_rms**2
        var = var + k**2 * (gain2 * light**2 + offset2)
    return np.sqrt(var)


def thermal_sigma_k(cfg: NoiseConfig, fpn: bool = False, product: str = "tapp", gain_signal_k=None):
    """Thermal-detector noise std in K: NETD [+ fixed pattern].

    fpn=True (and enable_fpn) adds, scaled by the residual of ``product``, the
    offsets sqrt(offset^2 + column^2 + row^2) (K) and, when ``gain_signal_k``
    (the incident power expressed in K, i.e. power / dS/dT) is given, the gain
    deviation sqrt(gain^2 + column^2 + row^2) times it. Without it the gain
    part is left out (it needs the incident power, which the products do not
    carry: the corrected signal has the background subtracted).
    """
    var = np.asarray(cfg.netd_k, np.float64) ** 2
    if fpn and cfg.enable_fpn:
        k = pattern_residual(cfg, product)
        var = var + k**2 * (cfg.fpn_offset_k**2 + cfg.fpn_column_offset_k**2 + cfg.fpn_row_offset_k**2)
        if gain_signal_k is not None:
            gain2 = cfg.fpn_gain_sigma**2 + cfg.fpn_column_gain_sigma**2 + cfg.fpn_row_gain_sigma**2
            var = var + k**2 * gain2 * np.asarray(gain_signal_k, np.float64) ** 2
    return np.sqrt(var)


def vis_luminance_chain(readout: Readout) -> Callable[[np.ndarray], np.ndarray]:
    """The VIS processing VisRawLinear + luminance apply, as a linear map of a mosaic.

    Demosaic, white balance, colour matrix, BT.601 luma; the black level and
    the scale are left out (an offset and the span the sigma is divided by).
    Pushing unit white noise through it gives the luminance grey's noise
    structure for a per-mosaic-site sigma (see fusion.noise_aware).
    """
    wb = np.asarray(readout.white_balance, np.float64)
    m = np.asarray(readout.device_to_linear_srgb, np.float64).reshape(3, 3)

    def chain(mosaic: np.ndarray) -> np.ndarray:
        return luminance((demosaic_bilinear(mosaic, readout.cfa) * wb) @ m.T)

    return chain


def normalisation_span(normalizer: Normalizer, cfg: NoiseConfig) -> float:
    """Span of the normaliser's window, in the unit its sigma must be in.

    VisRawLinear: scale_value (DN) * electrons_per_dn -> electrons. IrLinear:
    hi - lo in the product unit (e- or W). IrTemperature: hi - lo in K. The
    normaliser must have been called (params() holds the applied window).
    """
    p = normalizer.params()
    if isinstance(normalizer, VisRawLinear):
        if "scale_value" not in p:
            raise ValueError("VisRawLinear has not been applied yet; no scale_value")
        return float(p["scale_value"]) * cfg.electrons_per_dn
    if isinstance(normalizer, (IrLinear, IrTemperature)):
        if "lo" not in p:
            raise ValueError(f"{normalizer.name} has not been applied yet; no window")
        return float(p["hi"] - p["lo"])
    raise ValueError(f"{normalizer.name}: not a linear normalisation, no noise span")


def estimate_dsdt(signal, tapp) -> float:
    """dS/dT (signal units per K) from a corrected/tapp image pair of one observation.

    Least-squares slope over the pixels with tapp between its 25th and 75th
    percentile, where the Planck curve is close to its local tangent.
    """
    s = np.asarray(signal, np.float64).ravel()
    t = np.asarray(tapp, np.float64).ravel()
    ok = np.isfinite(s) & np.isfinite(t)
    s, t = s[ok], t[ok]
    lo, hi = np.percentile(t, [25, 75])
    mid = (t >= lo) & (t <= hi)
    slope = np.polyfit(t[mid], s[mid], 1)[0]
    return float(slope)


def band_sigma(cfg: NoiseConfig, normalizer: Normalizer, *, total_e=None,
               dsdt: float | None = None, fpn: bool = False, gain_signal_k=None):
    """Std of the band's normalised image noise, plus the parameters used.

    Photon: pass total_e (scalar or array of total electrons; see
    total_electrons / electrons_from_raw); the result has total_e's shape.
    With IrTemperature also pass dsdt (electrons per K of the corrected
    product, see estimate_dsdt): sigma = sigma_e / |dsdt| / span_K.
    Thermal: with IrTemperature sigma = sigma_K / span_K; with IrLinear pass
    dsdt (signal unit per K): sigma = sigma_K * |dsdt| / span. fpn adds the
    fixed pattern left in the normaliser's product (see photon_sigma_e,
    thermal_sigma_k). Returns (sigma, info) where info records span, unit,
    the switches and the pattern residual used.
    """
    span = normalisation_span(normalizer, cfg)
    if span <= 0:
        raise ValueError(f"non-positive normalisation span {span}")
    product = normalizer.product
    info = {"detector": cfg.detector, "span": span, "fpn": fpn, "normalizer": normalizer.name,
            "pattern_residual": pattern_residual(cfg, product) if fpn and cfg.enable_fpn else None}
    if cfg.detector == "photon":
        if total_e is None:
            raise ValueError("photon detector needs total_e")
        sigma_e = photon_sigma_e(total_e, cfg, fpn, product)
        if isinstance(normalizer, IrTemperature):
            if dsdt is None or dsdt == 0:
                raise ValueError("a photon detector windowed in Kelvin needs dsdt (electrons per K)")
            return sigma_e / abs(dsdt) / span, {**info, "unit": "K", "dsdt": dsdt}
        return sigma_e / span, {**info, "unit": "e-"}
    if cfg.detector == "thermal":
        sigma_k = thermal_sigma_k(cfg, fpn, product, gain_signal_k)
        if isinstance(normalizer, IrTemperature):
            return np.float64(sigma_k / span) if np.ndim(sigma_k) == 0 else sigma_k / span, \
                {**info, "unit": "K", "netd_k": cfg.netd_k}
        if dsdt is None:
            raise ValueError("thermal detector with a signal-domain normaliser needs dsdt")
        return (sigma_k * abs(dsdt) / span,
                {**info, "unit": "signal", "netd_k": cfg.netd_k, "dsdt": dsdt})
    raise ValueError(f"unknown detector {cfg.detector!r}")

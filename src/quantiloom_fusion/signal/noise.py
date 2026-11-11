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

Simplifications: the enable_* switches of the config are ignored (all terms
are counted); for VIS the sigma is the per-mosaic-site value, a conservative
bound for the luminance grey (demosaic and the luminance sum average noise).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

import numpy as np

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


def parse_noise_config(toml_text: str) -> NoiseConfig:
    """Read the noise fields of a camera_config_toml text.

    Reads sensor.detector, sensor.exposure.time_s, sensor.readout.{electrons_per_dn,
    black_level_dn}, sensor.photon.{read_noise_e_rms, dark_current_e_s, prnu_sigma,
    dsnu_e_rms} and sensor.thermal.{netd_k, netd_reference_temperature_k}.
    """
    s = tomllib.loads(toml_text)["sensor"]
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


def photon_sigma_e(total_e, cfg: NoiseConfig, fpn: bool = False):
    """Photon-detector noise std in electrons: sqrt(total + read^2 [+ fixed pattern]).

    fpn=True adds PRNU (prnu_sigma * total) and DSNU (dsnu_e_rms) in quadrature.
    """
    total = np.maximum(np.asarray(total_e, np.float64), 0.0)
    var = total + cfg.read_noise_e_rms**2
    if fpn:
        var = var + (cfg.prnu_sigma * total) ** 2 + cfg.dsnu_e_rms**2
    return np.sqrt(var)


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
               dsdt: float | None = None, fpn: bool = False):
    """Std of the band's normalised image noise, plus the parameters used.

    Photon: pass total_e (scalar or array of total electrons; see
    total_electrons / electrons_from_raw); the result has total_e's shape.
    Thermal: with IrTemperature sigma = netd_k / span_K; with IrLinear pass
    dsdt (signal unit per K, see estimate_dsdt): sigma = netd_k * dsdt / span.
    Returns (sigma, info) where info records span, unit and the switches.
    """
    span = normalisation_span(normalizer, cfg)
    if span <= 0:
        raise ValueError(f"non-positive normalisation span {span}")
    info = {"detector": cfg.detector, "span": span, "fpn": fpn, "normalizer": normalizer.name}
    if cfg.detector == "photon":
        if total_e is None:
            raise ValueError("photon detector needs total_e")
        if isinstance(normalizer, IrTemperature):
            raise ValueError("a photon detector cannot be normalised in Kelvin here")
        return photon_sigma_e(total_e, cfg, fpn) / span, {**info, "unit": "e-"}
    if cfg.detector == "thermal":
        if isinstance(normalizer, IrTemperature):
            return np.float64(cfg.netd_k / span), {**info, "unit": "K", "netd_k": cfg.netd_k}
        if dsdt is None:
            raise ValueError("thermal detector with a signal-domain normaliser needs dsdt")
        return (np.float64(cfg.netd_k * abs(dsdt) / span),
                {**info, "unit": "signal", "netd_k": cfg.netd_k, "dsdt": dsdt})
    raise ValueError(f"unknown detector {cfg.detector!r}")

"""Noise model: photon total-electron formula, thermal NETD, normalisation spans."""

import dataclasses

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.dataset import ThreeBandDataset
from quantiloom_fusion.signal import IrLinear, IrTemperature, VisRawLinear, normalized_input
from quantiloom_fusion.signal.noise import (
    band_sigma,
    electrons_from_raw,
    estimate_dsdt,
    parse_noise_config,
    photon_sigma_e,
    thermal_sigma_k,
    total_electrons,
)

from dataset_fixture import build_fixture

TOML = """
[sensor]
detector = "{detector}"
[sensor.exposure]
time_s = 2
[sensor.readout]
electrons_per_dn = 5
black_level_dn = 10
[sensor.photon]
read_noise_e_rms = 3
dark_current_e_s = 10
prnu_sigma = 0.01
dsnu_e_rms = 4
[sensor.thermal]
netd_k = 0.05
netd_reference_temperature_k = 300
"""


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return ThreeBandDataset(build_fixture(tmp_path_factory.mktemp("ds")))


def test_parse_noise_config():
    cfg = parse_noise_config(TOML.format(detector="photon"))
    assert (cfg.detector, cfg.exposure_s, cfg.electrons_per_dn, cfg.black_level_dn) == ("photon", 2.0, 5.0, 10.0)
    assert (cfg.read_noise_e_rms, cfg.dark_current_e_s, cfg.prnu_sigma, cfg.dsnu_e_rms) == (3.0, 10.0, 0.01, 4.0)
    assert (cfg.netd_k, cfg.netd_reference_temperature_k) == (0.05, 300.0)


def test_photon_sigma_uses_total_electrons_not_net_signal():
    cfg = parse_noise_config(TOML.format(detector="photon"))
    total = total_electrons(1000.0, 500.0, cfg)
    assert total == pytest.approx(1000.0 + 500.0 + 10.0 * 2.0)
    assert photon_sigma_e(total, cfg) == pytest.approx(np.sqrt(1520.0 + 9.0))
    assert photon_sigma_e(total, cfg) > photon_sigma_e(1000.0, cfg) * 1.2
    assert electrons_from_raw(np.array([10.0, 110.0]), cfg).tolist() == [0.0, 500.0]


FPN = TOML.replace("[sensor.photon]\n", """[sensor.photon]
apply_nuc = true
nuc_residual_fraction = 0.03
enable_fpn = true
dsnu_reference_exposure_s = 1
column_gain_sigma = 0.02
column_offset_e_rms = 3
row_gain_sigma = 0.01
row_offset_e_rms = 2
""")
THERMAL_FPN = TOML.replace("[sensor.thermal]\n", """[sensor.thermal]
enable_fpn = true
apply_nuc = true
nuc_residual_fraction = 0.03
fpn_gain_sigma = 0.016
fpn_offset_k = 1.1
fpn_column_gain_sigma = 0.023
fpn_column_offset_k = 1.55
fpn_row_gain_sigma = 0.01
fpn_row_offset_k = 0.65
""")


def test_fixed_pattern_noise_is_a_switch_that_reads_every_component():
    assert photon_sigma_e(1000.0, parse_noise_config(TOML.format(detector="photon")), fpn=True) == \
        pytest.approx(np.sqrt(1009.0))  # config without enable_fpn: no pattern
    cfg = parse_noise_config(FPN.format(detector="photon"))
    off = photon_sigma_e(1000.0, cfg)
    # Light = total - mean dark (10 e/s * 2 s); DSNU scales with exposure/reference (2/1).
    pattern = (0.01**2 + 0.02**2 + 0.01**2) * 980.0**2 + (4.0 * 2.0) ** 2 + 3.0**2 + 2.0**2
    raw = photon_sigma_e(1000.0, cfg, fpn=True, product="rawdn")
    corrected = photon_sigma_e(1000.0, cfg, fpn=True, product="corrected")
    assert raw == pytest.approx(np.sqrt(1009.0 + pattern))
    assert corrected == pytest.approx(np.sqrt(1009.0 + 0.03**2 * pattern))
    assert off < corrected < raw


def test_bias_pattern_is_in_rawdn_only_and_gated_by_enable_fpn():
    biased = FPN.replace("[sensor.photon]\n", "[sensor.photon]\nbias_dn_rms = 0.4\n")
    cfg = parse_noise_config(biased.format(detector="photon"))
    assert cfg.bias_dn_rms == 0.4 and parse_noise_config(FPN.format(detector="photon")).bias_dn_rms == 0.0
    plain = parse_noise_config(FPN.format(detector="photon"))
    bias_e2 = (0.4 * 5.0) ** 2  # DN -> e- at 5 e-/DN
    for product in ("rawdn", "corrected", "tapp"):
        base = photon_sigma_e(1000.0, plain, fpn=True, product=product) ** 2
        got = photon_sigma_e(1000.0, cfg, fpn=True, product=product) ** 2
        # The renderer subtracts each pixel's bias exactly when it forms the corrected signal.
        assert got == pytest.approx(base + (bias_e2 if product == "rawdn" else 0.0))
    assert photon_sigma_e(1000.0, cfg, fpn=False) == photon_sigma_e(1000.0, plain, fpn=False)
    off = parse_noise_config(TOML.replace("[sensor.photon]\n", "[sensor.photon]\nbias_dn_rms = 0.4\n")
                             .format(detector="photon"))
    assert photon_sigma_e(1000.0, off, fpn=True) == pytest.approx(np.sqrt(1009.0))  # enable_fpn off


SYNTH = """
[sensor]
detector = "photon"
[sensor.exposure]
time_s = 1.0
[sensor.readout]
electrons_per_dn = 5.0
black_level_dn = 64
analog_gain = 1.0
adc_bits = 12
[sensor.photon]
read_noise_e_rms = 3.0
dark_current_e_s = 10.0
prnu_sigma = 0.005
dsnu_e_rms = 0.05
dsnu_reference_exposure_s = 0.01
bias_dn_rms = 0.5
enable_fpn = true
apply_nuc = false
nuc_residual_fraction = 1.0
[sensor.thermal]
netd_k = 0.0
netd_reference_temperature_k = 300
"""


def _synthetic_raw(cfg, light_e, n, rng):
    """A low-light photon readout drawn the way Quantiloom 0.10.1's CpuCameraPipeline draws it.

    Per pixel: PRNU gain, a lognormal dark charge with the mean dark charge and the DSNU at the
    exposure as its mean and rms, Poisson light and dark shot noise, Gaussian read noise, a fixed
    bias in DN, the black level, rounding and the ADC clamp at [0, 2^12 - 1].
    """
    mean_dark = cfg.dark_current_e_s * cfg.exposure_s
    dsnu = cfg.dsnu_e_rms * cfg.exposure_s / cfg.dsnu_reference_exposure_s
    s2 = np.log1p((dsnu / mean_dark) ** 2)
    dark = mean_dark * np.exp(np.sqrt(s2) * rng.standard_normal(n) - 0.5 * s2)
    light = light_e * (1.0 + cfg.prnu_sigma * rng.standard_normal(n))
    electrons = rng.poisson(light) + rng.poisson(dark) + cfg.read_noise_e_rms * rng.standard_normal(n)
    dn = electrons / cfg.electrons_per_dn + cfg.black_level_dn + cfg.bias_dn_rms * rng.standard_normal(n)
    return np.clip(np.floor(dn + 0.5), 0.0, 4095.0)


def test_noise_model_matches_a_synthetic_readout_on_a_black_level():
    cfg = parse_noise_config(SYNTH)
    rng = np.random.default_rng(20261116)
    light, n = 20.0, 400_000
    raw = _synthetic_raw(cfg, light, n, rng)
    assert raw.min() > 0.0  # the 64 DN pedestal keeps the dark side off the ADC floor
    e = electrons_from_raw(raw, cfg)
    total = light + cfg.dark_current_e_s * cfg.exposure_s
    assert e.mean() == pytest.approx(total, abs=0.1)
    quantisation = cfg.electrons_per_dn**2 / 12.0  # rounding to DN; not part of the model
    model = photon_sigma_e(total, cfg, fpn=True, product="rawdn") ** 2
    assert e.var() == pytest.approx(model + quantisation, rel=0.03)
    # Leaving the bias pattern out would miss the measured variance by more than the tolerance.
    no_bias = photon_sigma_e(total, dataclasses.replace(cfg, bias_dn_rms=0.0), fpn=True) ** 2
    assert abs(e.var() - (no_bias + quantisation)) > 0.06 * e.var()


def test_thermal_fixed_pattern_counts_offsets_and_optionally_the_gain():
    cfg = parse_noise_config(THERMAL_FPN.format(detector="thermal"))
    offsets = 1.1**2 + 1.55**2 + 0.65**2
    assert thermal_sigma_k(cfg) == pytest.approx(0.05)
    assert thermal_sigma_k(cfg, fpn=True) == pytest.approx(np.sqrt(0.05**2 + 0.03**2 * offsets))
    assert thermal_sigma_k(cfg, fpn=True, product="rawdn") == pytest.approx(np.sqrt(0.05**2 + offsets))
    gain = 0.016**2 + 0.023**2 + 0.01**2
    assert thermal_sigma_k(cfg, fpn=True, gain_signal_k=75.0) == \
        pytest.approx(np.sqrt(0.05**2 + 0.03**2 * (offsets + gain * 75.0**2)))


def test_photon_detector_windowed_in_kelvin_converts_through_dsdt(dataset):
    cfg = parse_noise_config(TOML.format(detector="photon"))
    obs = next(iter(dataset)).observations[Band.LWIR]
    kelvin = IrTemperature(window_k=(260.0, 340.0))
    normalized_input(dataset, obs, kelvin)
    sigma, info = band_sigma(cfg, kelvin, total_e=np.array([500.0]), dsdt=-50.0)
    assert sigma[0] == pytest.approx(np.sqrt(509.0) / 50.0 / 80.0) and info["unit"] == "K"
    with pytest.raises(ValueError, match="dsdt"):
        band_sigma(cfg, kelvin, total_e=np.array([500.0]))


def test_vis_sigma_divides_by_the_adc_span(dataset):
    cfg = parse_noise_config(TOML.format(detector="photon"))
    obs = next(iter(dataset)).observations[Band.VIS]
    norm = VisRawLinear(scale="adc")
    normalized_input(dataset, obs, norm)
    span_e = norm.params()["scale_value"] * cfg.electrons_per_dn
    sigma, info = band_sigma(cfg, norm, total_e=np.array([500.0]))
    assert sigma[0] == pytest.approx(np.sqrt(500.0 + 9.0) / span_e)
    assert info["span"] == pytest.approx(span_e) and info["fpn"] is False


def test_thermal_sigma_from_netd(dataset):
    cfg = parse_noise_config(TOML.format(detector="thermal"))
    obs = next(iter(dataset)).observations[Band.LWIR]
    kelvin = IrTemperature(window_k=(260.0, 340.0))
    normalized_input(dataset, obs, kelvin)
    sigma, _ = band_sigma(cfg, kelvin)
    assert sigma == pytest.approx(0.05 / 80.0)

    linear = IrLinear(mode="fixed", fixed=(0.0, 2.0))
    normalized_input(dataset, obs, linear)
    sigma, info = band_sigma(cfg, linear, dsdt=0.4)
    assert sigma == pytest.approx(0.05 * 0.4 / 2.0) and info["dsdt"] == 0.4
    with pytest.raises(ValueError, match="dsdt"):
        band_sigma(cfg, linear)


def test_estimate_dsdt_recovers_slope():
    temp = np.linspace(260.0, 340.0, 4000)
    assert estimate_dsdt(0.4 * (temp - 260.0) + 3.0, temp) == pytest.approx(0.4)

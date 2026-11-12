"""Noise model: photon total-electron formula, thermal NETD, normalisation spans."""

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

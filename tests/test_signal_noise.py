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


def test_fixed_pattern_noise_is_a_switch():
    cfg = parse_noise_config(TOML.format(detector="photon"))
    off = photon_sigma_e(1000.0, cfg)
    on = photon_sigma_e(1000.0, cfg, fpn=True)
    assert on == pytest.approx(np.sqrt(1000.0 + 9.0 + (0.01 * 1000.0) ** 2 + 16.0))
    assert on > off


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

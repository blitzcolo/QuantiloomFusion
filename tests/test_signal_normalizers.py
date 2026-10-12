"""Normalizers on the dataset fixture: ranges, params and validation."""

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.dataset import (
    CameraGeometry,
    Observation,
    Product,
    SignalSemantics,
    ThreeBandDataset,
)
from quantiloom_fusion.signal import (
    IrLinear,
    IrTemperature,
    Normalizer,
    VisDisplay,
    VisRawLinear,
    normalized_input,
)

from dataset_fixture import build_fixture
from test_dataset_types import GEO_KWARGS, READOUT


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return ThreeBandDataset(build_fixture(tmp_path_factory.mktemp("ds")))


def test_implementations_satisfy_protocol():
    for n in (VisDisplay(), VisRawLinear(), IrLinear(), IrTemperature()):
        assert isinstance(n, Normalizer)
        assert n.product in ("display", "rawdn", "corrected", "tapp")


def test_vis_display(dataset):
    obs = next(iter(dataset)).observations[Band.VIS]
    out = normalized_input(dataset, obs, VisDisplay())
    assert out.dtype == np.float32 and out.shape == (6, 8, 3)
    assert np.allclose(out, [0.2, 0.4, 0.6])
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_vis_raw_linear_adc(dataset):
    obs = next(iter(dataset)).observations[Band.VIS]
    n = VisRawLinear(scale="adc")
    out = normalized_input(dataset, obs, n)
    assert out.dtype == np.float32 and out.shape == (6, 8, 3)
    # Constant-colour mosaic demosaics to a flat linear RGB field.
    assert np.allclose(out, np.array([400.0, 600.0, 300.0]) / 4095.0, atol=1e-6)
    assert n.params()["scale_value"] == pytest.approx(4095.0)


def test_vis_raw_linear_percentile(dataset):
    obs = next(iter(dataset)).observations[Band.VIS]
    n = VisRawLinear(scale="percentile")
    out = normalized_input(dataset, obs, n)
    # Constant image: p99.5 = the constant luminance 400*.299+600*.587+300*.114.
    assert n.params()["scale_value"] == pytest.approx(506.0)
    assert np.all(out[..., 1] == 1.0)  # green is above the luma scale and clips


def test_ir_linear_percentile(dataset):
    obs = next(iter(dataset)).observations[Band.LWIR]
    n = IrLinear(mode="percentile", low=1.0, high=99.0)
    out = normalized_input(dataset, obs, n)
    assert out.dtype == np.float32 and out.shape == (4, 5)
    # corrected is a 0..100 ramp over 5 columns x 4 rows: p1=0, p99=100.
    assert n.params()["lo"] == pytest.approx(0.0)
    assert n.params()["hi"] == pytest.approx(100.0)
    assert out.min() == 0.0 and out.max() == 1.0


def test_ir_linear_fixed(dataset):
    obs = next(iter(dataset)).observations[Band.MWIR]
    n = IrLinear(mode="fixed", fixed=(0.0, 100.0))
    out = normalized_input(dataset, obs, n)
    assert n.params()["lo"] == 0.0 and n.params()["hi"] == 100.0
    assert np.allclose(out[0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_ir_temperature(dataset):
    obs = next(iter(dataset)).observations[Band.LWIR]
    n = IrTemperature(window_k=(260.0, 340.0))
    out = normalized_input(dataset, obs, n)
    assert np.allclose(out, (295.0 - 260.0) / 80.0)
    assert n.params()["lo"] == 260.0 and n.params()["hi"] == 340.0


def test_validation_rejects_wrong_semantics(dataset):
    obs = next(iter(dataset)).observations[Band.LWIR]
    vis = next(iter(dataset)).observations[Band.VIS]
    # VisRawLinear needs an rggb readout: the mono lwir observation fails.
    with pytest.raises(ValueError, match="rggb"):
        VisRawLinear()(dataset.load(obs, "rawdn"), obs)
    # IrLinear needs a mono sensor: the rggb vis observation fails.
    with pytest.raises(ValueError, match="mono"):
        IrLinear()(dataset.load(vis, "corrected"), vis)
    # A product with the wrong kind fails VisDisplay's kind check.
    bad_sig = SignalSemantics(kind="raw_dn", unit="DN", transfer="linear", cfa="1",
                              channels=("Raw",), colour_space="device_native", storage="float32")
    bad = Observation(band=Band.VIS, geometry=CameraGeometry(**GEO_KWARGS), readout=READOUT,
                      spectral_mode="m",
                      products={"display": Product("display", "f.exr", bad_sig),
                                "tapp": Product("tapp", "g.exr", bad_sig)},
                      record="r")
    with pytest.raises(ValueError, match="display_srgb"):
        VisDisplay()(np.zeros((2, 2, 3), np.float32), bad)
    with pytest.raises(ValueError, match="'K'"):
        IrTemperature()(np.zeros((2, 2), np.float32), bad)


def test_constructor_validation():
    with pytest.raises(ValueError, match="scale"):
        VisRawLinear(scale="max")
    with pytest.raises(ValueError, match="fixed"):
        IrLinear(mode="fixed")
    with pytest.raises(ValueError, match="window"):
        IrTemperature(window_k=(340.0, 260.0))
    assert "scale_value" not in VisRawLinear().params()

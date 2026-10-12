"""Band constants must mirror the dataset generator's band set."""

from quantiloom_fusion.core.bands import BANDS, Band
from quantiloom_fusion.datagen.constants import BANDS as GENERATOR_BANDS


def test_band_values():
    assert [b.value for b in Band] == ["vis", "mwir", "lwir"]


def test_bands_match_generator():
    assert tuple(BANDS) == tuple(GENERATOR_BANDS)


def test_band_str_is_value():
    assert f"{Band.VIS}" == "vis"
    assert Band.MWIR == "mwir"

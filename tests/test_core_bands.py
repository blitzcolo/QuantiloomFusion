"""Band constants must mirror the dataset generator's band set."""

import importlib.util
from pathlib import Path

from quantiloom_fusion.core.bands import BANDS, Band

GENERATOR = Path(__file__).resolve().parents[1] / "tools" / "generate_env_dataset.py"
spec = importlib.util.spec_from_file_location("generator", GENERATOR)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def test_band_values():
    assert [b.value for b in Band] == ["vis", "mwir", "lwir"]


def test_bands_match_generator():
    assert tuple(BANDS) == tuple(g.BANDS)


def test_band_str_is_value():
    assert f"{Band.VIS}" == "vis"
    assert Band.MWIR == "mwir"

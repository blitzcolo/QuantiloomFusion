"""Tests for dataset.exr: channel IO round-trips and shape handling."""

import os
from pathlib import Path

import numpy as np
import pytest

from quantiloom_fusion.core import jsonio
from quantiloom_fusion.dataset import exr

DATASET = os.environ.get("QUANTILOOM_TEST_DATASET")


def test_round_trip_float32_and_uint32(tmp_path):
    channels = {
        "Y": np.arange(48, dtype=np.float32).reshape(6, 8) * 0.5,
        "id": np.arange(48, dtype=np.uint32).reshape(6, 8),
    }
    path = tmp_path / "t.exr"
    exr.write_image(path, channels, {"author": "fixture"})
    back = exr.read_channels(path)
    assert back["Y"].dtype == np.float32 and back["id"].dtype == np.uint32
    np.testing.assert_array_equal(back["Y"], channels["Y"])
    np.testing.assert_array_equal(back["id"], channels["id"])
    assert exr.read_header(path)["author"] == "fixture"


def test_single_channel_is_2d_multi_channel_is_stacked(tmp_path):
    path = tmp_path / "t.exr"
    exr.write_image(path, {n: np.full((4, 5), v, dtype=np.float32) for n, v in (("R", 1.0), ("G", 2.0), ("B", 3.0))})
    one = exr.read_image(path, ["G"])
    assert one.shape == (4, 5) and np.all(one == 2.0)
    stacked = exr.read_image(path, ["R", "G", "B"])
    assert stacked.shape == (4, 5, 3)
    np.testing.assert_array_equal(stacked[..., 2], np.full((4, 5), 3.0))
    # default order is the header channel list (alphabetical)
    assert exr.read_image(path).shape == (4, 5, 3)
    assert exr.read_image(path)[..., 0].mean() == 3.0  # B sorts first


def test_missing_channel_keyerror_lists_available(tmp_path):
    path = tmp_path / "t.exr"
    exr.write_image(path, {"Y": np.zeros((2, 2), dtype=np.float32)})
    with pytest.raises(KeyError, match="Y"):
        exr.read_channels(path, ["nope"])


@pytest.mark.skipif(not DATASET, reason="set QUANTILOOM_TEST_DATASET for real export tests")
def test_real_dataset_exr_shapes():
    root = Path(DATASET).resolve()
    sample = jsonio.load_json(root / "samples" / "concrete_a000_e00_r00_n00" / "sample.json")
    obs = sample["observations"]["lwir"]
    geom = obs["products"]["measurement"]["provenance"]["geometry"]
    rawdn = exr.read_image(root / obs["products"]["rawdn"]["path"])
    assert rawdn.shape == (geom["height"], geom["width"])
    truth = exr.read_image(root / "samples" / "concrete_a000_e00_r00_n00" / "lwir" / "v0001" / "frame_truth_material_id.exr")
    assert truth.shape == (geom["height"], geom["width"]) and truth.dtype == np.uint32
    display = exr.read_image(root / obs["products"]["display"]["path"])
    assert display.shape == (geom["height"], geom["width"], 3)

"""ThreeBandDataset over the tiny fixture: iteration, product IO, safety."""

import os
from pathlib import Path

import numpy as np
import pytest

from quantiloom_fusion.core import jsonio
from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.dataset import Product, ThreeBandDataset

from dataset_fixture import build_fixture

DATASET = os.environ.get("QUANTILOOM_TEST_DATASET")


def test_reader_iteration_and_shapes(tmp_path):
    root = build_fixture(tmp_path / "ds", samples=2)
    ds = ThreeBandDataset(root)
    assert len(ds) == 2
    assert ds.sample_ids() == ("fixture_s00", "fixture_s01")
    assert [s.sample_id for s in ds] == list(ds.sample_ids())
    s = ds.sample("fixture_s00")
    for band, (w, h) in {Band.VIS: (8, 6), Band.MWIR: (6, 5), Band.LWIR: (5, 4)}.items():
        rawdn = ds.load(s.observations[band], "rawdn")
        assert rawdn.shape == (h, w) and rawdn.dtype == np.float32


def test_material_id_and_instance_id_are_distinct_products(tmp_path):
    # Both files carry a channel named "instance_id"; products are named by
    # file, not by channel.
    root = build_fixture(tmp_path / "ds")
    ds = ThreeBandDataset(root)
    lwir = ds.sample("fixture_s00").observations[Band.LWIR]
    iid = ds.load(lwir, "truth_instance_id")
    mid = ds.load(lwir, "truth_material_id")
    assert iid.dtype == np.uint32 and mid.dtype == np.uint32
    assert np.all(iid == 1) and np.all(mid == 2)
    mats = ds.load_json(lwir, "truth_materials")
    assert mats[0]["material_id"] == 2


def test_unsafe_product_paths_rejected(tmp_path):
    ds = ThreeBandDataset(build_fixture(tmp_path / "ds"))
    for bad in ("../escape.exr", "a/../b.exr", "abs\\win.exr", "/abs.exr", "C:/abs.exr"):
        with pytest.raises(ValueError):
            ds.path(Product("x", bad))


def test_load_rejects_non_exr(tmp_path):
    ds = ThreeBandDataset(build_fixture(tmp_path / "ds"))
    lwir = ds.sample("fixture_s00").observations[Band.LWIR]
    with pytest.raises(ValueError, match="EXR"):
        ds.load(lwir, "truth_materials")


def test_dataset_id_mismatches_detected(tmp_path):
    root = build_fixture(tmp_path / "ds")
    index = jsonio.load_json(root / "dataset.index.json")
    index["dataset_id"] = "other"
    jsonio.atomic_json(root / "dataset.index.json", index)
    with pytest.raises(ValueError, match="dataset_id"):
        ThreeBandDataset(root)

    root2 = build_fixture(tmp_path / "ds2")
    manifest = root2 / "samples" / "fixture_s00" / "sample.json"
    doc = jsonio.load_json(manifest)
    doc["dataset_id"] = "other"
    jsonio.atomic_json(manifest, doc)
    ds = ThreeBandDataset(root2)
    with pytest.raises(ValueError, match="dataset_id"):
        ds.sample("fixture_s00")


def test_reader_references_load_on_the_vis_grid(tmp_path):
    ds = ThreeBandDataset(build_fixture(tmp_path / "ds", samples=1, references=True))
    assert [e.references for e in ds.index.entries] == [True]
    vis = ds.sample("fixture_s00").observations[Band.VIS]
    refs = ds.references("fixture_s00")
    assert set(refs) == {Band.VIS, Band.MWIR, Band.LWIR}
    for band, ref in refs.items():
        assert ref.geometry == vis.geometry
        assert ds.load(ref, "rawdn").shape == (vis.geometry.height, vis.geometry.width)
        assert f"/ref_{band}/" in ref.product("rawdn").path


def test_reader_references_absent_for_v1(tmp_path):
    ds = ThreeBandDataset(build_fixture(tmp_path / "ds", samples=1))
    assert ds.sample("fixture_s00").references == {}
    assert [e.references for e in ds.index.entries] == [False]
    with pytest.raises(KeyError, match="no co-view references"):
        ds.references("fixture_s00")


@pytest.mark.skipif(not DATASET, reason="set QUANTILOOM_TEST_DATASET for real dataset tests")
def test_real_dataset_reader():
    ds = ThreeBandDataset(DATASET)
    for s in ds:
        lwir = s.observations[Band.LWIR]
        tapp = ds.load(lwir, "tapp")
        ts = ds.load(lwir, "truth_surface_temperature")
        assert tapp.shape == ts.shape == (lwir.geometry.height, lwir.geometry.width)

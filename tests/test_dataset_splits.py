"""split_group fold tests on the fixture plus a gated real-data check."""

import os

import pytest

from quantiloom_fusion.dataset import ThreeBandDataset, leave_one_group_out

from dataset_fixture import build_fixture

DATASET = os.environ.get("QUANTILOOM_TEST_DATASET")


def test_leave_one_group_out_two_groups(tmp_path):
    ds = ThreeBandDataset(build_fixture(tmp_path / "ds", samples=4, groups=["g0", "g1"]))
    splits = leave_one_group_out(ds)
    assert [g for _, _, g in splits] == ["g0", "g1"]  # first-appearance order
    train, test, group = splits[0]
    assert group == "g0"
    assert test == ("fixture_s00", "fixture_s02")
    assert train == ("fixture_s01", "fixture_s03")
    for train_ids, test_ids, _ in splits:
        assert not set(train_ids) & set(test_ids)
        assert set(train_ids) | set(test_ids) == set(ds.sample_ids())


def test_leave_one_group_out_single_group(tmp_path):
    ds = ThreeBandDataset(build_fixture(tmp_path / "ds", samples=3))
    (train, test, group), = leave_one_group_out(ds)
    assert group == "fixture-group"
    assert train == () and test == ds.sample_ids()


@pytest.mark.skipif(not DATASET, reason="QUANTILOOM_TEST_DATASET unset")
def test_real_dataset_single_shared_group():
    ds = ThreeBandDataset(DATASET)
    splits = leave_one_group_out(ds)
    assert len(splits) == 1  # day/night/moonlight share one glTF: one group
    train, test, group = splits[0]
    assert train == () and len(test) == len(ds)

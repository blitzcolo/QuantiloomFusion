"""Interface tests for Metric, ResultKey, and ResultStore."""

import dataclasses

import pytest

from quantiloom_fusion.evaluation import Metric, ResultKey, ResultStore


class ZeroMetric(Metric):
    name = "zero"
    version = 1

    def __call__(self, out, x, sample):
        return {"zero": 0.0}


def test_metric_is_abstract():
    with pytest.raises(TypeError):
        Metric()


def test_metric_subclass_callable():
    assert ZeroMetric()(None, None, None) == {"zero": 0.0}


def test_result_key_is_frozen():
    key = ResultKey("ds", "s1", "fp", "zero", 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        key.metric = "other"


def test_result_store_roundtrip(tmp_path):
    with ResultStore(tmp_path / "results.jsonl") as store:
        key = ResultKey("ds", "s1", "fp", "zero", 1)
        assert store.path == tmp_path / "results.jsonl"
        assert not store.has(key)
        store.append(key, {"zero": 0.0})
        assert store.has(key)
        assert store.get(key) == {"zero": 0.0}
        with pytest.raises(ValueError, match="duplicate"):
            store.append(key, {"zero": 1.0})

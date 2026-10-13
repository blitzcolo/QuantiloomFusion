"""ResultStore: JSONL persistence, crash-tail repair, NaN handling, locking."""

import json
import os

import numpy as np
import pytest

from quantiloom_fusion.evaluation import ResultKey, ResultStore


def _key(i=0):
    return ResultKey("ds", f"s{i}", "fp", "EN", 1)


def _line(key, values):
    return (json.dumps({"key": {"dataset_id": key.dataset_id,
                                "sample_id": key.sample_id,
                                "method_fingerprint": key.method_fingerprint,
                                "metric": key.metric,
                                "metric_version": key.metric_version},
                        "values": values},
                       sort_keys=True, separators=(",", ":")) + "\n")


def test_append_writes_one_compact_lf_line(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultStore(path) as store:
        store.append(_key(), {"EN": 1.5})
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n") and raw.count(b"\n") == 1
    rec = json.loads(raw)
    assert rec["key"]["metric"] == "EN" and rec["values"] == {"EN": 1.5}


def test_reload_sees_appended_rows(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultStore(path) as store:
        store.append(_key(0), {"EN": 1.0})
        store.append(_key(1), {"EN": 2.0})
    with ResultStore(path) as store:
        assert store.get(_key(0)) == {"EN": 1.0}
        assert store.get(_key(1)) == {"EN": 2.0}
        assert store.get(_key(2)) is None
        assert len(list(store.rows())) == 2


def test_duplicate_append_raises(tmp_path):
    with ResultStore(tmp_path / "r.jsonl") as store:
        store.append(_key(), {"EN": 1.0})
        with pytest.raises(ValueError, match="duplicate"):
            store.append(_key(), {"EN": 9.0})


def test_nan_and_inf_written_as_null_and_read_as_nan(tmp_path):
    path = tmp_path / "r.jsonl"
    with ResultStore(path) as store:
        store.append(_key(), {"good": 1.0, "nan": float("nan"),
                              "inf": float("inf"), "ninf": float("-inf")})
    rec = json.loads(path.read_text())
    assert rec["values"] == {"good": 1.0, "nan": None, "inf": None, "ninf": None}
    with ResultStore(path) as store:
        got = store.get(_key())
        assert got["good"] == 1.0
        assert np.isnan(got["nan"]) and np.isnan(got["inf"]) and np.isnan(got["ninf"])


def test_truncated_tail_is_repaired_on_open(tmp_path):
    path = tmp_path / "r.jsonl"
    good = _line(_key(0), {"EN": 1.0})
    path.write_bytes(good.encode() + b'{"key": {"dataset_id": "ds", "sam')
    with ResultStore(path) as store:
        # The torn row is gone and appended rows land cleanly after it.
        assert not store.has(_key(1))
        store.append(_key(1), {"EN": 2.0})
    assert path.read_bytes() == good.encode() + _line(_key(1), {"EN": 2.0}).encode()


def test_corrupt_middle_line_raises_with_line_number(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_bytes(_line(_key(0), {"EN": 1.0}).encode()
                     + b"not json\n"
                     + _line(_key(1), {"EN": 2.0}).encode())
    with pytest.raises(ValueError, match=":2:"):
        ResultStore(path)


def test_rows_iterates_in_file_order(tmp_path):
    with ResultStore(tmp_path / "r.jsonl") as store:
        for i in range(3):
            store.append(_key(i), {"EN": float(i)})
        assert [k.sample_id for k, _ in store.rows()] == ["s0", "s1", "s2"]


def test_missing_file_is_empty_not_error(tmp_path):
    with ResultStore(tmp_path / "deep" / "r.jsonl") as store:
        assert store.get(_key()) is None


def test_second_writer_is_rejected(tmp_path):
    with ResultStore(tmp_path / "r.jsonl"):
        with pytest.raises(ValueError):
            ResultStore(tmp_path / "r.jsonl")


def test_append_after_close_raises(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    store.close()
    with pytest.raises(ValueError):
        store.append(_key(), {"EN": 1.0})

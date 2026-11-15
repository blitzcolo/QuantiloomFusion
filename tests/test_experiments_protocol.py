"""Protocol runner: strict config, layered rows in the store, resume after interruption."""

import json

import pytest
from dataset_fixture import build_fixture

from quantiloom_fusion.evaluation.base import ResultStore
from quantiloom_fusion.evaluation.layer_report import REPORT_GROUPS
from quantiloom_fusion.experiments.protocol_run import load_protocol_config, run_protocol

CONFIG = """name = "protocol-fixture"
dataset = "{dataset}"
bands = ["vis", "lwir"]
vis_input = "raw_adc"
ir_input = "temperature"
items = ["matrix"]
registration = ["oracle", "infinity_homography"]
operators = ["laplacian_pyramid_masked"]
"""


def _setup(tmp_path, extra=""):
    root = build_fixture(tmp_path / "ds", samples=2, size=(32, 24), references=True)
    config = tmp_path / "protocol.toml"
    config.write_text(CONFIG.format(dataset=root.as_posix()) + extra, encoding="utf-8")
    return config, tmp_path / "out"


def _rows(out):
    with ResultStore(out / "protocol-fixture" / "results.jsonl") as store:
        return {k: v for k, v in store.rows()}


def test_matrix_rows_cover_registrations_truth_modes_and_layers(tmp_path):
    config, out = _setup(tmp_path)
    assert run_protocol(config, out_root=out) == 0
    rows = _rows(out)
    # 2 samples x 2 registrations x 1 operator x 3 truth modes (none + both, references present) x 6 layer groups
    assert len(rows) == 2 * 2 * 3 * 6
    assert {k.metric for k in rows} == {f"layer_{g}" for g in REPORT_GROUPS}
    variants = json.loads((out / "protocol-fixture" / "variants.json").read_text())
    modes = {v["spec"]["truth_mode"] for v in variants.values()}
    assert modes == {"none", "registration_method", "fusion_operator"}
    assert (out / "protocol-fixture" / "protocol_summary.md").read_text().count("| all |") > 0


def test_a_rerun_adds_nothing_and_an_interrupted_store_is_completed(tmp_path):
    config, out = _setup(tmp_path)
    run_protocol(config, out_root=out)
    full = _rows(out)
    store = out / "protocol-fixture" / "results.jsonl"
    before = store.read_bytes()
    run_protocol(config, out_root=out)
    assert store.read_bytes() == before
    lines = store.read_bytes().split(b"\n")
    store.write_bytes(b"\n".join(lines[:7]) + b"\n")  # as if killed after seven rows
    run_protocol(config, out_root=out)
    assert set(_rows(out)) == set(full)


def test_config_is_strict(tmp_path):
    config, _ = _setup(tmp_path, 'unknown_key = 1\n')
    with pytest.raises(ValueError, match="unknown_key"):
        load_protocol_config(config)
    config, _ = _setup(tmp_path / "b")
    config.write_text(config.read_text().replace('"raw_adc"', '"display"'), encoding="utf-8")
    with pytest.raises(ValueError, match="vis_input"):
        load_protocol_config(config)
    config.write_text(config.read_text().replace('"display"', '"raw_adc"')
                      + '[registration_study]\nfree_steps = [0]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="free_steps"):
        load_protocol_config(config)


def test_a_changed_config_needs_a_new_name(tmp_path):
    config, out = _setup(tmp_path)
    run_protocol(config, out_root=out)
    config.write_text(config.read_text().replace('["oracle", "infinity_homography"]', '["oracle"]'),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="new experiment name"):
        run_protocol(config, out_root=out)

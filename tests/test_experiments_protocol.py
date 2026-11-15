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
    # 2 samples x 2 registrations x 1 operator x 4 truth modes (none + the three, references present) x 6 layer groups
    assert len(rows) == 2 * 2 * 4 * 6
    assert {k.metric for k in rows} == {f"layer_{g}" for g in REPORT_GROUPS}
    variants = json.loads((out / "protocol-fixture" / "variants.json").read_text())
    modes = {v["spec"]["truth_mode"] for v in variants.values()}
    assert modes == {"none", "registration_method", "registration_method_fullmask", "fusion_operator"}
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


def test_sensitivity_sweeps_every_correlation_length_and_the_bias_once(tmp_path):
    root = build_fixture(tmp_path / "ds", samples=1, size=(32, 24), references=True)
    config = tmp_path / "protocol.toml"
    config.write_text(CONFIG.format(dataset=root.as_posix()).replace('items = ["matrix"]', 'items = ["sensitivity"]')
                      + "\n[sensitivity]\nbias = [0.0, 0.01]\nfield = [0.0, 1.0]\ncorr_len_px = [8, 32]\n"
                      "inflation = false\n", encoding="utf-8")
    assert load_protocol_config(config).sensitivity.corr_len_px == (8.0, 32.0)
    assert run_protocol(config, out_root=tmp_path / "out") == 0
    rows = _rows(tmp_path / "out")
    # 1 sample x 1 operator x (bias 2 amplitudes + field 2 lengths x 2 amplitudes) x 3 truth modes x groups
    assert len(rows) == (2 + 2 * 2) * 3 * len(REPORT_GROUPS)
    variants = json.loads((tmp_path / "out" / "protocol-fixture" / "variants.json").read_text())
    lengths = {(v["spec"]["perturbation"], v["spec"]["corr_len_px"]) for v in variants.values()}
    assert lengths == {("inverse_depth_bias", None), ("smooth_field", 8.0), ("smooth_field", 32.0)}


@pytest.mark.parametrize("value", ["-8", "[]", "[8, 0]", '"x"'])
def test_correlation_lengths_must_be_positive_numbers(tmp_path, value):
    config, _ = _setup(tmp_path, f"\n[sensitivity]\ncorr_len_px = {value}\n")
    with pytest.raises(ValueError, match="corr_len_px"):
        load_protocol_config(config)


def test_hypothesis_count_reaches_the_sweep_parameters(tmp_path):
    config, out = _setup(tmp_path, 'n_hypotheses = 40\nz_min_m = 1.5\n')
    text = config.read_text(encoding="utf-8").replace('registration = ["oracle", "infinity_homography"]',
                                                      'registration = ["oracle", "epipolar_sweep"]')
    config.write_text(text, encoding="utf-8")
    cfg = load_protocol_config(config)
    params = cfg.sweep_params("grad_ncc")
    assert (params.n_hypotheses, params.z_min_m) == (40, 1.5)
    assert load_protocol_config(_setup(tmp_path / "default")[0]).sweep_params("grad_ncc").n_hypotheses == 16
    # (The fixture rig has a zero baseline, so the sweep itself is not run here.)


@pytest.mark.parametrize("value", ["1", "2.5", "true"])
def test_hypothesis_count_must_be_an_integer_of_at_least_two(tmp_path, value):
    config, _ = _setup(tmp_path, f"n_hypotheses = {value}\n")
    with pytest.raises(ValueError, match="n_hypotheses"):
        load_protocol_config(config)

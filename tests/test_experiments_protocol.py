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


def test_sensitivity_can_be_restricted_to_operators_and_truth_modes(tmp_path):
    root = build_fixture(tmp_path / "ds", samples=1, size=(32, 24), references=True)
    config = tmp_path / "protocol.toml"
    text = (CONFIG.format(dataset=root.as_posix()).replace('items = ["matrix"]', 'items = ["sensitivity"]')
            .replace('operators = ["laplacian_pyramid_masked"]', 'operators = ["laplacian_pyramid_masked", "noise_aware_pyramid"]'))
    config.write_text(text + '\n[sensitivity]\nbias = [0.0, 0.01]\nfield = [0.0]\ninflation = false\n'
                      'operators = ["laplacian_pyramid_masked"]\ntruth_modes = ["registration_method"]\n',
                      encoding="utf-8")
    cfg = load_protocol_config(config)
    assert cfg.sensitivity.operators == ("laplacian_pyramid_masked",)
    assert run_protocol(config, out_root=tmp_path / "out") == 0
    variants = json.loads((tmp_path / "out" / "protocol-fixture" / "variants.json").read_text())
    specs = [v["spec"] for v in variants.values()]
    assert {s["operator"] for s in specs} == {"laplacian_pyramid_masked"}
    assert {s["truth_mode"] for s in specs} == {"registration_method"}


def test_sensitivity_operators_must_be_configured_operators(tmp_path):
    config, _ = _setup(tmp_path, '\n[sensitivity]\noperators = ["noise_aware_pyramid"]\n')
    with pytest.raises(ValueError, match="operators"):
        load_protocol_config(config)


def test_an_empty_free_step_list_skips_the_free_search_control(tmp_path):
    config, _ = _setup(tmp_path, '\n[registration_study]\nsimilarities = ["grad_ncc"]\nfree_steps = []\n')
    from quantiloom_fusion.experiments.protocol_run import _Run
    cfg = load_protocol_config(config)
    assert cfg.registration_study.free_steps == ()
    methods = [m for m, *_ in _Run.study_methods(type("R", (), {"cfg": cfg})())]
    assert "free_search" not in methods and "epipolar_sweep" in methods



# --- zeroshot item ---------------------------------------------------------------------------

ZS_ITEMS = 'items = ["zeroshot"]\nzeroshot = ["umf_cmgr"]\n'


def _zs_setup(tmp_path, samples=2):
    root = build_fixture(tmp_path / "ds", samples=samples, size=(32, 24), references=True)
    config = tmp_path / "protocol.toml"
    config.write_text(CONFIG.format(dataset=root.as_posix()).replace('items = ["matrix"]\n', ZS_ITEMS),
                      encoding="utf-8")
    return config, tmp_path / "out"


def _fake_zeroshot_method():
    """A stand-in model (no torch): grey VIS out, the pure-scale map as its own warp."""
    from quantiloom_fusion.core.bands import Band
    from quantiloom_fusion.fusion.base import Capabilities, FusionMethod, FusionOutput
    from quantiloom_fusion.fusion.zeroshot_assets import pure_scale_map
    from quantiloom_fusion.geometry.warp import Warp
    from quantiloom_fusion.signal.colour import luminance

    class Fake(FusionMethod):
        name = "fake_zeroshot"
        capabilities = Capabilities(frozenset({Band.VIS, Band.LWIR}), False, True, False, False)

        def fuse(self, x):
            gv, gl = x.geometry[Band.VIS], x.geometry[Band.LWIR]
            m = pure_scale_map(gv, gl)
            valid = (m[..., 0] >= 0) & (m[..., 0] <= gl.width - 1) & (m[..., 1] >= 0) & (m[..., 1] <= gl.height - 1)
            assert x.images[Band.VIS].ndim == 3 and x.warps is None  # display RGB, raw pair
            return FusionOutput(luminance(x.images[Band.VIS]).astype("float32"),
                                {"warps": {Band.LWIR: Warp(Band.VIS, Band.LWIR, m, valid, {})}, "runtime_s": 0.25})

    return Fake()


IDENTITY = {"model": "umf_cmgr", "code_commit": "b54a82e", "preprocessing": "umf-cmgr-v1: test",
            "weights_sha256": "CPSTN/x.pth=aa;weights/reg_0280.pth=bb;weights/fus_0280.pth=cc"}


def test_zeroshot_config_needs_known_models_the_item_and_lwir(tmp_path):
    config, _ = _zs_setup(tmp_path)
    assert load_protocol_config(config).zeroshot == ("umf_cmgr",)
    text = config.read_text(encoding="utf-8")
    for bad, match in ((text.replace('zeroshot = ["umf_cmgr"]', 'zeroshot = ["rfnet"]'), "zeroshot"),
                       (text.replace('zeroshot = ["umf_cmgr"]\n', ""), "zeroshot"),
                       (text.replace('zeroshot = ["umf_cmgr"]', 'zeroshot = ["umf_cmgr", "umf_cmgr"]'), "duplicate"),
                       (text.replace('items = ["zeroshot"]', 'items = ["matrix"]'), "not in items"),
                       (text.replace('bands = ["vis", "lwir"]', 'bands = ["vis", "mwir"]'), "lwir")):
        config.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_protocol_config(config)


def test_zeroshot_rows_carry_model_identity_input_domain_and_an_anchor(tmp_path, monkeypatch):
    import quantiloom_fusion.experiments.protocol_run as pr

    monkeypatch.setattr(pr, "model_identity", lambda name: dict(IDENTITY))
    monkeypatch.setattr(pr, "_zeroshot_method", lambda name: _fake_zeroshot_method())
    config, out = _zs_setup(tmp_path)
    assert run_protocol(config, out_root=out) == 0
    rows = _rows(out)
    assert len(rows) == 2 * 2 * len(REPORT_GROUPS)  # 2 samples x (model + anchor) x groups
    variants = json.loads((out / "protocol-fixture" / "variants.json").read_text())
    specs = {v["spec"]["model"]: v["spec"] for v in variants.values()}
    assert set(specs) == {"umf_cmgr", "anchor"}
    model, anchor = specs["umf_cmgr"], specs["anchor"]
    assert model["weights_sha256"] == IDENTITY["weights_sha256"] and model["code_commit"] == "b54a82e"
    assert model["registration"] == "model_own" and anchor["registration"] == "oracle"
    assert anchor["operator"] == "laplacian_pyramid_masked"
    for spec in (model, anchor):
        assert spec["truth_mode"] == "fusion_operator" and spec["truth_rule"] == "laplacian_pyramid_masked"
        assert spec["input_domain"] == "vis_display+lwir_temperature_260-340K"
        assert spec["vis_input"] == "display" and spec["ir_window_k"] == [260.0, 340.0]
        assert spec["bands"] == ["vis", "lwir"]
    fp = {v["spec"]["model"]: v["fingerprint"] for v in variants.values()}
    model_rows = [v for k, v in rows.items() if k.method_fingerprint == fp["umf_cmgr"]]
    assert all(v["runtime_s"] == 0.25 and "truth_PSNR" in v and "epe_mean_px" in v for v in model_rows)
    assert any("hp_dev_RMSE" in v for v in model_rows)
    assert "| umf_cmgr | model_own |" in (out / "protocol-fixture" / "protocol_summary.md").read_text()
    # Changed weights are a new variant, not a resumed one.
    monkeypatch.setattr(pr, "model_identity", lambda name: IDENTITY | {"weights_sha256": "changed"})
    run_protocol(config, out_root=out)
    assert len(_rows(out)) == 2 * 3 * len(REPORT_GROUPS)


def test_zeroshot_without_weights_writes_error_rows_and_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTILOOM_ZEROSHOT_MODELS", str(tmp_path / "no-models"))
    config, out = _zs_setup(tmp_path)
    assert run_protocol(config, out_root=out) == 1
    errors = [json.loads(line) for line in (out / "protocol-fixture" / "errors.jsonl").read_text().splitlines()]
    assert len(errors) == 2 and all("FileNotFoundError" in e["error"] and "umf_cmgr" in e["label"] for e in errors)
    assert len(_rows(out)) == 2 * len(REPORT_GROUPS)  # the anchor still ran
    assert run_protocol(config, out_root=out) == 1  # retried, still failing loudly


def test_zeroshot_without_torch_writes_error_rows_and_fails(tmp_path, monkeypatch):
    import quantiloom_fusion.experiments.protocol_run as pr

    def no_torch(name):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(pr, "model_identity", lambda name: dict(IDENTITY))
    monkeypatch.setattr(pr, "_zeroshot_method", no_torch)
    config, out = _zs_setup(tmp_path, samples=1)
    assert run_protocol(config, out_root=out) == 1
    errors = [json.loads(line) for line in (out / "protocol-fixture" / "errors.jsonl").read_text().splitlines()]
    assert len(errors) == 1 and "torch" in errors[0]["error"] and errors[0]["fingerprint"]



# --- band_ablation item ----------------------------------------------------------------------

BAND_SETS = 'band_sets = [["vis"], ["vis", "mwir"], ["vis", "lwir"], ["vis", "mwir", "lwir"]]\n'


def _ba_setup(tmp_path, items='["matrix", "band_ablation"]', band_sets=BAND_SETS):
    root = build_fixture(tmp_path / "ds", samples=2, size=(32, 24), references=True)
    config = tmp_path / "protocol.toml"
    text = (CONFIG.format(dataset=root.as_posix()).replace('bands = ["vis", "lwir"]', 'bands = ["vis", "mwir", "lwir"]')
            .replace('items = ["matrix"]', f"items = {items}")
            .replace('registration = ["oracle", "infinity_homography"]', 'registration = ["oracle"]'))
    config.write_text(text + band_sets, encoding="utf-8")
    return config, tmp_path / "out"


def test_band_ablation_rows_share_one_reference_and_the_full_set_matches_the_matrix(tmp_path):
    config, out = _ba_setup(tmp_path)
    cfg = load_protocol_config(config)
    assert [tuple(map(str, s)) for s in cfg.band_sets] == [("vis",), ("vis", "mwir"), ("vis", "lwir"),
                                                           ("vis", "mwir", "lwir")]
    assert run_protocol(config, out_root=out) == 0
    rows = _rows(out)
    variants = json.loads((out / "protocol-fixture" / "variants.json").read_text())
    specs = {v["fingerprint"]: v["spec"] for v in variants.values()}
    ablation = {fp: s for fp, s in specs.items() if s["item"] == "band_ablation"}
    assert len(ablation) == 4
    assert len([k for k in rows if k.method_fingerprint in ablation]) == 2 * 4 * len(REPORT_GROUPS)
    for spec in ablation.values():
        assert spec["reference_band_set"] == ["vis", "mwir", "lwir"]
        assert spec["truth_mode"] == "fusion_operator" and spec["truth_rule"] == "laplacian_pyramid_masked"
    alone = next(fp for fp, s in ablation.items() if s["band_set"] == ["vis"])
    assert (ablation[alone]["operator"], ablation[alone]["registration"]) == ("reference_only", "none")
    full = next(fp for fp, s in ablation.items() if s["band_set"] == ["vis", "mwir", "lwir"])
    assert (ablation[full]["operator"], ablation[full]["registration"]) == ("laplacian_pyramid_masked", "oracle")
    matrix = next(fp for fp, s in specs.items() if s["item"] == "matrix" and s["truth_mode"] == "fusion_operator")
    by_key = {(k.method_fingerprint, k.sample_id, k.metric): v for k, v in rows.items()}
    for (fp, sid, metric), v in by_key.items():
        if fp == full:
            assert v["truth_MSE"] == by_key[(matrix, sid, metric)]["truth_MSE"] or (
                v["truth_MSE"] != v["truth_MSE"])  # NaN on an empty group
        if fp == alone:
            assert v["SCD"] != v["SCD"] and "cleansrc_Qabf" in v and "truth_PSNR_demeaned" in v
    md = (out / "protocol-fixture" / "protocol_summary.md").read_text()
    assert "| vis+mwir+lwir | vis+mwir+lwir |" in md and "| vis | vis+mwir+lwir |" in md


@pytest.mark.parametrize("band_sets, match", [
    ('band_sets = [["lwir"]]\n', "band_sets"),
    ('band_sets = [["vis", "vis"]]\n', "band_sets"),
    ('band_sets = [["vis", "swir"]]\n', "band_sets"),
    ('band_sets = []\n', "band_sets"),
    ('band_sets = [["vis"], ["vis"]]\n', "duplicate"),
    ('band_sets = [["vis", "mwir"], ["vis", "lwir", "mwir"], ["vis", "mwir", "lwir"]]\n', "duplicate"),
    ("", "band_sets"),
])
def test_band_sets_are_validated(tmp_path, band_sets, match):
    config, _ = _ba_setup(tmp_path, band_sets=band_sets)
    with pytest.raises(ValueError, match=match):
        load_protocol_config(config)


def test_band_sets_need_the_item_and_configured_bands(tmp_path):
    config, _ = _ba_setup(tmp_path, items='["matrix"]')
    with pytest.raises(ValueError, match="not in items"):
        load_protocol_config(config)
    config, _ = _ba_setup(tmp_path / "b")
    config.write_text(config.read_text().replace('bands = ["vis", "mwir", "lwir"]', 'bands = ["vis", "lwir"]'),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="band_sets"):
        load_protocol_config(config)

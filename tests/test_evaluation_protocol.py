"""Evaluation protocol on a synthetic tilted plane seen by two cameras with a baseline."""

import numpy as np
import pytest
from dataset_fixture import build_fixture
from test_geometry_oracle import geometry, raycast

from quantiloom_fusion.core.bands import BANDS, Band
from quantiloom_fusion.evaluation.layer_report import FusedView, combine_layers, layer_masks, source_ref_metrics
from quantiloom_fusion.geometry import Layer
from quantiloom_fusion.evaluation.perturb import smooth_field_warp
from quantiloom_fusion.evaluation.protocol import (
    PerturbationKind,
    Sweep,
    epe_columns,
    evaluate_point,
    inflation_experiment,
    is_calibrated,
    perturbed_warps,
    sensitivity_curve,
)
from quantiloom_fusion.dataset import ThreeBandDataset
from quantiloom_fusion.evaluation.scene import InputSpec, build_scene, load_scene
from quantiloom_fusion.evaluation.source_ref import ms_ssim_fusion_n, qabf_n, scd_n
from quantiloom_fusion.evaluation.truth_ref import TruthMode
from quantiloom_fusion.fusion import Capabilities, FusionMethod, FusionOutput, create
from quantiloom_fusion.signal.normalize import IrLinear, IrTemperature, VisDisplay, VisRawLinear

NORM = {"vis": "vis_display", "ir": "ir_temperature", "params": {"lwir": {"lo": 260.0, "hi": 340.0}},
        "scene_independent": True}
NORMAL = (0.0, 0.6, 0.8)


def _texture(p):
    dx = (p[..., 0] + 0.8) % 1.6 - 0.8
    dy = (p[..., 1] + 0.8) % 1.6 - 0.8
    return (0.1 + 0.8 * np.exp(-(dx**2 + dy**2) / (2 * 0.15**2))).astype(np.float32)


def _scene(norm=NORM, with_clean=True):
    ref = raycast(geometry(96, 72, 100.0, pitch=0.9), normal=NORMAL, offset=10.0)
    src = raycast(geometry(96, 72, 100.0, pitch=0.9, t=(0.3, 0.0, 0.0)), normal=NORMAL, offset=10.0)
    images = {Band.VIS: _texture(ref.world_position), Band.LWIR: _texture(src.world_position)}
    scene = build_scene("synthetic", images, {Band.VIS: ref, Band.LWIR: src}, norm)
    if not with_clean:
        return scene
    clean = {Band.VIS: images[Band.VIS], Band.LWIR: 1.0 - images[Band.VIS]}
    return type(scene)(**{**scene.__dict__, "clean": clean})


class Constant(FusionMethod):
    name = "constant"
    capabilities = Capabilities(frozenset(BANDS), True, False, False, False)

    def fuse(self, x):
        return FusionOutput(np.full(x.images[x.reference].shape[:2], 0.3, np.float32))


def _row(rows, layer, **where):
    return next(r for r in rows if r["layer"] == layer and all(r[k] == v for k, v in where.items()))


@pytest.mark.parametrize("kind", list(PerturbationKind))
def test_zero_amplitude_reproduces_oracle(kind):
    scene = _scene()
    warps = perturbed_warps(scene, Sweep(kind, [0.0]), 0.0)
    oracle = scene.oracle_warps[Band.LWIR]
    np.testing.assert_array_equal(warps[Band.LWIR].map_xy, oracle.map_xy)
    np.testing.assert_array_equal(warps[Band.LWIR].valid, oracle.valid)


def test_epe_grows_monotonically_with_inverse_depth_bias():
    scene = _scene()
    sweep = Sweep(PerturbationKind.INVERSE_DEPTH_BIAS, [])
    epe = [epe_columns(scene, perturbed_warps(scene, sweep, d))["epe_mean_px"]
           for d in (0.0, 0.002, 0.004, 0.008, 0.016)]
    assert epe[0] == 0.0
    assert all(b > a for a, b in zip(epe, epe[1:]))
    mrad = epe_columns(scene, perturbed_warps(scene, sweep, 0.016))["epe_mean_mrad"]
    assert mrad == pytest.approx(epe[-1] / 100.0 * 1000.0)


def test_smooth_field_has_requested_rms_and_is_seeded():
    oracle = _scene().oracle_warps[Band.LWIR]
    a = smooth_field_warp(oracle, 2.0, 6.0, seed=3)
    disp = a.map_xy - oracle.map_xy
    assert np.sqrt(np.nanmean(np.sum(disp**2, axis=-1))) == pytest.approx(2.0)
    np.testing.assert_array_equal(a.map_xy, smooth_field_warp(oracle, 2.0, 6.0, seed=3).map_xy)
    assert not np.array_equal(a.map_xy, smooth_field_warp(oracle, 2.0, 6.0, seed=4).map_xy)


def test_no_reference_metrics_inflate_with_misregistration():
    sweep = Sweep(PerturbationKind.INVERSE_DEPTH_BIAS, [0.0, 0.1, 0.2])
    rows = inflation_experiment(_scene(), create("max_abs"), sweep)
    assert [r["amplitude"] for r in rows] == [0.0, 0.1, 0.2]
    assert rows[0]["ctx_AG_ratio"] == 1.0
    for r in rows[1:]:
        assert r["ctx_AG"] > rows[0]["ctx_AG"] and r["ctx_SF"] > rows[0]["ctx_SF"]
        assert r["epe_mean_px"] > rows[0]["epe_mean_px"]


def test_source_metrics_ignore_a_band_outside_its_own_support():
    rng = np.random.default_rng(0)
    a = rng.random((192, 192))
    b = rng.random((192, 192))
    support = np.zeros((192, 192), bool)
    support[40:150, 30:160] = True
    garbage = np.where(support, b, rng.random((192, 192)))
    fused = 0.5 * (a + b)
    mask = np.ones((192, 192))
    tampered = np.where(support, rng.random((192, 192)), b)
    for metric in (scd_n, qabf_n):
        kept = metric(fused, [a, b], mask=mask, supports=[None, support])
        assert metric(fused, [a, garbage], mask=mask, supports=[None, support]) == pytest.approx(kept, abs=1e-12)
        assert abs(metric(fused, [a, tampered], mask=mask, supports=[None, support]) - kept) > 1e-3
    leak = abs(ms_ssim_fusion_n(fused, [a, b], mask=mask, supports=[None, support])
               - ms_ssim_fusion_n(fused, [a, garbage], mask=mask, supports=[None, support]))
    assert 0.0 < leak < 1e-3

    def wrapped(src):
        view = FusedView(fused, {Band.VIS: a, Band.LWIR: src},
                         {Band.VIS: np.ones_like(support), Band.LWIR: support}, Band.VIS)
        return source_ref_metrics(view, mask > 0.5)

    kept, same, changed = wrapped(b), wrapped(garbage), wrapped(tampered)
    for key in kept:
        assert same[key] == pytest.approx(kept[key], abs=1e-12)
        assert abs(changed[key] - kept[key]) > 1e-3


def test_truth_mode_fusion_scores_a_constant_operator_badly_where_self_reference_is_perfect():
    scene = _scene()
    op = Constant()
    reg = _row(evaluate_point(scene, op, scene.oracle_warps, TruthMode.REGISTRATION), "correspondable")
    fus = _row(evaluate_point(scene, op, scene.oracle_warps, TruthMode.FUSION), "correspondable")
    assert reg["truth_mode"] == "registration_method" and fus["truth_mode"] == "fusion_operator"
    assert reg["truth_PSNR"] > 96.0
    assert fus["truth_PSNR"] < 30.0
    assert fus["truth_rule"] == "laplacian_pyramid_masked" and "cleansrc_SCD" in fus and "cleansrc_SCD" not in reg


def test_truth_modes_need_clean_images_and_a_fixed_window():
    op = create("max_abs")
    scene = _scene(with_clean=False)
    with pytest.raises(ValueError, match="clean"):
        evaluate_point(scene, op, scene.oracle_warps, TruthMode.REGISTRATION)
    adaptive = _scene(norm={**NORM, "scene_independent": False})
    with pytest.raises(ValueError, match="fixed normalisation"):
        evaluate_point(adaptive, op, adaptive.oracle_warps, TruthMode.REGISTRATION)


def test_layers_partition_and_occluded_layer_reports_deviation_from_reference_band():
    scene = _scene()
    masks = layer_masks(scene.layers)
    assert not (masks["correspondable"] & masks["occluded_oov"]).any()
    assert masks["correspondable"].any() and masks["occluded_oov"].any()
    good = _row(evaluate_point(scene, create("max_abs"), scene.oracle_warps, None), "occluded_oov")
    assert good["dev_RMSE"] < 1e-6 and np.isnan(good["SCD"])
    bad = _row(evaluate_point(scene, Constant(), scene.oracle_warps, None), "occluded_oov")
    assert bad["dev_RMSE"] > 0.05


def test_combined_layer_is_the_worst_over_sources():
    corr, edge, occ = (np.array([int(layer)], np.int8) for layer in
                       (Layer.CORRESPONDABLE, Layer.EDGE, Layer.OCCLUDED))
    assert combine_layers([corr, edge]).item() == Layer.EDGE
    assert combine_layers([edge, occ, corr]).item() == Layer.OCCLUDED


def test_rows_record_normalisation_calibration_and_context_prefix():
    rows = sensitivity_curve(_scene(), create("max_abs"), Sweep(PerturbationKind.SMOOTH_FIELD, [0.0, 1.0]))
    assert {r["layer"] for r in rows} == {"correspondable", "edge", "occluded_oov"}
    for r in rows:
        assert (r["norm_vis"], r["norm_ir"]) == ("vis_display", "ir_temperature")
        assert '"lo": 260.0' in r["norm_params"]
        assert r["calibrated"] is True and r["registration"] == "oracle+smooth_field"
        assert {"ctx_EN", "ctx_SD", "ctx_SF", "ctx_AG"} <= set(r) and not {"EN", "AG"} & set(r)
    assert is_calibrated("oracle") and is_calibrated("epipolar_sweep") and not is_calibrated("resize")
    with pytest.raises(KeyError):
        is_calibrated("some_deep_model")


def test_load_scene_reads_references_on_the_vis_grid_and_records_the_window(tmp_path):
    root = build_fixture(tmp_path, samples=1, size=(32, 24), references=True)
    spec = InputSpec(VisDisplay(), IrLinear(mode="fixed", fixed=(0.0, 100.0)))
    scene = load_scene(ThreeBandDataset(root), "fixture_s00", spec)
    assert scene.images[Band.VIS].ndim == 2
    assert set(scene.clean) == set(BANDS)
    assert all(c.shape == scene.images[Band.VIS].shape for c in scene.clean.values())
    assert scene.norm["scene_independent"] and scene.norm["params"]["lwir"]["hi"] == 100.0
    assert load_scene(ThreeBandDataset(build_fixture(tmp_path / "v1", samples=1)), "fixture_s00", spec).clean is None


def test_input_spec_rejects_ir_display_and_flags_adaptive_windows():
    with pytest.raises(ValueError, match="never the display"):
        InputSpec(VisDisplay(), VisDisplay())
    with pytest.raises(ValueError, match="VIS normaliser"):
        InputSpec(IrLinear(), IrTemperature())
    assert InputSpec(VisRawLinear(), IrTemperature()).scene_independent
    assert not InputSpec(VisDisplay(), IrLinear()).scene_independent
    assert InputSpec(VisDisplay(), IrLinear(mode="fixed", fixed=(0.0, 100.0))).scene_independent

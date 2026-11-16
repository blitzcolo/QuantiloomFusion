import numpy as np
import pytest

from dataset_fixture import build_fixture
from quantiloom_fusion.dataset.reader import ThreeBandDataset
from quantiloom_fusion.dataset.splits import leave_one_group_out
from quantiloom_fusion.dataset.types import Condition
from quantiloom_fusion.evaluation.complementarity import pixel_masks
from quantiloom_fusion.evaluation.complementarity_contrast import (
    band_radiance,
    boundary_contrast,
    boundary_pairs,
    decompose_contrast,
)
from quantiloom_fusion.evaluation.complementarity_report import _regression_cells
from quantiloom_fusion.evaluation.complementarity_stats import (
    FrameId,
    PairArrays,
    RegressionScores,
    cmi_estimate,
    folds_for_condition,
    pair_frame,
    r2_vs_zero,
    regression_scores,
    view_group_folds,
)

LWIR_NM = (8000.0, 12000.0)
SIGMA_SB = 5.670374419e-8


def _arrays(rng, n, lwir_mode, y_offset=0.0):
    v = rng.normal(size=(n, 6))
    u = rng.normal(size=n)
    y = 2.0 * v[:, 3] + (1.5 * u if lwir_mode == "informative" else 0.0) + 0.3 * rng.normal(size=n)
    lw_core = u + 0.05 * rng.normal(size=n) if lwir_mode == "informative" else rng.normal(size=n)
    x_lwir = np.stack([lw_core, rng.normal(size=n), rng.normal(size=n)], axis=1)
    return PairArrays(y + y_offset, v, x_lwir + y_offset, rng.integers(0, 3, n))


def _frames(lwir_mode, offsets=None, n=1500):
    rng = np.random.default_rng(7)
    frames = []
    for i in range(8):
        off = 0.0 if offsets is None else offsets[i]
        ident = FrameId(f"s{i}", f"g{i % 4}", Condition.DAY)
        frames.append(pair_frame(ident, _arrays(rng, n, lwir_mode, off), 1000, np.random.default_rng(i)))
    return frames


def test_band_radiance_matches_stefan_boltzmann_over_wide_band():
    got = band_radiance(300.0, (500.0, 500_000.0), samples=400_001)
    assert got == pytest.approx(SIGMA_SB * 300.0**4 / np.pi, rel=2e-3)


def test_pure_temperature_contrast_has_zero_emissivity_term():
    t1, t2 = np.array([290.0, 300.0]), np.array([310.0, 295.0])
    eps = np.array([0.9, 0.7])
    c = decompose_contrast(t1, eps, t2, eps, LWIR_NM, l_down=5.0)
    assert np.all(c.emissivity == 0.0)
    assert c.temperature == pytest.approx(c.d_radiance, rel=1e-12)


def test_pure_emissivity_contrast_has_zero_temperature_term():
    t = np.array([300.0, 320.0])
    c = decompose_contrast(t, np.array([0.95, 0.9]), t, np.array([0.5, 0.6]), LWIR_NM, l_down=8.0)
    assert np.all(c.temperature == 0.0)
    assert c.emissivity == pytest.approx(c.d_radiance, rel=1e-12)


def test_symmetric_split_is_exact_and_linear_split_reports_interaction():
    rng = np.random.default_rng(0)
    t1, t2 = rng.uniform(260, 340, 50), rng.uniform(260, 340, 50)
    e1, e2 = rng.uniform(0.1, 1.0, 50), rng.uniform(0.1, 1.0, 50)
    c = decompose_contrast(t1, e1, t2, e2, LWIR_NM, l_down=12.0)
    assert c.temperature + c.emissivity == pytest.approx(c.d_radiance, rel=1e-9, abs=1e-9)
    assert c.linear_temperature + c.linear_emissivity + c.interaction == pytest.approx(c.d_radiance, rel=1e-9, abs=1e-9)
    assert np.abs(c.interaction).max() > 0.0


def test_boundary_pairs_only_cross_label_and_valid():
    label = np.array([[0, 0, 1], [0, 2, 1]])
    valid = np.ones(label.shape, bool)
    assert boundary_pairs(label, valid)[0].size == 4
    valid[0, 2] = False
    assert boundary_pairs(label, valid)[0].size == 3


def test_boundary_contrast_attributes_pure_cases_to_one_term():
    label = np.array([[0, 1], [0, 1]])
    valid = np.ones(label.shape, bool)
    same_t = boundary_contrast(np.full((2, 2), 300.0), np.array([[0.9, 0.5]] * 2), label, valid, LWIR_NM)
    assert same_t.temperature_share == 0.0
    same_e = boundary_contrast(np.array([[300.0, 320.0]] * 2), np.full((2, 2), 0.9), label, valid, LWIR_NM)
    assert same_e.temperature_share == 1.0


def test_frame_offset_in_target_does_not_change_residuals():
    offsets = [40.0, -25.0, 310.0, 5.0, -90.0, 12.0, 0.5, 77.0]
    base = regression_scores(_frames("none")[:6], _frames("none")[6:])
    shifted = regression_scores(_frames("none", offsets)[:6], _frames("none", offsets)[6:])
    for name in ("var_total", "ridge_vis", "ridge_both", "knn_vis", "knn_both"):
        assert getattr(shifted, name) == pytest.approx(getattr(base, name), rel=1e-9)


def test_residual_reduction_is_near_zero_when_lwir_adds_nothing():
    s = regression_scores(_frames("none")[:6], _frames("none")[6:])
    assert abs(1 - s.ridge_both / s.ridge_vis) < 0.03


def test_residual_reduction_is_large_when_lwir_carries_extra_information():
    s = regression_scores(_frames("informative")[:6], _frames("informative")[6:])
    assert 1 - s.ridge_both / s.ridge_vis > 0.7
    assert 1 - s.knn_both / s.knn_vis > 0.3


def _codes(rng, n):
    v = rng.integers(0, 4, n)
    return v, rng.integers(0, 5, n)


def test_cmi_is_near_zero_when_lwir_independent_of_y_given_vis():
    rng = np.random.default_rng(1)
    v, x = _codes(rng, 20000)
    y = (v + (rng.random(20000) < 0.3)).astype(np.intp)
    est = cmi_estimate(y, x.astype(np.intp), v.astype(np.intp), 10, np.random.default_rng(2))
    assert abs(est.corrected) < 0.01


def test_cmi_is_large_when_lwir_carries_y_information():
    rng = np.random.default_rng(1)
    v, x = _codes(rng, 20000)
    est = cmi_estimate(x.astype(np.intp), x.astype(np.intp), v.astype(np.intp), 10, np.random.default_rng(2))
    assert est.corrected > 1.0


def test_cmi_is_near_zero_when_lwir_only_repeats_vis():
    rng = np.random.default_rng(1)
    v = rng.integers(0, 4, 20000)
    y = (v + (rng.random(20000) < 0.3)).astype(np.intp)
    est = cmi_estimate(y, v.astype(np.intp), v.astype(np.intp), 10, np.random.default_rng(2))
    assert abs(est.corrected) < 0.01


def test_folds_never_split_a_group_and_report_composition():
    groups = ["gA", "gB", "gC", "gD"]
    conds = (Condition.DAY, Condition.NIGHT, Condition.MOONLIGHT)
    meta = {f"{g}_{c.value}_{k}": (g, c) for g in groups for c in conds for k in range(2)}
    ids = tuple(meta)
    logo = [
        (tuple(s for s in ids if meta[s][0] != g), tuple(s for s in ids if meta[s][0] == g), g)
        for g in groups
    ]
    for cond in conds:
        folds = folds_for_condition(logo, meta, cond)
        assert len(folds) == 4
        for f in folds:
            assert not set(f.train_groups) & set(f.test_groups)
            assert {meta[s][0] for s in f.test_ids} == set(f.test_groups)
            assert {meta[s][0] for s in f.train_ids} == set(f.train_groups)
            assert all(meta[s][1] == cond for s in f.train_ids + f.test_ids)
            assert len(f.train_groups) == 3


def test_folds_from_real_leave_one_group_out_on_fixture(tmp_path):
    ds = ThreeBandDataset(build_fixture(tmp_path, samples=6, groups=["g0", "g1", "g2"]))
    meta = {s.sample_id: (s.split_group, s.condition) for s in ds}
    folds = folds_for_condition(leave_one_group_out(ds), meta, Condition.DAY)
    assert sorted(f.test_groups[0] for f in folds) == ["g0", "g1", "g2"]
    assert all(set(f.test_ids).isdisjoint(f.train_ids) for f in folds)


def test_null_emissivity_pixels_stay_in_regression_but_not_in_contrast():
    shape = (2, 3)
    temp = np.full(shape, 300.0)
    eps = np.array([[0.9, np.nan, 0.8], [np.nan, 0.9, 0.9]])
    label = np.array([[0, 1, 2], [1, 0, 2]])
    feats = np.ones(shape + (2,))
    warp_valid = np.ones(shape, bool)
    warp_valid[1, 2] = False
    regression, contrast = pixel_masks(warp_valid, temp, eps, label, feats, feats)
    assert regression.sum() == 5
    assert contrast.sum() == 3
    assert regression[0, 1] and regression[1, 0]
    assert not contrast[0, 1] and not contrast[1, 0]
    assert not np.any(contrast & ~regression)
    assert not regression[1, 2]


def test_r2_is_one_minus_sse_over_zero_predictor_variance():
    s = RegressionScores(n_test=10, var_total=4.0, ridge_vis=2.0, ridge_both=1.0, knn_vis=8.0, knn_both=4.0)
    assert s.r2_ridge_vis == pytest.approx(0.5)
    assert s.r2_ridge_both == pytest.approx(0.75)
    assert s.r2_knn_vis == pytest.approx(-1.0)
    assert s.r2_knn_both == 0.0
    assert r2_vs_zero(4.0, 4.0) == 0.0
    frames = _frames("none")
    zero = regression_scores(frames[:6], frames[6:])
    assert zero.r2_ridge_vis == pytest.approx(1 - zero.ridge_vis / zero.var_total)
    assert zero.r2_ridge_vis > 0.5


def test_regression_cells_report_n_a_for_an_isothermal_target():
    # An isothermal frame de-means to zero everywhere: zero total variance and zero residuals.
    cells = _regression_cells(100, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert cells == ["100", "0.00", "0.00", "0.00", "n/a", "n/a", "n/a", "0.00", "0.00", "n/a", "n/a", "n/a"]
    cells = _regression_cells(10, 4.0, 2.0, 1.0, 8.0, 4.0)
    assert cells[4:7] == ["+50.0%", "+0.50", "+0.75"] and cells[9:] == ["+50.0%", "-1.00", "+0.00"]


def test_view_group_folds_never_share_a_view_group_between_train_and_test():
    samples = [(f"s{az}_{k}", float(az)) for az in (0, 45, 90, 135) for k in range(3)]
    azimuth = dict(samples)
    folds = view_group_folds(samples, "g")
    assert len(folds) == 4
    assert {f.kind for f in folds} == {"within_scene_view"}
    assert sorted(s for f in folds for s in f.test_ids) == sorted(s for s, _ in samples)
    for f in folds:
        assert {azimuth[s] for s in f.train_ids}.isdisjoint({azimuth[s] for s in f.test_ids})
        assert len(f.test_ids) == 3 and len(f.train_ids) == 9
        assert f.train_groups == f.test_groups == ("g",)


def test_view_group_folds_blocks_group_contiguous_azimuths_and_need_two_groups():
    samples = [(f"s{az}", float(az)) for az in range(0, 360, 45)]
    folds = view_group_folds(samples, "g", blocks=2)
    assert [f.test_views for f in folds] == [("0+45+90+135",), ("180+225+270+315",)]
    assert set(folds[0].test_ids).isdisjoint(folds[0].train_ids)
    assert view_group_folds([("a", 0.0), ("b", 0.0)], "g") == []

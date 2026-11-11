import numpy as np
import pytest

from dataset_fixture import build_fixture
from quantiloom_fusion.dataset.reader import ThreeBandDataset
from quantiloom_fusion.dataset.splits import leave_one_group_out
from quantiloom_fusion.dataset.types import Condition
from quantiloom_fusion.evaluation.complementarity_contrast import (
    band_radiance,
    boundary_contrast,
    boundary_pairs,
    decompose_contrast,
)
from quantiloom_fusion.evaluation.complementarity_stats import (
    FrameId,
    PairArrays,
    blocked_folds,
    cmi_estimate,
    folds_for_condition,
    pair_frame,
    regression_scores,
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


def test_blocked_folds_partition_one_group_and_are_labelled_within_group():
    folds = blocked_folds([f"s{i}" for i in range(8)], "g", 2)
    assert [f.kind for f in folds] == ["within_group_blocked"] * 2
    assert sorted(s for f in folds for s in f.test_ids) == [f"s{i}" for i in range(8)]

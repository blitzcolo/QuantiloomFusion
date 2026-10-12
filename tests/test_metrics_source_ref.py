"""Tests for the source-referenced fusion metrics (SCD, Qabf, MS-SSIM).

Unit tests are ported from paper-exp tests/test_metrics.py (the SCD, Qabf and
MS-SSIM sections, paper-exp 42024cd 2026-08-29).  Equivalence tests compare
the verbatim two-source port against the originals, and reduction tests check
that each ``*_n`` generalisation equals its two-source original on two
sources.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from quantiloom_fusion.evaluation.source_ref import (
    ms_ssim,
    ms_ssim_fusion,
    ms_ssim_fusion_n,
    qabf,
    qabf_n,
    scd,
    scd_n,
)

from paper_exp_loader import PAPER_EXP, paper_exp_modules


@pytest.fixture
def rng():
    return np.random.default_rng(20240814)


@pytest.fixture
def noise(rng):
    return rng.random((64, 64)).astype(np.float32)


def _centred_mask(h=64, w=64, margin=16):
    mask = np.zeros((h, w), np.float32)
    mask[margin:h - margin, margin:w - margin] = 1.0
    return mask


# --- Ported from paper-exp tests/test_metrics.py: SCD ---------------------


def test_scd_correlates_each_difference_with_the_other_source(rng):
    vis = rng.random((64, 64))
    ir = rng.random((64, 64))
    average = 0.5 * (vis + ir)
    assert scd(average, vis, ir) > 1.0
    unrelated = vis + 0.5 * rng.random((64, 64))
    assert scd(average, vis, ir) > scd(unrelated, vis, ir)


def test_scd_is_blind_to_how_much_of_the_other_source_was_added(rng):
    vis, ir = rng.random((64, 64)), rng.random((64, 64))
    timid = 0.9 * vis + 0.1 * ir
    bold = 0.5 * vis + 0.5 * ir
    assert scd(timid, vis, ir) == pytest.approx(scd(bold, vis, ir), abs=1e-12)


def test_scd_of_a_pass_through_fusion_is_undefined_rather_than_low(rng):
    vis, ir = rng.random((64, 64)), rng.random((64, 64))
    assert np.isnan(scd(vis, vis, ir))
    assert np.isnan(scd(ir, vis, ir))


def test_scd_is_symmetric_under_exchanging_the_sources(rng):
    vis, ir = rng.random((32, 32)), rng.random((32, 32))
    fused = 0.5 * (vis + ir)
    assert scd(fused, vis, ir) == pytest.approx(scd(fused, ir, vis), abs=1e-9)


def test_scd_rejects_mismatched_shapes(rng):
    with pytest.raises(ValueError):
        scd(rng.random((64, 64)), rng.random((64, 64)), rng.random((16, 16)))


# --- Ported from paper-exp tests/test_metrics.py: Qabf --------------------


def test_qabf_scores_inverted_polarity_edges_as_preserved(noise):
    assert qabf(1.0 - noise, noise, noise) == pytest.approx(
        qabf(noise, noise, noise), rel=1e-6)
    assert qabf(1.0 - noise, noise, noise) > 0.9


def test_qabf_is_bounded_and_ordered_by_degradation(noise):
    perfect = qabf(noise, noise, noise)
    blurred = qabf(cv2.GaussianBlur(noise, (0, 0), 2.0), noise, noise)
    assert 0.0 <= blurred < perfect <= 1.0


def test_qabf_of_flat_sources_is_undefined_rather_than_zero():
    flat = np.full((32, 32), 0.5, np.float32)
    assert np.isnan(qabf(flat, flat, flat))


def test_qabf_mask_restricts_the_weighting(rng):
    mask = _centred_mask()
    vis = rng.random((64, 64)).astype(np.float32)
    ir = rng.random((64, 64)).astype(np.float32)
    fused = vis.copy()
    fused[mask < 0.5] = 0.0
    assert qabf(fused, vis, ir, mask=mask) > qabf(fused, vis, ir)


# --- Ported from paper-exp tests/test_metrics.py: MS-SSIM -----------------


def test_ms_ssim_of_identical_images_is_one(rng):
    img = rng.random((256, 256))
    assert ms_ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ms_ssim_decreases_monotonically_with_blur(noise):
    values = [ms_ssim(noise, cv2.GaussianBlur(noise, (0, 0), s) if s else noise)
              for s in (0, 1, 2, 4)]
    assert values == sorted(values, reverse=True)


def test_ms_ssim_drops_scales_it_cannot_measure_instead_of_repeating_one(rng):
    small = rng.random((24, 24))
    assert ms_ssim(small, small) == pytest.approx(1.0, abs=1e-6)
    tiny = rng.random((8, 8))
    assert ms_ssim(tiny, tiny) == pytest.approx(1.0, abs=1e-6)


def test_ms_ssim_fusion_averages_over_both_sources(rng):
    vis, ir = rng.random((64, 64)), rng.random((64, 64))
    expected = 0.5 * (ms_ssim(vis, vis) + ms_ssim(vis, ir))
    assert ms_ssim_fusion(vis, vis, ir) == pytest.approx(expected, abs=1e-9)


def test_ms_ssim_empty_mask_is_undefined(rng):
    img = rng.random((64, 64))
    assert np.isnan(ms_ssim(img, img, mask=np.zeros_like(img)))


# --- Equivalence against the original two-source implementations ----------

paper_exp = pytest.mark.skipif(
    not PAPER_EXP.exists(), reason="paper-exp checkout not present"
)


def _call_each(fused, vis, ir, mask=None, ir_support=None):
    ours = {
        "scd": scd(fused, vis, ir, mask, ir_support),
        "qabf": qabf(fused, vis, ir, mask, ir_support),
        "ms_ssim_fusion": ms_ssim_fusion(fused, vis, ir, mask, ir_support),
    }
    with paper_exp_modules():
        from paper_exp.metrics import source_ref as orig

        theirs = {
            "scd": orig.scd(fused, vis, ir, mask, ir_support),
            "qabf": orig.qabf(fused, vis, ir, mask, ir_support),
            "ms_ssim_fusion": orig.ms_ssim_fusion(
                fused, vis, ir, mask, ir_support),
        }
    return ours, theirs


def _assert_equal(ours, theirs, tol=1e-9):
    for name in ours:
        got, exp = ours[name], theirs[name]
        assert (np.isnan(got) and np.isnan(exp)) or got == pytest.approx(
            exp, abs=tol), f"{name}: {got} vs {exp}"


@paper_exp
def test_two_source_equivalence_unmasked(rng):
    vis = rng.random((96, 80)).astype(np.float32)
    ir = rng.random((96, 80)).astype(np.float32)
    fused = 0.6 * vis + 0.4 * ir
    _assert_equal(*_call_each(fused, vis, ir))


@paper_exp
def test_two_source_equivalence_masked_with_support(rng):
    vis = rng.random((96, 80)).astype(np.float32)
    ir = rng.random((96, 80)).astype(np.float32)
    fused = 0.6 * vis + 0.4 * ir
    mask = _centred_mask(96, 80, margin=20)
    support = np.zeros((96, 80), np.float32)
    support[:, :60] = 1.0  # IR covers the left 60 columns
    _assert_equal(*_call_each(fused, vis, ir, mask, support))


@paper_exp
def test_two_source_equivalence_support_no_mask(rng):
    vis = rng.random((96, 80)).astype(np.float32)
    ir = rng.random((96, 80)).astype(np.float32)
    fused = 0.6 * vis + 0.4 * ir
    support = np.zeros((96, 80), np.float32)
    support[:, :60] = 1.0
    _assert_equal(*_call_each(fused, vis, ir, None, support))


# --- N-source generalisations ---------------------------------------------

def _three(rng):
    a = rng.random((64, 64))
    b = rng.random((64, 64))
    c = rng.random((64, 64))
    return a, b, c


def test_scd_n_reduces_to_scd(rng):
    vis, ir = rng.random((64, 64)), rng.random((64, 64))
    fused = 0.5 * (vis + ir)
    mask = _centred_mask()
    support = np.zeros((64, 64), np.float32)
    support[:, :40] = 1.0
    assert scd_n(fused, [vis, ir]) == pytest.approx(scd(fused, vis, ir), abs=1e-9)
    assert scd_n(fused, [vis, ir], mask, [None, support]) == pytest.approx(
        scd(fused, vis, ir, mask, support), abs=1e-9)


def test_scd_n_three_sources_symmetric_and_different(rng):
    a, b, c = _three(rng)
    fused = (a + b + c) / 3
    forward = scd_n(fused, [a, b, c])
    permuted = scd_n(fused, [c, a, b])
    assert np.isfinite(forward)
    assert forward == pytest.approx(permuted, abs=1e-9)


def test_scd_n_supports_intersect_the_region(rng):
    a, b, c = _three(rng)
    fused = (a + b + c) / 3
    half = np.zeros((64, 64), np.float32)
    half[:, :32] = 1.0
    # Restricting c's support to the left half confines every term there.
    assert scd_n(fused, [a, b, c], supports=[None, None, half]) == pytest.approx(
        scd_n(fused[:, :32], [a[:, :32], b[:, :32], c[:, :32]]), abs=1e-9)


def test_scd_n_rejects_single_source(rng):
    with pytest.raises(ValueError):
        scd_n(rng.random((8, 8)), [rng.random((8, 8))])


def test_qabf_n_reduces_to_qabf(rng):
    vis = rng.random((64, 64)).astype(np.float32)
    ir = rng.random((64, 64)).astype(np.float32)
    fused = 0.5 * (vis + ir)
    mask = _centred_mask()
    support = np.zeros((64, 64), np.float32)
    support[:, :40] = 1.0
    assert qabf_n(fused, [vis, ir]) == pytest.approx(
        qabf(fused, vis, ir), abs=1e-9)
    assert qabf_n(fused, [vis, ir], mask, [None, support]) == pytest.approx(
        qabf(fused, vis, ir, mask, support), abs=1e-9)


def test_qabf_n_three_sources(rng):
    a, b, c = _three(rng)
    fused = (a + b + c) / 3
    value = qabf_n(fused, [a, b, c])
    assert np.isfinite(value) and 0.0 <= value <= 1.0
    assert value == pytest.approx(qabf_n(fused, [c, a, b]), abs=1e-9)


def test_qabf_n_support_weight_zeroes_outside_footprint(rng):
    a, b, c = _three(rng)
    fused = (a + b + c) / 3
    # c's support is empty: its weight is zero everywhere, so the result is
    # the two-source value over a and b.
    empty = np.zeros((64, 64), np.float32)
    assert qabf_n(fused, [a, b, c], supports=[None, None, empty]) == pytest.approx(
        qabf_n(fused, [a, b]), abs=1e-9)


def test_ms_ssim_fusion_n_reduces_to_ms_ssim_fusion(rng):
    vis, ir = rng.random((64, 64)), rng.random((64, 64))
    fused = 0.5 * (vis + ir)
    mask = _centred_mask()
    support = np.zeros((64, 64), np.float32)
    support[:, :40] = 1.0
    assert ms_ssim_fusion_n(fused, [vis, ir]) == pytest.approx(
        ms_ssim_fusion(fused, vis, ir), abs=1e-9)
    assert ms_ssim_fusion_n(fused, [vis, ir], mask, [None, support]) == pytest.approx(
        ms_ssim_fusion(fused, vis, ir, mask, support), abs=1e-9)


def test_ms_ssim_fusion_n_drops_absent_source(rng):
    a, b, c = _three(rng)
    fused = (a + b + c) / 3
    empty = np.zeros((64, 64), np.float32)
    # c's interior is empty: its term drops and the mean renormalises.
    assert ms_ssim_fusion_n(fused, [a, b, c],
                            supports=[None, None, empty]) == pytest.approx(
        ms_ssim_fusion_n(fused, [a, b]), abs=1e-9)


def test_ms_ssim_fusion_n_all_empty_is_undefined(rng):
    a, b = rng.random((64, 64)), rng.random((64, 64))
    empty = np.zeros((64, 64), np.float32)
    assert np.isnan(ms_ssim_fusion_n(a, [a, b], supports=[empty, empty]))


def test_n_source_rejects_mismatched_shapes(rng):
    a, b = rng.random((64, 64)), rng.random((64, 64))
    small = rng.random((16, 16))
    for fn in (scd_n, qabf_n, ms_ssim_fusion_n):
        with pytest.raises(ValueError):
            fn(a, [a, small])
        with pytest.raises(ValueError):
            fn(a, [a, b], supports=[None])

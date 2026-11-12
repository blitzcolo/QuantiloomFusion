"""Physical complementarity of VIS and LWIR, independent of any fusion algorithm.

Reference grid: the IR band (default LWIR). ``oracle_warp(ref=IR, src=VIS)``
gives the correspondable region (``Warp.valid``); every cross-band pixel pair,
and every number below, lives only there. VIS features are sampled onto the IR
grid; VIS is low-pass filtered to the IR pixel footprint before sampling.

Targets are ``truth_surface_temperature`` (never ``tapp``) and the truth
material class of the IR band. Regression target is the per-frame de-meaned
surface temperature. LWIR inputs are the apparent temperature, de-meaned
per frame the same way. All results are separate per illumination condition,
with cross-validation folds in the result: leave-one-group-out by split_group
(cross-scene) and leave-one-view-group-out inside each scene (within-scene).

Masks: temperature regression and conditional mutual information use every
correspondable pixel with a valid temperature, material and features. Only the
contrast decomposition also needs a finite band emissivity; materials defined by
optical constants (sea water, bare metal) have none and enter everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..core.bands import Band
from ..dataset.reader import ThreeBandDataset
from ..dataset.splits import leave_one_group_out
from ..dataset.types import Condition, Sample
from ..geometry.warp import apply_warp, load_depth_view, oracle_warp
from ..signal.normalize import VisDisplay
from .complementarity_contrast import BoundaryContrast, boundary_contrast, band_radiance
from .complementarity_stats import (
    CmiBins,
    CmiEstimate,
    Fold,
    FrameId,
    PairArrays,
    PairFrame,
    RegressionScores,
    cmi_for_fold,
    folds_for_condition,
    pair_frame,
    regression_scores,
    view_group_folds,
)

L_DOWN_MODES = ("none", "scene_mean")
_LUMA = np.array([0.2126, 0.7152, 0.0722])


@dataclass(frozen=True)
class AnalysisSettings:
    ref_band: Band = Band.LWIR
    src_band: Band = Band.VIS
    max_px_per_frame: int = 6000
    seed: int = 0
    ridge_alpha: float = 1.0
    knn_k: int = 20
    cmi_bins: CmiBins = field(default_factory=CmiBins)
    cmi_permutations: int = 20
    view_group_blocks: int | None = None  # None: one view group per distinct azimuth


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    scores: RegressionScores
    cmi_temperature: CmiEstimate
    cmi_material: CmiEstimate


@dataclass(frozen=True)
class ConditionResult:
    condition: Condition
    mode: str  # 'logo' (cross-scene folds exist) or 'none' (single split_group)
    group_samples: dict[str, tuple[str, ...]]  # split_group -> sample ids of this condition
    group_assets: dict[str, str]  # split_group -> asset name
    folds: tuple[FoldResult, ...]  # leave-one-group-out (cross-scene); empty without mode 'logo'
    within_scene_folds: dict[str, tuple[FoldResult, ...]]  # split_group -> leave-one-view-group-out
    n_regression_pixels: int  # pixels entering regression/CMI (before subsampling)
    n_contrast_pixels: int  # of those, pixels with a finite emissivity (contrast decomposition)
    null_emissivity_pixels: dict[str, int]  # material name -> regression pixels without emissivity
    contrast: dict[str, BoundaryContrast]  # by L_down mode, pooled over the condition
    contrast_by_group: dict[str, dict[str, BoundaryContrast]]


@dataclass(frozen=True)
class ComplementarityReport:
    dataset_id: str
    ref_band: Band
    src_band: Band
    groups: tuple[str, ...]
    logo_possible: bool
    conditions: dict[Condition, ConditionResult]
    settings: AnalysisSettings


def _asset_of(sample: Sample) -> str:
    return sample.scene_id.removesuffix("_night").removesuffix("_moonlight")


def _vis_features(rgb: np.ndarray, footprint_ratio: float) -> np.ndarray:
    """(H, W, 6): r, g, b and luminance at the IR footprint, coarse luminance, gradient."""
    s = 0.5 * footprint_ratio
    rgb_s = ndimage.gaussian_filter(rgb.astype(np.float64), sigma=(s, s, 0))
    lum = rgb_s @ _LUMA
    coarse = ndimage.gaussian_filter(rgb.astype(np.float64) @ _LUMA, sigma=3.0 * footprint_ratio)
    grad = np.hypot(ndimage.sobel(lum, axis=0), ndimage.sobel(lum, axis=1))
    return np.dstack([rgb_s, lum, coarse, grad])


def _lwir_features(tapp: np.ndarray) -> np.ndarray:
    t = tapp.astype(np.float64)
    smooth = ndimage.gaussian_filter(t, 1.0)
    grad = np.hypot(ndimage.sobel(smooth, axis=0), ndimage.sobel(smooth, axis=1))
    return np.dstack([t, ndimage.gaussian_filter(t, 3.0), grad])


@dataclass(frozen=True)
class _BuiltFrame:
    frame: PairFrame
    contrast: dict[str, BoundaryContrast]
    n_regression: int
    n_contrast: int
    null_emissivity: dict[int, int]  # material label code -> pixels without emissivity


def pixel_masks(
    warp_valid: np.ndarray,
    temp: np.ndarray,
    eps: np.ndarray,
    label: np.ndarray,
    x_vis: np.ndarray,
    x_lwir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(regression mask, contrast mask); the latter additionally needs a finite emissivity."""
    regression = (
        warp_valid
        & (temp > 0)
        & (label >= 0)
        & np.isfinite(x_vis).all(axis=2)
        & np.isfinite(x_lwir).all(axis=2)
    )
    return regression, regression & np.isfinite(eps)


def _build_frame(
    dataset: ThreeBandDataset,
    sample: Sample,
    index: int,
    settings: AnalysisSettings,
    name_codes: dict[str, int],
) -> _BuiltFrame:
    ref, src = sample.observations[settings.ref_band], sample.observations[settings.src_band]
    warp = oracle_warp(
        load_depth_view(dataset, ref),
        load_depth_view(dataset, src),
        ref_band=settings.ref_band,
        src_band=settings.src_band,
    )
    temp = dataset.load(ref, "truth_surface_temperature").astype(np.float64)
    mat_id = dataset.load(ref, "truth_material_id").astype(np.int64)
    tapp = dataset.load(ref, "tapp")
    materials = dataset.load_json(ref, "truth_materials")
    band_nm = (float(materials[0]["band_nm"][0]), float(materials[0]["band_nm"][1]))
    eps_lut = np.full(max(m["material_id"] for m in materials) + 1, np.nan)
    code_lut = np.full(eps_lut.size, -1, dtype=np.intp)
    for m in materials:
        eps_lut[m["material_id"]] = m["band_emissivity"]
        code_lut[m["material_id"]] = name_codes.setdefault(m["name"], len(name_codes))
    in_table = mat_id < eps_lut.size
    eps = np.where(in_table, eps_lut[np.minimum(mat_id, eps_lut.size - 1)], np.nan)
    label = np.where(in_table, code_lut[np.minimum(mat_id, code_lut.size - 1)], -1)

    rgb = VisDisplay()(dataset.load(src, "display"), src)
    ratio = src.geometry.width / ref.geometry.width
    x_vis_full = apply_warp(_vis_features(rgb, ratio), warp)
    x_lwir_full = _lwir_features(tapp)
    valid, contrast_valid = pixel_masks(warp.valid, temp, eps, label, x_vis_full, x_lwir_full)
    contrast = {}
    for mode in L_DOWN_MODES:
        l_down = (
            float(band_radiance(temp[contrast_valid].mean(), band_nm))
            if mode == "scene_mean" and contrast_valid.any()
            else 0.0
        )
        contrast[mode] = boundary_contrast(temp, eps, label, contrast_valid, band_nm, l_down)
    codes, counts = np.unique(label[valid & ~contrast_valid], return_counts=True)
    null_eps = {int(c): int(n) for c, n in zip(codes, counts, strict=True)}
    ident = FrameId(sample.sample_id, sample.split_group, sample.condition)
    arrays = PairArrays(temp[valid], x_vis_full[valid], x_lwir_full[valid], label[valid])
    rng = np.random.default_rng([settings.seed, index])
    frame = pair_frame(ident, arrays, settings.max_px_per_frame, rng)
    return _BuiltFrame(frame, contrast, int(valid.sum()), int(contrast_valid.sum()), null_eps)


def _evaluate_fold(
    fold: Fold, frames: dict[str, PairFrame], settings: AnalysisSettings, rng: np.random.Generator
) -> FoldResult:
    train = [frames[s] for s in fold.train_ids]
    test = [frames[s] for s in fold.test_ids]
    scores = regression_scores(train, test, settings.ridge_alpha, settings.knn_k)
    cmi_t, cmi_m = cmi_for_fold(train, test, settings.cmi_bins, settings.cmi_permutations, rng)
    return FoldResult(fold, scores, cmi_t, cmi_m)


def analyse_dataset(
    dataset: ThreeBandDataset, settings: AnalysisSettings | None = None
) -> ComplementarityReport:
    """Run every analysis on a dataset; one ConditionResult per condition present."""
    settings = settings or AnalysisSettings()
    samples = list(dataset)
    meta = {s.sample_id: (s.split_group, s.condition) for s in samples}
    azimuth = {s.sample_id: float(s.parameters.azimuth_deg) for s in samples}
    assets = {s.split_group: _asset_of(s) for s in samples}
    groups = tuple(dict.fromkeys(s.split_group for s in samples))
    logo = leave_one_group_out(dataset)
    name_codes: dict[str, int] = {}
    built = {
        s.sample_id: _build_frame(dataset, s, i, settings, name_codes)
        for i, s in enumerate(samples)
    }
    frames = {k: v.frame for k, v in built.items()}
    rng = np.random.default_rng(settings.seed)
    results: dict[Condition, ConditionResult] = {}
    for condition in Condition:
        ids = [s.sample_id for s in samples if s.condition == condition]
        if not ids:
            continue
        folds = folds_for_condition(logo, meta, condition) if len(groups) > 1 else []
        group_samples = {g: tuple(i for i in ids if meta[i][0] == g) for g in groups}
        within = {
            g: view_group_folds([(i, azimuth[i]) for i in group_samples[g]], g, settings.view_group_blocks)
            for g in groups
            if group_samples[g]
        }
        code_name = {c: n for n, c in name_codes.items()}
        null_px: dict[str, int] = {}
        for i in ids:
            for c, n in built[i].null_emissivity.items():
                null_px[code_name[c]] = null_px.get(code_name[c], 0) + n
        by_group = {
            g: {m: sum((built[i].contrast[m] for i in group_samples[g]), BoundaryContrast()) for m in L_DOWN_MODES}
            for g in groups
            if group_samples[g]
        }
        total = {m: sum((by_group[g][m] for g in by_group), BoundaryContrast()) for m in L_DOWN_MODES}
        results[condition] = ConditionResult(
            condition,
            "logo" if folds else "none",
            {g: v for g, v in group_samples.items() if v},
            {g: assets[g] for g in group_samples if group_samples[g]},
            tuple(_evaluate_fold(f, frames, settings, rng) for f in folds),
            {g: tuple(_evaluate_fold(f, frames, settings, rng) for f in fs) for g, fs in within.items()},
            sum(built[i].n_regression for i in ids),
            sum(built[i].n_contrast for i in ids),
            dict(sorted(null_px.items())),
            total,
            by_group,
        )
    return ComplementarityReport(dataset.dataset_id, settings.ref_band, settings.src_band, groups, len(groups) > 1, results, settings)

"""Markdown tables for a ComplementarityReport (one section per condition).

R^2 = 1 - SSE / sum(y^2), y = per-frame de-meaned surface temperature: the
reference is the zero predictor (each held-out frame's own mean temperature),
not the training mean. R^2 < 0 means worse than that reference; a "reduction"
between two models is only meaningful next to their R^2.
"""

from __future__ import annotations

from collections.abc import Sequence

from .complementarity import L_DOWN_MODES, ComplementarityReport, ConditionResult, FoldResult
from .complementarity_stats import r2_vs_zero

_REGRESSION_HEADER = [
    "n px", "total var", "ridge VIS", "ridge VIS+LWIR", "reduction", "R2 VIS", "R2 VIS+LWIR",
    "kNN VIS", "kNN VIS+LWIR", "reduction", "R2 VIS", "R2 VIS+LWIR",
]


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows), ""]


def _group_name(res: ConditionResult, group: str) -> str:
    return f"{res.group_assets[group]} ({group[:8]})"


def _pooled(folds: Sequence[FoldResult], pick) -> float:
    w = [f.scores.n_test for f in folds]
    return float(sum(wi * pick(f) for wi, f in zip(w, folds, strict=True)) / sum(w))


def _regression_cells(n: int, total: float, rv: float, rb: float, kv: float, kb: float) -> list[str]:
    return [
        str(n), f"{total:.2f}", f"{rv:.2f}", f"{rb:.2f}", f"{1 - rb / rv:+.1%}",
        f"{r2_vs_zero(rv, total):+.2f}", f"{r2_vs_zero(rb, total):+.2f}",
        f"{kv:.2f}", f"{kb:.2f}", f"{1 - kb / kv:+.1%}",
        f"{r2_vs_zero(kv, total):+.2f}", f"{r2_vs_zero(kb, total):+.2f}",
    ]


def _pooled_cells(folds: Sequence[FoldResult]) -> list[str]:
    picks = (lambda f: f.scores.var_total, lambda f: f.scores.ridge_vis, lambda f: f.scores.ridge_both,
             lambda f: f.scores.knn_vis, lambda f: f.scores.knn_both)
    total, rv, rb, kv, kb = (_pooled(folds, p) for p in picks)
    return _regression_cells(sum(f.scores.n_test for f in folds), total, rv, rb, kv, kb)


def _fold_cells(f: FoldResult) -> list[str]:
    s = f.scores
    return _regression_cells(s.n_test, s.var_total, s.ridge_vis, s.ridge_both, s.knn_vis, s.knn_both)


def _logo_rows(res: ConditionResult) -> list[list[str]]:
    rows = [[", ".join(_group_name(res, g) for g in f.fold.test_groups), *_fold_cells(f)] for f in res.folds]
    rows.append(["**pooled (pixel-weighted)**", *_pooled_cells(res.folds)])
    return rows


def _within_rows(res: ConditionResult) -> list[list[str]]:
    rows = []
    for g, folds in res.within_scene_folds.items():
        if not folds:
            continue
        views = ", ".join(v for f in folds for v in f.fold.test_views)
        rows.append([f"{_group_name(res, g)}: {len(folds)} folds, held-out views [{views}] deg",
                     *_pooled_cells(folds)])
    all_folds = [f for fs in res.within_scene_folds.values() for f in fs]
    if len(res.within_scene_folds) > 1 and all_folds:
        n_scenes = sum(1 for fs in res.within_scene_folds.values() if fs)
        rows.append([f"**pooled over {n_scenes} scenes, {len(all_folds)} folds**", *_pooled_cells(all_folds)])
    return rows


def _cmi_cells(f: FoldResult) -> tuple[str, str]:
    t, m = f.cmi_temperature, f.cmi_material
    return (f"{t.corrected:.3f} [{t.plugin:.3f}, {t.null_mean:.3f}]",
            f"{m.corrected:.3f} [{m.plugin:.3f}, {m.null_mean:.3f}]")


def _cmi_rows(res: ConditionResult) -> list[list[str]]:
    rows = [["cross-scene, held-out " + ", ".join(_group_name(res, g) for g in f.fold.test_groups), *_cmi_cells(f)]
            for f in res.folds]
    for g, folds in res.within_scene_folds.items():
        if not folds:
            continue
        mean_t = sum(f.cmi_temperature.corrected for f in folds) / len(folds)
        mean_m = sum(f.cmi_material.corrected for f in folds) / len(folds)
        rows.append([f"within-scene {_group_name(res, g)}, mean over {len(folds)} folds", f"{mean_t:.3f}", f"{mean_m:.3f}"])
    return rows


def _condition_section(report: ComplementarityReport, res: ConditionResult) -> list[str]:
    cross = res.mode == "logo"
    out = [f"### {res.condition.value}", ""]
    if not cross:
        out += ["> Only one split_group exists: leave-one-group-out (cross-scene) is impossible. "
                "Only the within-scene split below is available; it is NOT evidence of cross-scene generalisation.", ""]
    out += [f"Cross-scene folds: **{len(res.folds)}** (leave-one-group-out by split_group); "
            f"groups in this condition: **{len(res.group_samples)}**. "
            f"Within-scene folds: **{sum(len(f) for f in res.within_scene_folds.values())}** "
            "(leave-one-view-group-out inside each scene; samples sharing an azimuth, or an azimuth block, "
            "are held out together).", ""]
    out += _table(["group (asset)", "samples"],
                  [[_group_name(res, g), f"{len(ids)}: " + ", ".join(ids)] for g, ids in res.group_samples.items()])
    if res.folds:
        out += ["**Cross-scene fold composition**", ""]
        out += _table(["fold", "test group(s)", "train group(s)", "n test / n train samples"],
                      [[str(i), ", ".join(_group_name(res, g) for g in f.fold.test_groups),
                        ", ".join(_group_name(res, g) for g in f.fold.train_groups),
                        f"{len(f.fold.test_ids)} / {len(f.fold.train_ids)}"] for i, f in enumerate(res.folds)])
    out += ["**VIS-unpredictable surface temperature (held-out MSE of per-frame de-meaned T_s, K^2; "
            "R^2 = 1 - SSE/SS_total against the zero predictor)**", ""]
    if res.folds:
        out += ["Cross-scene (leave-one-group-out):", ""]
        out += _table(["held-out", *_REGRESSION_HEADER], _logo_rows(res))
    if any(res.within_scene_folds.values()):
        out += ["Within-scene (leave-one-view-group-out; train and test never share a view group):", ""]
        out += _table(["scene: folds, held-out views", *_REGRESSION_HEADER], _within_rows(res))
    out += ["**Conditional mutual information I(Y; LWIR | VIS), nats (permutation-bias-corrected; "
            "plug-in and null in brackets for cross-scene folds)**", ""]
    out += _table(["split", "I(T_s; LWIR|VIS)", "I(material; LWIR|VIS)"], _cmi_rows(res))
    null_px = ", ".join(f"{k}: {v}" for k, v in res.null_emissivity_pixels.items()) or "none"
    out += [f"Pixels: **{res.n_regression_pixels}** enter regression/CMI (before per-frame subsampling); "
            f"**{res.n_contrast_pixels}** of them have a finite band emissivity and enter the contrast "
            f"decomposition. Without emissivity (regression/CMI only), by material: {null_px}.", ""]
    out += ["**Boundary contrast decomposition (IR band radiance, 4-neighbour cross-material pairs)**", ""]
    out += _table(["L_down", "pairs", "mean |dL| (W/m2/sr)", "mean |temperature term|", "mean |emissivity term|",
                   "temperature share", "pairs temp-dominant"],
                  [[m, str(c.n_pairs), f"{c.sum_abs_d_radiance / c.n_pairs:.3f}",
                    f"{c.sum_abs_temperature / c.n_pairs:.3f}", f"{c.sum_abs_emissivity / c.n_pairs:.3f}",
                    f"{c.temperature_share:.1%}", f"{c.temperature_dominant_fraction:.1%}"]
                   for m in L_DOWN_MODES if (c := res.contrast[m]).n_pairs])
    return out


def render_markdown(report: ComplementarityReport) -> str:
    s = report.settings
    out = [f"## Dataset `{report.dataset_id[:12]}`: {report.src_band.value} vs {report.ref_band.value} "
           f"(reference grid: {report.ref_band.value})", "",
           f"split_groups: {len(report.groups)}; leave-one-group-out possible: **{report.logo_possible}**. "
           f"Pixels/frame <= {s.max_px_per_frame}; ridge alpha {s.ridge_alpha}; kNN k {s.knn_k}; "
           f"CMI bins {s.cmi_bins}; permutations {s.cmi_permutations}; view-group blocks {s.view_group_blocks}; "
           f"seed {s.seed}.", ""]
    for res in report.conditions.values():
        out += _condition_section(report, res)
    return "\n".join(out)

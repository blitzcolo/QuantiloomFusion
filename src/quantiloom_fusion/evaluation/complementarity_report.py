"""Markdown tables for a ComplementarityReport (one section per condition)."""

from __future__ import annotations

from .complementarity import L_DOWN_MODES, ComplementarityReport, ConditionResult, FoldResult


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows), ""]


def _group_name(res: ConditionResult, group: str) -> str:
    return f"{res.group_assets[group]} ({group[:8]})"


def _pooled(folds: tuple[FoldResult, ...], pick) -> float:
    w = [f.scores.n_test for f in folds]
    return float(sum(wi * pick(f) for wi, f in zip(w, folds, strict=True)) / sum(w))


def _regression_rows(res: ConditionResult) -> list[list[str]]:
    def line(label: str, n: int, vals: tuple[float, ...]) -> list[str]:
        total, rv, rb, kv, kb = vals
        return [
            label, str(n), f"{total:.2f}",
            f"{rv:.2f}", f"{rb:.2f}", f"{1 - rb / rv:+.1%}",
            f"{kv:.2f}", f"{kb:.2f}", f"{1 - kb / kv:+.1%}",
        ]

    def vals(s) -> tuple[float, ...]:
        return (s.var_total, s.ridge_vis, s.ridge_both, s.knn_vis, s.knn_both)

    rows = [line(", ".join(_group_name(res, g) for g in f.fold.test_groups), f.scores.n_test, vals(f.scores)) for f in res.folds]
    picks = (lambda f: f.scores.var_total, lambda f: f.scores.ridge_vis, lambda f: f.scores.ridge_both,
             lambda f: f.scores.knn_vis, lambda f: f.scores.knn_both)
    pooled = tuple(_pooled(res.folds, p) for p in picks)
    rows.append(line("**pooled (pixel-weighted)**", sum(f.scores.n_test for f in res.folds), pooled))
    return rows


def _condition_section(report: ComplementarityReport, res: ConditionResult) -> list[str]:
    cross = res.mode == "logo"
    out = [f"### {res.condition.value}", ""]
    if not cross:
        out += ["> WITHIN-GROUP ONLY, NOT cross-scene: only one split_group exists, so leave-one-group-out is "
                "impossible. Folds are contiguous azimuth blocks of the same scene; neighbouring azimuths share "
                "scene content, so numbers are optimistic and are not evidence of cross-scene generalisation.", ""]
    out += [f"Folds: **{len(res.folds)}** ({'leave-one-group-out by split_group' if cross else 'within-group azimuth blocks'}); "
            f"groups in this condition: **{len(res.group_samples)}**.", ""]
    out += _table(["group (asset)", "samples"],
                  [[_group_name(res, g), f"{len(ids)}: " + ", ".join(ids)] for g, ids in res.group_samples.items()])
    out += ["**Fold composition**", ""]
    out += _table(["fold", "test group(s)", "train group(s)", "n test / n train samples"],
                  [[str(i), ", ".join(_group_name(res, g) for g in f.fold.test_groups),
                    ", ".join(_group_name(res, g) for g in f.fold.train_groups),
                    f"{len(f.fold.test_ids)} / {len(f.fold.train_ids)}"] for i, f in enumerate(res.folds)])
    out += ["**VIS-unpredictable surface temperature (held-out MSE of per-frame de-meaned T_s, K^2)**", ""]
    out += _table(["held-out", "n px", "total var", "ridge VIS", "ridge VIS+LWIR", "reduction",
                   "kNN VIS", "kNN VIS+LWIR", "reduction"], _regression_rows(res))
    out += ["**Conditional mutual information I(Y; LWIR | VIS), nats (permutation-bias-corrected; plug-in and null in brackets)**", ""]
    out += _table(["held-out", "I(T_s; LWIR|VIS)", "I(material; LWIR|VIS)"],
                  [[", ".join(_group_name(res, g) for g in f.fold.test_groups),
                    f"{f.cmi_temperature.corrected:.3f} [{f.cmi_temperature.plugin:.3f}, {f.cmi_temperature.null_mean:.3f}]",
                    f"{f.cmi_material.corrected:.3f} [{f.cmi_material.plugin:.3f}, {f.cmi_material.null_mean:.3f}]"]
                   for f in res.folds])
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
           f"CMI bins {s.cmi_bins}; permutations {s.cmi_permutations}; seed {s.seed}.", ""]
    for res in report.conditions.values():
        out += _condition_section(report, res)
    return "\n".join(out)

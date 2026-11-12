"""The thesis evaluation protocol: sensitivity of fusion metrics to registration error.

One result row = (operator, registration, perturbation amplitude, layer group),
carrying the normalisation choice, the calibrated/uncalibrated flag, the truth
mode and the registration error (EPE).  No-reference metrics (``ctx_*``) are
context only; the inflation experiment is the one place they are shown against
misregistration, to demonstrate that they reward ghosting.
"""

from __future__ import annotations

import csv
import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.bands import Band
from ..fusion import FusionInput, FusionMethod, aligned_sources
from ..geometry.warp import Warp
from ..registration.base import registration_error
from .layer_report import FusedView, layer_masks, layer_metrics
from .perturb import inverse_depth_bias_warp, smooth_field_warp
from .scene import EvalScene
from .truth_ref import TruthMode, clean_input, clean_source_metrics, truth_metrics, truth_reference

CALIBRATED: dict[str, bool] = {
    "oracle": True, "infinity_homography": True, "epipolar_sweep": True, "free_search": False,
    "resize": False,
}


def is_calibrated(registration: str) -> bool:
    """Whether a registration method uses the camera calibration (perturbed oracles inherit the flag)."""
    return CALIBRATED[registration.split("+")[0]]


class PerturbationKind(enum.StrEnum):
    INVERSE_DEPTH_BIAS = "inverse_depth_bias"
    SMOOTH_FIELD = "smooth_field"


@dataclass(frozen=True)
class Sweep:
    """Perturbation family, its amplitudes (1/m for the bias, px for the field) and the truth mode."""

    kind: PerturbationKind
    amplitudes: Sequence[float]
    truth_mode: TruthMode | None = None
    corr_len_px: float = 8.0
    seed: int = 0


def perturbed_warps(scene: EvalScene, sweep: Sweep, amplitude: float) -> dict[Band, Warp]:
    match sweep.kind:
        case PerturbationKind.INVERSE_DEPTH_BIAS:
            return {b: inverse_depth_bias_warp(scene.views[scene.reference], scene.views[b], w, amplitude)
                    for b, w in scene.oracle_warps.items()}
        case PerturbationKind.SMOOTH_FIELD:
            return {b: smooth_field_warp(w, amplitude, sweep.corr_len_px, sweep.seed + i)
                    for i, (b, w) in enumerate(scene.oracle_warps.items())}


def epe_columns(scene: EvalScene, warps: dict[Band, Warp]) -> dict[str, float]:
    """EPE against the oracle on its correspondable pixels, averaged over the source bands."""
    stats = {b: registration_error(w, scene.oracle_warps[b]) for b, w in warps.items()}
    mean_px = float(np.nanmean([s["epe_mean"] for s in stats.values()]))
    mrad = float(np.nanmean([s["epe_mean"] / scene.geometry[b].fx * 1000.0 for b, s in stats.items()]))
    return {"epe_mean_px": mean_px, "epe_median_px": float(np.nanmean([s["epe_median"] for s in stats.values()])),
            "epe_p95_px": float(np.nanmean([s["epe_p95"] for s in stats.values()])), "epe_mean_mrad": mrad}


def evaluate_point(scene: EvalScene, operator: FusionMethod, warps: dict[Band, Warp],
                   truth_mode: TruthMode | None) -> list[dict]:
    """Fuse once and report every layer group (rows without registration/amplitude columns)."""
    x = FusionInput(scene.reference, scene.images, scene.geometry, warps)
    fused = operator.fuse(x).image
    sources, supports = aligned_sources(x)
    view = FusedView(fused, sources, supports, scene.reference)
    reference, rule = None, ""
    if truth_mode is not None:
        if scene.clean is None:
            raise ValueError(f"{scene.sample_id} has no co-view clean images; truth-referenced metrics need them")
        if not scene.norm.get("scene_independent", True):
            raise ValueError("truth-referenced metrics need a fixed normalisation window: a per-image "
                             "percentile window maps the clean and the noisy image differently")
        supports = ({b: w.support for b, w in scene.oracle_warps.items()}
                    if truth_mode == TruthMode.REGISTRATION else None)
        reference, rule = truth_reference(truth_mode, operator,
                                          clean_input(scene.clean, scene.geometry, scene.reference, supports))
    rows = []
    for group, mask in layer_masks(scene.layers).items():
        row = {"operator": operator.name, "truth_mode": truth_mode.value if truth_mode else "",
               "truth_rule": rule} | scene.norm_columns() | layer_metrics(view, group, mask)
        if reference is not None:
            row |= truth_metrics(fused, reference, mask)
            if truth_mode == TruthMode.FUSION:
                row |= clean_source_metrics(fused, scene.clean, mask)
        rows.append(row)
    return rows


def sensitivity_curve(scene: EvalScene, operator: FusionMethod, sweep: Sweep) -> list[dict]:
    """Rows for every amplitude and layer group: EPE next to the fusion metrics."""
    rows = []
    for amplitude in sweep.amplitudes:
        warps = perturbed_warps(scene, sweep, amplitude)
        registration = next(iter(warps.values())).provenance["method"]
        base = {"sample_id": scene.sample_id, "registration": registration,
                "calibrated": is_calibrated(registration), "perturbation": sweep.kind.value,
                "amplitude": float(amplitude)} | epe_columns(scene, warps)
        rows += [base | r for r in evaluate_point(scene, operator, warps, sweep.truth_mode)]
    return rows


def inflation_experiment(scene: EvalScene, operator: FusionMethod, sweep: Sweep) -> list[dict]:
    """Correspondable-layer ctx_AG / ctx_SF relative to the amplitude-0 value, per amplitude.

    Ghosting adds edges, so ratios above 1 mean the no-reference metric rewards
    misregistration.
    """
    rows = [r for r in sensitivity_curve(scene, operator, sweep) if r["layer"] == "correspondable"]
    base = next(r for r in rows if r["amplitude"] == 0.0)
    return [r | {"ctx_AG_ratio": r["ctx_AG"] / base["ctx_AG"], "ctx_SF_ratio": r["ctx_SF"] / base["ctx_SF"]}
            for r in rows]


def write_csv(rows: Sequence[dict], path: Path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_epe_vs_metric(rows: Sequence[dict], metric: str, path: Path) -> bool:
    """EPE-vs-metric curve per operator on the correspondable layer; False without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for op in dict.fromkeys(r["operator"] for r in rows):
        pts = [(r["epe_mean_px"], r[metric]) for r in rows
               if r["operator"] == op and r["layer"] == "correspondable" and not math.isnan(r[metric])]
        ax.plot(*zip(*sorted(pts)), marker="o", label=op)
    ax.set_xlabel("EPE vs oracle (px)")
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True

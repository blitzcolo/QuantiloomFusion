"""Registration method interface and registration-error metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..dataset.types import Observation
from ..geometry.warp import Warp


class Registrar(ABC):
    """Estimates a Warp that aligns a source band onto a reference band."""

    name: str

    @abstractmethod
    def estimate(
        self,
        ref: Observation,
        src: Observation,
        ref_image: np.ndarray,
        src_image: np.ndarray,
    ) -> Warp: ...


def registration_error(estimate: Warp, oracle: Warp) -> dict[str, float]:
    """EPE stats in src pixels over oracle.valid & estimate.valid & finite map."""
    if estimate.ref_band != oracle.ref_band or estimate.src_band != oracle.src_band:
        raise ValueError(
            f"band mismatch: estimate {estimate.ref_band}->{estimate.src_band}, "
            f"oracle {oracle.ref_band}->{oracle.src_band}"
        )
    if estimate.map_xy.shape != oracle.map_xy.shape:
        raise ValueError(f"grid mismatch: {estimate.map_xy.shape} vs {oracle.map_xy.shape}")
    m = oracle.valid & estimate.valid & np.isfinite(estimate.map_xy).all(axis=2)
    n = int(m.sum())
    epe = np.linalg.norm(estimate.map_xy - oracle.map_xy, axis=2)[m]
    nan = float("nan")
    return {
        "epe_mean": float(epe.mean()) if n else nan,
        "epe_median": float(np.median(epe)) if n else nan,
        "epe_p95": float(np.percentile(epe, 95)) if n else nan,
        "epe_max": float(epe.max()) if n else nan,
        "n": float(n),
        "coverage": n / int(oracle.valid.sum()) if oracle.valid.any() else 0.0,
    }

"""Dense warp fields mapping one band's frame onto a reference band."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.bands import Band
from ..dataset.types import Observation


@dataclass(frozen=True)
class Warp:
    """map_xy gives, for each reference pixel, the source pixel to sample."""

    ref_band: Band
    src_band: Band
    map_xy: np.ndarray  # (H_ref, W_ref, 2) float
    valid: np.ndarray  # (H_ref, W_ref) bool
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.map_xy.ndim != 3 or self.map_xy.shape[2] != 2:
            raise ValueError(f"map_xy must be (H, W, 2), got {self.map_xy.shape}")
        if self.valid.shape != self.map_xy.shape[:2]:
            raise ValueError(f"valid shape {self.valid.shape} != map_xy shape {self.map_xy.shape[:2]}")
        if self.valid.dtype != bool:
            raise ValueError(f"valid must be bool, got {self.valid.dtype}")


def oracle_warp(ref: Observation, src: Observation, *, edge_px: int, z_rel_tol: float) -> Warp:
    """Ground-truth warp from oracle reprojection + bidirectional z-test."""
    raise NotImplementedError


def apply_warp(src_image: np.ndarray, warp: Warp, interp: str = "linear") -> np.ndarray:
    """Sample src_image at warp.map_xy; invalid pixels become NaN."""
    raise NotImplementedError

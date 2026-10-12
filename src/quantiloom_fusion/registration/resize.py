"""ResizeRegistrar: FOV-scaling lower bound (same axis, no distortion/rotation)."""

from __future__ import annotations

import numpy as np

from ..dataset.types import Observation
from ..geometry.camera import pixel_grid
from ..geometry.warp import Warp
from .base import Registrar


class ResizeRegistrar(Registrar):
    """Align by sensor scaling only: u_src = (u_ref - c_ref) * f_src/f_ref + c_src.

    Pixel-index coords: index = continuous - 0.5 on both sides.
    """

    name = "resize"

    def estimate(
        self,
        ref: Observation,
        src: Observation,
        ref_image: np.ndarray,
        src_image: np.ndarray,
    ) -> Warp:
        gr, gs = ref.geometry, src.geometry
        grid = pixel_grid(gr)
        x = (grid[..., 0] + 0.5 - gr.cx) * (gs.fx / gr.fx) + gs.cx - 0.5
        y = (grid[..., 1] + 0.5 - gr.cy) * (gs.fy / gr.fy) + gs.cy - 0.5
        map_xy = np.stack([x, y], axis=-1)
        valid = (x >= 0) & (x <= gs.width - 1) & (y >= 0) & (y <= gs.height - 1)
        return Warp(
            ref_band=ref.band,
            src_band=src.band,
            map_xy=map_xy,
            valid=valid,
            provenance={"method": self.name},
        )

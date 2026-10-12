"""InfinityHomographyRegistrar: known K/distortion/rotation, ignores parallax."""

from __future__ import annotations

import numpy as np

from ..dataset.types import Observation
from ..geometry.camera import BrownConradyCamera, pixel_grid
from ..geometry.warp import Warp
from .base import Registrar


class InfinityHomographyRegistrar(Registrar):
    """Map ref rays into the src frame via R_src @ R_ref^T and reproject.

    Exact for coincident centres; baselines produce parallax error that grows
    as f * b / depth — the point of this baseline is to measure it.
    """

    name = "infinity_homography"

    def estimate(
        self,
        ref: Observation,
        src: Observation,
        ref_image: np.ndarray,
        src_image: np.ndarray,
    ) -> Warp:
        gr, gs = ref.geometry, src.geometry
        ref_cam = BrownConradyCamera(gr)
        src_cam = BrownConradyCamera(gs)
        rays = ref_cam.unproject(pixel_grid(gr).reshape(-1, 2), np.ones(gr.height * gr.width))
        r_ref = np.asarray(gr.world_to_camera).reshape(4, 4)[:3, :3]
        r_src = np.asarray(gs.world_to_camera).reshape(4, 4)[:3, :3]
        pts = rays @ (r_src @ r_ref.T).T  # rotation only; baseline ignored
        map_xy = src_cam.project(pts).reshape(gr.height, gr.width, 2)
        x, y = map_xy[..., 0], map_xy[..., 1]
        valid = (
            (pts[:, 2] > 0).reshape(gr.height, gr.width)
            & (x >= 0) & (x <= gs.width - 1) & (y >= 0) & (y <= gs.height - 1)
        )
        return Warp(
            ref_band=ref.band,
            src_band=src.band,
            map_xy=map_xy,
            valid=valid,
            provenance={"method": self.name},
        )

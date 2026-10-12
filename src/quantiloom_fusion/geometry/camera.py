"""Brown-Conrady camera model built from a product's CameraGeometry."""

from __future__ import annotations

import numpy as np

from ..dataset.types import CameraGeometry


class BrownConradyCamera:
    """Pinhole camera with k1,k2,p1,p2,k3 distortion, from dataset provenance."""

    def __init__(self, geometry: CameraGeometry) -> None:
        self.geometry = geometry

    @classmethod
    def from_geometry(cls, g: CameraGeometry) -> BrownConradyCamera:
        return cls(g)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project (N,3) camera-space points to (N,2) pixel coordinates."""
        raise NotImplementedError

    def unproject(self, pixels: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Lift (N,2) pixel coordinates with depth z to (N,3) camera-space points."""
        raise NotImplementedError

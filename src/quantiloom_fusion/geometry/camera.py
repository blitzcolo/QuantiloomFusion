"""Brown-Conrady camera model built from a product's CameraGeometry.

All pixel coordinates use the package index convention: pixel (row i, col j)
has centre (x=j, y=i), i.e. index coords = continuous image coords - 0.5.
"""

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

    def _distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward OpenCV-form Brown-Conrady map on normalized coords."""
        k1, k2, p1, p2, k3 = self.geometry.distortion
        r2 = x * x + y * y
        rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        return (
            x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        )

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project (N,3) camera points to (N,2) pixel-index coords; NaN if Z <= 0."""
        g = self.geometry
        pts = np.asarray(points_cam, dtype=np.float64)
        z = pts[:, 2]
        out = np.full((len(pts), 2), np.nan)
        front = z > 0
        xd, yd = self._distort(pts[front, 0] / z[front], pts[front, 1] / z[front])
        out[front, 0] = g.fx * xd + g.cx - 0.5
        out[front, 1] = g.fy * yd + g.cy - 0.5
        return out

    def unproject(self, pixels: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Lift (N,2) pixel-index coords at depth z to (N,3) camera points.

        The distortion map is inverted by fixed-point iteration (the OpenCV
        scheme); these lenses converge to ~1e-15 normalized within 20 steps.
        """
        g = self.geometry
        px = np.asarray(pixels, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        xd = (px[:, 0] + 0.5 - g.cx) / g.fx
        yd = (px[:, 1] + 0.5 - g.cy) / g.fy
        k1, k2, p1, p2, k3 = g.distortion
        x, y = xd.copy(), yd.copy()
        for _ in range(20):
            r2 = x * x + y * y
            ic = 1.0 / (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3)))
            nx = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) * ic
            ny = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) * ic
            step = max(np.abs(nx - x).max(initial=0.0), np.abs(ny - y).max(initial=0.0))
            x, y = nx, ny
            if step < 1e-15:
                break
        return np.stack([x * z, y * z, z], axis=1)


def to_camera(g: CameraGeometry, points_world: np.ndarray) -> np.ndarray:
    """Apply world_to_camera to (N,3) world points -> (N,3) camera points."""
    w2c = np.asarray(g.world_to_camera, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(points_world, dtype=np.float64)
    return pts @ w2c[:3, :3].T + w2c[:3, 3]


def to_world(g: CameraGeometry, points_cam: np.ndarray) -> np.ndarray:
    """Apply camera_to_world to (N,3) camera points -> (N,3) world points."""
    c2w = np.asarray(g.camera_to_world, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(points_cam, dtype=np.float64)
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def pixel_grid(g: CameraGeometry) -> np.ndarray:
    """(H, W, 2) float64 grid of (x, y) pixel-index coords for g's image."""
    j, i = np.meshgrid(np.arange(g.width), np.arange(g.height))
    return np.stack([j, i], axis=-1).astype(np.float64)

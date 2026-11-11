"""Calibration-constrained epipolar inverse-depth sweep, and a 2D free-search control.

With known intrinsics/extrinsics the residual after the infinity homography
lies on the epipolar line and its size depends only on depth. For inverse
depth d = 1 / z (1/m) the ref pixel u lands at

    u'(u; d) = pi_s( R_{s<-r} * pi_r^{-1}(u) + d * t_{s<-r} )

with pi_r^{-1}(u) the undistorted z = 1 ray; d = 0 is exactly the infinity
homography. A sweep scores each hypothesis with a local similarity
(registration.similarity), keeps the best per pixel, and falls back to d = 0
where the peak is not salient, the reference is textureless, or the reverse
(src -> ref) sweep disagrees.
"""

from __future__ import annotations

from collections.abc import Iterator
import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from ..dataset.types import CameraGeometry, Observation
from ..geometry.camera import BrownConradyCamera, pixel_grid
from ..geometry.warp import Warp
from .base import Registrar
from .params import SweepParams
from .prepare import prepare, structure_ok
from .similarity import Features, features, score

_NO_SCORE = -1e9


@dataclass(frozen=True)
class _Rig:
    """Rotation and baseline (m) taking `a` camera points to `b` camera points."""

    rays: np.ndarray  # (H*W, 3) undistorted unit-depth rays of a's pixel grid
    rot_rays: np.ndarray  # rays rotated into b's frame (the d = 0 point)
    t_m: np.ndarray  # (3,)
    cam_b: BrownConradyCamera
    shape: tuple[int, int]


def _rig(ga: CameraGeometry, gb: CameraGeometry) -> _Rig:
    rays = BrownConradyCamera(ga).unproject(pixel_grid(ga).reshape(-1, 2), np.ones(ga.height * ga.width))
    w2c_a = np.asarray(ga.world_to_camera).reshape(4, 4)
    w2c_b = np.asarray(gb.world_to_camera).reshape(4, 4)
    c2w_a = np.asarray(ga.camera_to_world).reshape(4, 4)
    t_world = w2c_b[:3, :3] @ c2w_a[:3, 3] + w2c_b[:3, 3]
    return _Rig(
        rays=rays,
        rot_rays=rays @ (w2c_b[:3, :3] @ w2c_a[:3, :3].T).T,
        t_m=t_world * ga.world_units_to_meters,
        cam_b=BrownConradyCamera(gb),
        shape=(ga.height, ga.width),
    )


def _project(rig: _Rig, d: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 2) b-pixel coords for inverse depth d (scalar or (H, W)) and the b-frame Z."""
    d_flat = d if np.isscalar(d) else np.asarray(d).reshape(-1, 1)
    pts = rig.rot_rays + d_flat * rig.t_m
    return rig.cam_b.project(pts).reshape(*rig.shape, 2), pts[:, 2].reshape(rig.shape)


def _in_bounds(map_xy: np.ndarray, gb: CameraGeometry) -> np.ndarray:
    x, y = map_xy[..., 0], map_xy[..., 1]
    return (x >= 0) & (x <= gb.width - 1) & (y >= 0) & (y <= gb.height - 1)


def _score_stack(ref_f: Features, src_img: np.ndarray, maps: Iterator[np.ndarray],
                 gb: CameraGeometry, params: SweepParams, rng: tuple[float, float]) -> np.ndarray:
    """(n, H, W) float32 scores; hypotheses whose source sample is out of bounds get _NO_SCORE."""
    out = []
    for map_xy in maps:
        ok = np.isfinite(map_xy).all(axis=2) & _in_bounds(map_xy, gb)
        mx = np.where(ok, map_xy[..., 0], -1.0).astype(np.float32)
        my = np.where(ok, map_xy[..., 1], -1.0).astype(np.float32)
        warped = cv2.remap(src_img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        s = score(params.similarity, ref_f, features(warped, params.similarity, params.bins, rng),
                  params.window, params.bins)
        out.append(np.where(ok, s, _NO_SCORE).astype(np.float32))
    return np.stack(out)


def _pick(scores: np.ndarray, grid_dist: np.ndarray, exclude: int, tex: np.ndarray, params: SweepParams
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best hypothesis index, peak saliency in [0, 1], and the confident mask per pixel."""
    best_idx = scores.argmax(axis=0)
    best = np.take_along_axis(scores, best_idx[None], axis=0)[0]
    has = scores > _NO_SCORE / 2
    worst = np.where(has, scores, np.inf).min(axis=0)
    worst = np.where(np.isfinite(worst), worst, best)
    second = np.full(best.shape, _NO_SCORE, np.float32)
    for k in range(len(scores)):
        far = grid_dist[k][best_idx] > exclude
        second = np.maximum(second, np.where(far, scores[k], _NO_SCORE))
    second = np.where(second > _NO_SCORE / 2, second, worst)
    saliency = (best - second) / (best - worst + 1e-9)
    saliency = np.where(best > _NO_SCORE / 2, saliency, 0.0)
    confident = (best > _NO_SCORE / 2) & (tex >= params.min_texture) & (saliency >= params.min_saliency)
    return best_idx, saliency, confident


@dataclass(frozen=True)
class _Sweep:
    d_values: np.ndarray
    px_per_d: float  # median src-pixel displacement per unit inverse depth
    best_idx: np.ndarray
    saliency: np.ndarray
    confident: np.ndarray


def _epipolar_sweep(ref_g: CameraGeometry, src_g: CameraGeometry, ref_image: np.ndarray,
                    src_image: np.ndarray, params: SweepParams) -> tuple[_Sweep, _Rig]:
    rig = _rig(ref_g, src_g)
    d_values = np.linspace(0.0, 1.0 / params.z_min_m, params.n_hypotheses)
    prep = prepare(ref_g, src_g, ref_image, src_image, params)
    maps = (_project(rig, float(d))[0] for d in d_values)
    scores = _score_stack(prep.ref_f, prep.src, maps, src_g, params, prep.rng)
    idx = np.arange(len(d_values))
    step = d_values[1] - d_values[0] if len(d_values) > 1 else 1.0 / params.z_min_m
    base, _ = _project(rig, 0.0)
    px_per_d = float(np.nanmedian(np.linalg.norm(_project(rig, step)[0] - base, axis=2))) / step
    exclude = max(1, math.ceil(params.separation_px / (px_per_d * step)))
    best_idx, saliency, confident = _pick(scores, np.abs(idx[:, None] - idx[None, :]), exclude, prep.tex, params)
    confident &= structure_ok(prep, _project(rig, d_values[best_idx])[0], params.noise_factor)
    return _Sweep(d_values, px_per_d, best_idx, saliency, confident), rig


class EpipolarSweepRegistrar(Registrar):
    """Per-pixel inverse-depth sweep along the calibrated epipolar line."""

    name = "epipolar_sweep"

    def __init__(self, params: SweepParams | None = None) -> None:
        self.params = params or SweepParams()

    def _consistent(self, fwd: _Sweep, rig: _Rig, ref: Observation, src: Observation,
                    ref_image: np.ndarray, src_image: np.ndarray) -> np.ndarray:
        """Forward estimate agrees with the src -> ref sweep at the landing pixel."""
        p = self.params
        rev, _ = _epipolar_sweep(src.geometry, ref.geometry, src_image, ref_image,
                                 replace(p, window=max(5, round(p.window / _scale(ref, src)) | 1)))
        d_fwd = fwd.d_values[fwd.best_idx]
        map_xy, z_s = _project(rig, d_fwd)
        d_in_src = d_fwd / np.where(z_s > 0, z_s, np.nan)
        gs = src.geometry
        ix = np.clip(np.rint(np.nan_to_num(map_xy[..., 0])), 0, gs.width - 1).astype(np.int64)
        iy = np.clip(np.rint(np.nan_to_num(map_xy[..., 1])), 0, gs.height - 1).astype(np.int64)
        d_rev = rev.d_values[rev.best_idx][iy, ix]
        return rev.confident[iy, ix] & (np.abs(d_in_src - d_rev) <= p.lr_tol_px / fwd.px_per_d)

    def estimate(self, ref: Observation, src: Observation, ref_image: np.ndarray,
                 src_image: np.ndarray) -> Warp:
        p = self.params
        fwd, rig = _epipolar_sweep(ref.geometry, src.geometry, ref_image, src_image, p)
        confident = fwd.confident & (fwd.best_idx > 0)
        if p.left_right and p.n_hypotheses > 1:
            confident = confident & self._consistent(fwd, rig, ref, src, ref_image, src_image)
        d_map = np.where(confident, fwd.d_values[fwd.best_idx], 0.0)
        map_xy, z_s = _project(rig, d_map)
        return Warp(
            ref_band=ref.band,
            src_band=src.band,
            map_xy=map_xy,
            valid=(z_s > 0) & _in_bounds(map_xy, src.geometry),
            provenance={
                "method": self.name,
                "similarity": str(p.similarity),
                "hypotheses": fwd.d_values,
                "z_min_m": p.z_min_m,
                "d_map": d_map,
                "confidence": np.where(confident, fwd.saliency, 0.0),
                "confident": confident,
                "params": p.__dict__,
            },
        )


def _scale(ref: Observation, src: Observation) -> float:
    """Reference-to-source focal length ratio (ref pixels per src pixel)."""
    return ref.geometry.fx / src.geometry.fx


class FreeSearchRegistrar(Registrar):
    """Control: 2D integer-grid offset search around the infinity homography.

    The search square is +-D with D the epipolar sweep's maximum displacement
    (|u'(d_max) - u'(0)| over the image), so only the epipolar constraint
    differs. Same similarity, saliency, texture and structure gates; no left-right check.
    """

    name = "free_search"

    def __init__(self, params: SweepParams | None = None, step_px: float = 1.0) -> None:
        self.params = params or SweepParams()
        self.step_px = step_px

    def estimate(self, ref: Observation, src: Observation, ref_image: np.ndarray,
                 src_image: np.ndarray) -> Warp:
        p, gs = self.params, src.geometry
        rig = _rig(ref.geometry, gs)
        base, _ = _project(rig, 0.0)
        far, _ = _project(rig, 1.0 / p.z_min_m)
        max_disp = float(np.nanmax(np.linalg.norm(far - base, axis=2)))
        half = max(1, int(np.ceil(max_disp / self.step_px)))
        axis = np.arange(-half, half + 1) * self.step_px
        offsets = np.array([(dx, dy) for dy in axis for dx in axis])
        grid_dist = np.abs(np.rint(offsets[:, None] / self.step_px) - np.rint(offsets[None] / self.step_px)).max(axis=2)
        prep = prepare(ref.geometry, gs, ref_image, src_image, p)
        scores = _score_stack(prep.ref_f, prep.src, (base + o for o in offsets), gs, p, prep.rng)
        best_idx, saliency, confident = _pick(scores, grid_dist, math.ceil(p.separation_px / self.step_px), prep.tex, p)
        confident &= structure_ok(prep, base + offsets[best_idx], p.noise_factor)
        shift = np.where(confident[..., None], offsets[best_idx], 0.0)
        map_xy = base + shift
        return Warp(
            ref_band=ref.band,
            src_band=src.band,
            map_xy=map_xy,
            valid=np.isfinite(map_xy).all(axis=2) & _in_bounds(map_xy, gs),
            provenance={
                "method": self.name,
                "similarity": str(p.similarity),
                "max_displacement_px": max_disp,
                "step_px": self.step_px,
                "n_offsets": len(offsets),
                "offset_map": shift,
                "confidence": np.where(confident, saliency, 0.0),
                "confident": confident,
                "params": p.__dict__,
            },
        )

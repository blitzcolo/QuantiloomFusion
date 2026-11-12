"""Dense warp fields mapping one band's frame onto a reference band.

All coordinates are pixel-index coords (see quantiloom_fusion.geometry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..core.bands import Band
from ..dataset.types import CameraGeometry, Observation
from .camera import BrownConradyCamera, pixel_grid, to_camera

if TYPE_CHECKING:
    from ..dataset.reader import ThreeBandDataset


@dataclass(frozen=True)
class Warp:
    """map_xy gives, for each reference pixel, the source pixel to sample.

    Two masks for two uses. ``valid`` is where registration error is
    measured. ``support`` is where the source is sampled for fusion: the
    pixels at which the source camera sees the same surface point. They
    differ for the oracle, whose ``valid`` leaves out the depth-edge band:
    there the correspondence is true but a pixel footprint straddles fore-
    and background, the mixing a real camera has even when perfectly
    registered, so it belongs in fusion and not in error statistics.
    ``support`` defaults to ``valid`` and must contain it.
    """

    ref_band: Band
    src_band: Band
    map_xy: np.ndarray  # (H_ref, W_ref, 2) float
    valid: np.ndarray  # (H_ref, W_ref) bool
    provenance: dict = field(default_factory=dict)
    support: np.ndarray | None = None  # (H_ref, W_ref) bool; None -> valid

    def __post_init__(self) -> None:
        if self.map_xy.ndim != 3 or self.map_xy.shape[2] != 2:
            raise ValueError(f"map_xy must be (H, W, 2), got {self.map_xy.shape}")
        if self.valid.shape != self.map_xy.shape[:2]:
            raise ValueError(f"valid shape {self.valid.shape} != map_xy shape {self.map_xy.shape[:2]}")
        if self.valid.dtype != bool:
            raise ValueError(f"valid must be bool, got {self.valid.dtype}")
        if self.support is None:
            object.__setattr__(self, "support", self.valid)
        elif self.support.shape != self.valid.shape or self.support.dtype != bool:
            raise ValueError(f"support must be a bool {self.valid.shape} mask")
        elif (self.valid & ~self.support).any():
            raise ValueError("support must contain valid")


@dataclass(frozen=True)
class DepthView:
    """The truth products one observation needs for oracle reprojection."""

    geometry: CameraGeometry
    camera_z: np.ndarray  # (H, W) float, truth_camera_z
    world_position: np.ndarray  # (H, W, 3) float, truth_world_position (X, Y, Z)
    validity: np.ndarray  # (H, W) int, truth_validity codes


def load_depth_view(dataset: ThreeBandDataset, observation: Observation) -> DepthView:
    """Load the camera_z / world_position / validity truth products of one band."""
    return DepthView(
        geometry=observation.geometry,
        camera_z=dataset.load(observation, "truth_camera_z").astype(np.float64),
        world_position=dataset.load(observation, "truth_world_position", ("X", "Y", "Z")).astype(np.float64),
        validity=dataset.load(observation, "truth_validity").astype(np.int32),
    )


def _edge_band(view: DepthView, edge_px: int, edge_rel_tol: float) -> np.ndarray:
    """Surface-discontinuity mask dilated by edge_px (square structuring element).

    Seeds are pixels whose second difference of world_position along a row or
    column exceeds edge_rel_tol * camera_z (robust on slanted surfaces, where
    first differences are large but the surface is continuous), or which
    4-neighbour a validity != 1 pixel.
    """
    z = view.camera_z
    p = view.world_position
    d2 = np.zeros(z.shape)
    d2[:, 1:-1] = np.maximum(
        d2[:, 1:-1],
        np.linalg.norm(p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2], axis=2) / z[:, 1:-1],
    )
    d2[1:-1, :] = np.maximum(
        d2[1:-1, :],
        np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=2) / z[1:-1, :],
    )
    seed = d2 > edge_rel_tol
    bad = view.validity != 1
    seed[:-1] |= bad[1:]
    seed[1:] |= bad[:-1]
    seed[:, :-1] |= bad[:, 1:]
    seed[:, 1:] |= bad[:, :-1]
    for _ in range(edge_px):
        padded = np.pad(seed, 1)
        seed = np.logical_or.reduce(
            [padded[i:i + z.shape[0], j:j + z.shape[1]] for i in range(3) for j in range(3)]
        )
    return seed


def _oracle_masks(
    ref: DepthView,
    src: DepthView,
    edge_px: int,
    edge_rel_tol: float,
    pos_rel_tol: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """map_xy plus the per-criterion masks shared by oracle_warp/diagnostics."""
    h, w = ref.camera_z.shape
    hs, ws = src.camera_z.shape
    cam_pts = to_camera(src.geometry, ref.world_position.reshape(-1, 3))
    map_xy = BrownConradyCamera(src.geometry).project(cam_pts).reshape(h, w, 2)
    map_xy[ref.validity != 1] = np.nan
    x, y = map_xy[..., 0], map_xy[..., 1]
    base = (ref.validity == 1) & np.isfinite(map_xy).all(axis=2)
    inside = base & (x >= 0) & (x <= ws - 1) & (y >= 0) & (y <= hs - 1)

    xf = np.floor(np.nan_to_num(x)).astype(np.int64)
    yf = np.floor(np.nan_to_num(y)).astype(np.int64)
    x0, x1 = xf.clip(0, ws - 1), (xf + 1).clip(0, ws - 1)
    y0, y1 = yf.clip(0, hs - 1), (yf + 1).clip(0, hs - 1)
    corners = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    src_support = np.logical_and.reduce([src.validity[a, b] == 1 for a, b in corners])

    wx = (x - np.floor(np.nan_to_num(x)))[..., None]
    wy = (y - np.floor(np.nan_to_num(y)))[..., None]
    wp = src.world_position
    p_src = (
        wp[y0, x0] * ((1 - wx) * (1 - wy))
        + wp[y0, x1] * (wx * (1 - wy))
        + wp[y1, x0] * ((1 - wx) * wy)
        + wp[y1, x1] * (wx * wy)
    )
    # Src seeing the same 3D surface point is the two-view visibility test:
    # an occluder or a src depth discontinuity between the 4 neighbours moves
    # the interpolated point off the surface.
    corr = np.linalg.norm(p_src - ref.world_position, axis=2) <= pos_rel_tol * ref.camera_z

    ref_edge = _edge_band(ref, edge_px, edge_rel_tol)
    src_edge = _edge_band(src, edge_px, edge_rel_tol)
    src_edge_at = np.logical_or.reduce([src_edge[a, b] for a, b in corners])
    valid = inside & src_support & corr & ~ref_edge & ~src_edge_at
    masks = {
        "base": base,
        "inside": inside,
        "src_support": src_support,
        "correspondence": corr,
        "ref_edge": ref_edge,
        "src_edge": src_edge_at,
        "valid": valid,
    }
    return map_xy, masks


def _sky_correspondence(ref: DepthView, src: DepthView) -> tuple[np.ndarray, np.ndarray]:
    """Map and mask of ref misses whose direction the source also sees as a miss.

    A miss (truth_validity 0) is sky at infinite distance: the baseline does
    not matter there, so its source pixel is the rotation-only reprojection
    of the ref ray. Kept where all 4 bilinear source neighbours are misses too.
    """
    gr, gs = ref.geometry, src.geometry
    h, w = ref.camera_z.shape
    hs, ws = src.camera_z.shape
    rays = BrownConradyCamera(gr).unproject(pixel_grid(gr).reshape(-1, 2), np.ones(h * w))
    r_ref = np.asarray(gr.world_to_camera, np.float64).reshape(4, 4)[:3, :3]
    r_src = np.asarray(gs.world_to_camera, np.float64).reshape(4, 4)[:3, :3]
    map_xy = BrownConradyCamera(gs).project(rays @ (r_src @ r_ref.T).T).reshape(h, w, 2)
    x, y = map_xy[..., 0], map_xy[..., 1]
    inside = np.isfinite(map_xy).all(axis=2) & (x >= 0) & (x <= ws - 1) & (y >= 0) & (y <= hs - 1)
    xf = np.floor(np.nan_to_num(x)).astype(np.int64)
    yf = np.floor(np.nan_to_num(y)).astype(np.int64)
    x0, x1 = xf.clip(0, ws - 1), (xf + 1).clip(0, ws - 1)
    y0, y1 = yf.clip(0, hs - 1), (yf + 1).clip(0, hs - 1)
    src_sky = np.logical_and.reduce([src.validity[a, b] == 0 for a, b in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))])
    return map_xy, (ref.validity == 0) & inside & src_sky


def oracle_warp(
    ref: DepthView,
    src: DepthView,
    *,
    ref_band: Band,
    src_band: Band,
    edge_px: int = 1,
    edge_rel_tol: float = 0.02,
    pos_rel_tol: float = 0.005,
) -> Warp:
    """Ground-truth warp: oracle reprojection + two-view position test.

    map_xy[i,j] is the src pixel-index coord of ref.world_position[i,j]: finite
    wherever the point is geometrically projectable (ref validity == 1 and
    src Z > 0), NaN otherwise. valid additionally requires inside bounds, all
    4 bilinear src neighbours valid, the sampled src surface point within
    pos_rel_tol * camera_z of the ref point, and neither edge band. support
    is valid with the edge bands put back (inside & src_support &
    correspondence), plus sky both cameras see (ref misses whose source
    neighbours are misses too, mapped at infinity, where map_xy is finite as
    well): only occluded and out-of-view pixels and other invalid ref pixels
    are left out.
    """
    map_xy, masks = _oracle_masks(ref, src, edge_px, edge_rel_tol, pos_rel_tol)
    sky_xy, sky = _sky_correspondence(ref, src)
    map_xy[sky] = sky_xy[sky]
    return Warp(
        ref_band=ref_band,
        src_band=src_band,
        map_xy=map_xy,
        valid=masks["valid"],
        support=(masks["inside"] & masks["src_support"] & masks["correspondence"]) | sky,
        provenance={
            "method": "oracle",
            "edge_px": edge_px,
            "edge_rel_tol": edge_rel_tol,
            "pos_rel_tol": pos_rel_tol,
        },
    )


def oracle_diagnostics(
    ref: DepthView,
    src: DepthView,
    *,
    edge_px: int = 1,
    edge_rel_tol: float = 0.02,
    pos_rel_tol: float = 0.005,
) -> dict[str, int]:
    """Per-criterion failure counts among ref-valid projectable pixels.

    Counts are not mutually exclusive: each is measured against the whole
    ref-valid base, so a pixel can fail several criteria at once.
    """
    _, m = _oracle_masks(ref, src, edge_px, edge_rel_tol, pos_rel_tol)
    base = m["base"]
    return {
        "ref_valid": int(base.sum()),
        "outside": int((base & ~m["inside"]).sum()),
        "src_support": int((base & ~m["src_support"]).sum()),
        "correspondence": int((base & ~m["correspondence"]).sum()),
        "ref_edge": int((base & m["ref_edge"]).sum()),
        "src_edge": int((base & m["src_edge"]).sum()),
        "final_valid": int(m["valid"].sum()),
    }


def apply_warp(src_image: np.ndarray, warp: Warp, interp: str = "linear") -> np.ndarray:
    """Sample src_image at warp.map_xy (index coords); ~support or out-of-range -> NaN.

    interp is "linear" (bilinear, never reads out of bounds) or "nearest".
    float inputs keep their dtype; integer inputs become float32.
    """
    img = np.asarray(src_image)
    if img.ndim not in (2, 3):
        raise ValueError(f"src_image must be (H, W) or (H, W, C), got {img.shape}")
    if interp not in ("linear", "nearest"):
        raise ValueError(f"unknown interp {interp!r}")
    hs, ws = img.shape[:2]
    x, y = warp.map_xy[..., 0], warp.map_xy[..., 1]
    ok = warp.support & np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= ws - 1) & (y >= 0) & (y <= hs - 1)
    dtype = img.dtype if np.issubdtype(img.dtype, np.floating) else np.float32
    out = np.full(warp.map_xy.shape[:2] + img.shape[2:], np.nan, dtype=dtype)
    if not ok.any():
        return out
    if interp == "nearest":
        out[ok] = img[np.rint(y[ok]).astype(np.int64), np.rint(x[ok]).astype(np.int64)]
        return out
    x0 = np.floor(x[ok]).astype(np.int64)
    y0 = np.floor(y[ok]).astype(np.int64)
    wx = (x[ok] - x0)[:, None] if img.ndim == 3 else x[ok] - x0
    wy = (y[ok] - y0)[:, None] if img.ndim == 3 else y[ok] - y0
    x0c = x0.clip(0, ws - 1)
    x1c = (x0 + 1).clip(0, ws - 1)
    y0c = y0.clip(0, hs - 1)
    y1c = (y0 + 1).clip(0, hs - 1)
    out[ok] = (
        img[y0c, x0c] * (1 - wx) * (1 - wy)
        + img[y0c, x1c] * wx * (1 - wy)
        + img[y1c, x0c] * (1 - wx) * wy
        + img[y1c, x1c] * wx * wy
    )
    return out

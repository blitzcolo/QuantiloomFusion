"""Controlled perturbations of the oracle correspondence.

Both perturbations start from an oracle :class:`Warp` and keep its ``valid``
mask, so registration error is always measured on the same correspondable
pixels.  Amplitude 0 returns the oracle ``map_xy`` bit for bit.

* inverse-depth bias: every reference pixel is reprojected with
  ``d = 1/z_true + delta_d`` instead of ``1/z_true``, i.e. the point slides
  along its reference ray (the epipolar line in the source image):
  ``u'(u; d) = pi_s(R pi_r^-1(u) + d t)`` up to the common scale.
* smooth random displacement field: Gaussian-filtered white noise with a
  correlation length (sigma of the filter, px) rescaled to a prescribed RMS
  displacement magnitude (px); seeded.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry.camera import BrownConradyCamera, to_camera
from ..geometry.warp import DepthView, Warp


def inverse_depth_bias_warp(ref: DepthView, src: DepthView, oracle: Warp, delta_d: float) -> Warp:
    """Oracle warp with the inverse depth shifted by ``delta_d`` (1/m)."""
    z = ref.camera_z
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / z + delta_d
        scale = np.ones_like(z) if delta_d == 0.0 else (1.0 / inv) / z
    centre = np.asarray(ref.geometry.camera_to_world, np.float64).reshape(4, 4)[:3, 3]
    points = ref.world_position + (scale - 1.0)[..., None] * (ref.world_position - centre)
    h, w = z.shape
    map_xy = BrownConradyCamera(src.geometry).project(
        to_camera(src.geometry, points.reshape(-1, 3))).reshape(h, w, 2)
    map_xy[ref.validity != 1] = np.nan
    map_xy[~(inv > 0)] = np.nan
    valid = oracle.valid & np.isfinite(map_xy).all(axis=2)
    return Warp(oracle.ref_band, oracle.src_band, map_xy, valid,
                {"method": "oracle+inverse_depth_bias", "delta_d": float(delta_d)})


def smooth_field_warp(oracle: Warp, amplitude_px: float, corr_len_px: float, seed: int) -> Warp:
    """Oracle warp plus a smooth random displacement of RMS magnitude ``amplitude_px``."""
    h, w = oracle.valid.shape
    noise = np.random.default_rng(seed).standard_normal((h, w, 2))
    field = ndimage.gaussian_filter(noise, sigma=(corr_len_px, corr_len_px, 0), mode="reflect")
    rms = float(np.sqrt(np.mean(np.sum(field**2, axis=-1))))
    disp = field * (amplitude_px / rms)
    return Warp(oracle.ref_band, oracle.src_band, oracle.map_xy + disp, oracle.valid.copy(),
                {"method": "oracle+smooth_field", "amplitude_px": float(amplitude_px),
                 "corr_len_px": float(corr_len_px), "seed": int(seed)})

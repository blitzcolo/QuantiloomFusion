"""Colour helpers: BT.601 full-range YCbCr and bilinear Bayer demosaicing.

The YCbCr convention is the full-range ("JPEG") one used by YCbCr-based
IR/VIS fusion code: Y in [0, 1] and Cb/Cr centred on 0.5. Fusion uses this
to fuse luminance and keep the VIS chroma: colour output = fused Y with the
VIS Cb/Cr put back.

Bayer patterns are named by the 2x2 top-left block of the mosaic
("rggb": row 0 = R G, row 1 = G B). numpy only.
"""

from __future__ import annotations

import numpy as np

_KR, _KB = 0.299, 0.114
_KG = 1.0 - _KR - _KB
_RGB_TO_YCC = np.array(
    [
        [_KR, _KG, _KB],
        [-0.5 * _KR / (1.0 - _KB), -0.5 * _KG / (1.0 - _KB), 0.5],
        [0.5, -0.5 * _KG / (1.0 - _KR), -0.5 * _KB / (1.0 - _KR)],
    ]
)
_YCC_TO_RGB = np.linalg.inv(_RGB_TO_YCC)
_OFFSET = np.array([0.0, 0.5, 0.5])

# pattern name -> 2x2 block of channel indices (r=0, g=1, b=2)
_PATTERNS = {
    "rggb": ((0, 1), (1, 2)),
    "bggr": ((2, 1), (1, 0)),
    "grbg": ((1, 0), (2, 1)),
    "gbrg": ((1, 2), (0, 1)),
}


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB in [0, 1] -> (H, W, 3) YCbCr, Cb/Cr centred on 0.5."""
    return np.asarray(rgb, np.float64) @ _RGB_TO_YCC.T + _OFFSET


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_ycbcr."""
    return (np.asarray(ycc, np.float64) - _OFFSET) @ _YCC_TO_RGB.T


def luminance(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) -> (H, W) BT.601 luma."""
    return np.asarray(rgb, np.float64) @ np.array([_KR, _KG, _KB])


def _box3(a: np.ndarray) -> np.ndarray:
    """3x3 box sum with reflect padding."""
    p = np.pad(a, 1, mode="reflect")
    s = np.zeros_like(a)
    for di in range(3):
        for dj in range(3):
            s += p[di : di + a.shape[0], dj : dj + a.shape[1]]
    return s


def demosaic_bilinear(raw: np.ndarray, pattern: str = "rggb") -> np.ndarray:
    """Bilinear Bayer demosaic of a (H, W) mosaic into (H, W, 3) float64.

    Each output channel is the weighted average of the nearest raw samples
    of that channel within a 3x3 window, normalized by the sample count, so
    flat fields and linear ramps are reproduced exactly in the interior.
    """
    raw = np.asarray(raw, np.float64)
    if raw.ndim != 2:
        raise ValueError(f"expected a (H, W) mosaic, got shape {raw.shape}")
    if pattern not in _PATTERNS:
        raise ValueError(f"unknown Bayer pattern {pattern!r}; expected one of {sorted(_PATTERNS)}")
    h, w = raw.shape
    block = _PATTERNS[pattern]
    out = np.empty((h, w, 3), np.float64)
    for c in range(3):
        mask = np.zeros((h, w))
        for i in range(2):
            for j in range(2):
                if block[i][j] == c:
                    mask[i::2, j::2] = 1.0
        out[..., c] = _box3(raw * mask) / _box3(mask)
    return out

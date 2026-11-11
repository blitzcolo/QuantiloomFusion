"""Five mutually exclusive layers of the reference pixel grid.

Decided in this order, first match wins:

1. INVALID: ref pixel is not a clean hit (truth_validity != 1: 0 miss,
   2 lens-invalid, 3 partial, 4 transmissive). Partial (3) is treated as
   invalid because its world position mixes surfaces and cannot anchor a
   correspondence.
2. OUT_OF_VIEW: the projected point falls outside the source image (or behind
   the source camera).
3. OCCLUDED: inside the source image, but the source has no trustworthy view of
   the same surface point. Either the bilinearly sampled source world position
   is farther than ``pos_rel_tol * depth`` from the ref point (both cameras'
   own truth is compared, not a one-way z test), or one of the 4 bilinear
   source neighbours is not a valid hit (``src_support`` failure: the source
   has no surface information there, so the position test is meaningless).
4. EDGE: has a correspondence but sits in a ref or source depth-discontinuity
   band (second difference of world position, see ``warp._edge_band``).
5. CORRESPONDABLE: the rest. Equals ``oracle_warp(...).valid`` exactly because
   the masks come from the same ``_oracle_masks`` call.

OCCLUDED and OUT_OF_VIEW pixels carry no source information; fusion there
should fall back to the reference band.
"""

from __future__ import annotations

import enum

import numpy as np

from .warp import DepthView, _oracle_masks


class Layer(enum.IntEnum):
    """Layer codes stored in the int8 map returned by ``layer_map``."""

    INVALID = 0
    OUT_OF_VIEW = 1
    OCCLUDED = 2
    EDGE = 3
    CORRESPONDABLE = 4


DEPTH_BINS_M: tuple[tuple[float, float], ...] = (
    (0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 50.0), (50.0, float("inf")),
)


def layer_map(
    ref: DepthView,
    src: DepthView,
    *,
    edge_px: int = 1,
    edge_rel_tol: float = 0.02,
    pos_rel_tol: float = 0.005,
) -> np.ndarray:
    """(H_ref, W_ref) int8 map of ``Layer`` codes; tolerances as in oracle_warp."""
    _, m = _oracle_masks(ref, src, edge_px, edge_rel_tol, pos_rel_tol)
    inside = m["inside"]
    seen = m["src_support"] & m["correspondence"]
    layers = np.full(ref.camera_z.shape, Layer.CORRESPONDABLE, dtype=np.int8)
    layers[inside & seen & ~m["valid"]] = Layer.EDGE
    layers[inside & ~seen] = Layer.OCCLUDED
    layers[~inside] = Layer.OUT_OF_VIEW
    layers[ref.validity != 1] = Layer.INVALID
    return layers


def layer_fractions(layers: np.ndarray) -> dict[str, float]:
    """Fraction of all ref pixels in each layer, keyed by layer name."""
    return {layer.name.lower(): float((layers == layer).mean()) for layer in Layer}


def layer_fractions_by_depth(
    layers: np.ndarray,
    ref_depth: np.ndarray,
    bins: tuple[tuple[float, float], ...] = DEPTH_BINS_M,
) -> dict[tuple[float, float], dict[str, float]]:
    """Per ref-depth bin [lo, hi): pixel count ``n`` and per-layer fractions of that bin.

    Pixels without a finite positive depth (misses) belong to no bin.
    """
    has_depth = np.isfinite(ref_depth) & (ref_depth > 0)
    out: dict[tuple[float, float], dict[str, float]] = {}
    for lo, hi in bins:
        sel = has_depth & (ref_depth >= lo) & (ref_depth < hi)
        n = int(sel.sum())
        frac = {layer.name.lower(): float((layers[sel] == layer).mean()) if n else float("nan") for layer in Layer}
        out[(lo, hi)] = {"n": float(n), **frac}
    return out

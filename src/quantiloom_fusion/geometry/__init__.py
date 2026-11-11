"""Brown-Conrady camera model, pixel grids, oracle reprojection, bidirectional z-test, edge bands, and validity masks. Depends on core and dataset.

All package APIs use pixel-index coordinates: pixel (row i, col j) has centre
(x=j, y=i), i.e. index = continuous image coords - 0.5. Warp.map_xy stores
(x, y) index coords per reference pixel.
"""
from .camera import BrownConradyCamera, pixel_grid, to_camera, to_world
from .layers import DEPTH_BINS_M, Layer, layer_fractions, layer_fractions_by_depth, layer_map
from .warp import (
    DepthView,
    Warp,
    apply_warp,
    load_depth_view,
    oracle_diagnostics,
    oracle_warp,
)

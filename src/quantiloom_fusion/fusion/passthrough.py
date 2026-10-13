"""The passthrough baseline: the reference band image, unchanged.

This is the single-band baseline — what "fusion" degenerates to when only
the reference exists.  It needs no registration and ignores every band but
the reference.
"""

from __future__ import annotations

import numpy as np

from ..core.bands import BANDS
from .base import Capabilities, FusionInput, FusionMethod, FusionOutput
from .registry import register


@register("reference_only")
class ReferenceOnly(FusionMethod):
    """Return ``x.images[x.reference]`` unchanged, support all-True."""

    capabilities = Capabilities(
        bands=frozenset(BANDS),
        needs_registration=False,
        heterogeneous_resolution=True,
        color_output=True,
        trainable=False,
    )

    def __init__(self):
        pass

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        ref = np.asarray(x.images[x.reference])
        return FusionOutput(ref.astype(np.float32),
                            {"support": {x.reference: np.ones(ref.shape[:2], bool)}})

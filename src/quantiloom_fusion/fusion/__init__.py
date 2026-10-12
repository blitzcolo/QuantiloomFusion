"""Fusion method interface, capabilities, registry, and baselines. Depends on core, dataset, geometry, signal, and registration."""
from .align import aligned_sources, colourize
from .base import Capabilities, FusionInput, FusionMethod, FusionOutput
from .classical import Average, LaplacianPyramid, MaxAbs
from .registry import available, create, register

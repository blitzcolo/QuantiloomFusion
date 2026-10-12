"""Fusion method interface, capabilities, registry, and baselines. Depends on core, dataset, geometry, signal, and registration."""
from .base import Capabilities, FusionInput, FusionMethod, FusionOutput
from .registry import available, create, register

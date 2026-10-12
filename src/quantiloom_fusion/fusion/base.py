"""Fusion method interface: capabilities, inputs, outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..core.bands import Band
from ..dataset.types import CameraGeometry
from ..geometry.warp import Warp


@dataclass(frozen=True)
class Capabilities:
    """What a fusion method supports, declared up front."""

    bands: frozenset[Band]
    needs_registration: bool
    heterogeneous_resolution: bool
    color_output: bool
    trainable: bool


@dataclass
class FusionInput:
    """One sample handed to a fusion method."""

    reference: Band
    images: dict[Band, np.ndarray]
    geometry: dict[Band, CameraGeometry]
    warps: dict[Band, Warp] | None = None  # per non-reference band

    def __post_init__(self) -> None:
        if self.reference not in self.images:
            raise ValueError(f"reference {self.reference} missing from images {sorted(self.images)}")
        extra = set(self.images) - set(self.geometry)
        if extra:
            raise ValueError(f"images without geometry: {sorted(extra)}")


@dataclass
class FusionOutput:
    image: np.ndarray
    extras: dict = field(default_factory=dict)


class FusionMethod(ABC):
    """A fusion method: declared capabilities plus a fuse() implementation."""

    name: str
    capabilities: Capabilities

    @abstractmethod
    def fuse(self, x: FusionInput) -> FusionOutput: ...

    def check_input(self, x: FusionInput) -> None:
        """Validate x against capabilities before fuse(); raises ValueError."""
        unsupported = set(x.images) - set(self.capabilities.bands)
        if unsupported:
            raise ValueError(f"{self.name} does not support bands {sorted(unsupported)}")
        if self.capabilities.needs_registration:
            needed = set(x.images) - {x.reference}
            missing = needed if x.warps is None else needed - set(x.warps)
            if missing:
                raise ValueError(f"{self.name} needs registration; missing warps for {sorted(missing)}")

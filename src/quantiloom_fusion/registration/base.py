"""Registration method interface and registration-error metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..dataset.types import Observation
from ..geometry.warp import Warp


class Registrar(ABC):
    """Estimates a Warp that aligns a source band onto a reference band."""

    name: str

    @abstractmethod
    def estimate(
        self,
        ref: Observation,
        src: Observation,
        ref_image: np.ndarray,
        src_image: np.ndarray,
    ) -> Warp: ...


def registration_error(estimate: Warp, oracle: Warp) -> dict[str, float]:
    """EPE mean/median/p95 in pixels over oracle.valid & estimate.valid."""
    raise NotImplementedError

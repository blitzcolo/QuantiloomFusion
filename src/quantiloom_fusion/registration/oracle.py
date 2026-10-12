"""OracleRegistrar: the geometry-truth warp as the registration upper bound."""

from __future__ import annotations

import numpy as np

from ..dataset.reader import ThreeBandDataset
from ..dataset.types import Observation
from ..geometry.warp import DepthView, Warp, load_depth_view, oracle_warp
from .base import Registrar


class OracleRegistrar(Registrar):
    """Wraps geometry.oracle_warp; input images are ignored."""

    name = "oracle"

    def __init__(self, dataset: ThreeBandDataset, **oracle_kwargs) -> None:
        self.dataset = dataset
        self.oracle_kwargs = oracle_kwargs
        self._views: dict[str, DepthView] = {}

    def _view(self, observation: Observation) -> DepthView:
        if observation.record not in self._views:
            self._views[observation.record] = load_depth_view(self.dataset, observation)
        return self._views[observation.record]

    def estimate(
        self,
        ref: Observation,
        src: Observation,
        ref_image: np.ndarray,
        src_image: np.ndarray,
    ) -> Warp:
        return oracle_warp(
            self._view(ref),
            self._view(src),
            ref_band=ref.band,
            src_band=src.band,
            **self.oracle_kwargs,
        )

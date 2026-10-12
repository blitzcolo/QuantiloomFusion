"""Normaliser protocol: product image + SignalSemantics -> float32 fusion input."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..dataset.types import SignalSemantics


@runtime_checkable
class Normalizer(Protocol):
    """Explicit, named mapping from a product signal to a fusion input.

    __call__ returns a float32 array; params() returns the normalisation
    parameters actually applied (for provenance in experiment records).
    """

    name: str

    def params(self) -> dict: ...

    def __call__(self, image: np.ndarray, semantics: SignalSemantics) -> np.ndarray: ...

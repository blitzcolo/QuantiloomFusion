"""ThreeBandDataset: typed read access to a published dataset directory."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

from ..core.jsonio import load_json
from ..core.paths import native_path
from . import exr
from .schema import (
    DatasetIndex,
    PlanSummary,
    parse_index,
    parse_plan_summary,
    parse_sample,
)
from .types import Observation, Product, Sample


class ThreeBandDataset:
    """Read a published QuantiloomFusion dataset: index, plan, samples, products."""

    def __init__(self, root: str | Path) -> None:
        self.root = native_path(root).resolve()
        self.index: DatasetIndex = parse_index(load_json(self.root / "dataset.index.json"))
        self.plan: PlanSummary = parse_plan_summary(load_json(self.root / "dataset.plan.json"))
        if self.index.dataset_id != self.plan.dataset_id:
            raise ValueError("index/plan dataset_id mismatch")
        self.dataset_id: str = self.index.dataset_id
        self._entries = {e.sample_id: e for e in self.index.entries}

    def __len__(self) -> int:
        return len(self.index.entries)

    def sample_ids(self) -> tuple[str, ...]:
        return tuple(e.sample_id for e in self.index.entries)

    def sample(self, sample_id: str) -> Sample:
        entry = self._entries.get(sample_id)
        if entry is None:
            raise KeyError(f"no such sample {sample_id!r}")
        s = parse_sample(load_json(self.root / entry.manifest))
        if s.dataset_id != self.dataset_id:
            raise ValueError(f"{sample_id}: manifest dataset_id {s.dataset_id} != index {self.dataset_id}")
        if s.split_group != entry.split_group:
            raise ValueError(f"{sample_id}: manifest split_group {s.split_group} != index {entry.split_group}")
        return s

    def __iter__(self) -> Iterator[Sample]:
        for entry in self.index.entries:
            yield self.sample(entry.sample_id)

    def path(self, product: Product) -> Path:
        rel = product.path
        if PurePosixPath(rel).is_absolute() or "\\" in rel or ":" in rel or ".." in PurePosixPath(rel).parts:
            raise ValueError(f"unsafe product path {rel!r}")
        return self.root / rel

    def load(self, observation: Observation, name: str, channels: Sequence[str] | None = None) -> np.ndarray:
        product = observation.product(name)
        if not product.path.endswith(".exr"):
            raise ValueError(f"{name!r} is not an EXR product: {product.path}")
        return exr.read_image(self.path(product), channels)

    def load_json(self, observation: Observation, name: str):
        product = observation.product(name)
        if not product.path.endswith(".json"):
            raise ValueError(f"{name!r} is not a JSON product: {product.path}")
        return load_json(self.path(product))

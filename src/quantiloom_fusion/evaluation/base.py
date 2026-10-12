"""Metric interface and the append-only experiment result store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..dataset.types import Sample
from ..fusion.base import FusionInput, FusionOutput


class Metric(ABC):
    """Scores a fusion output against its input and the sample's ground truth."""

    name: str
    version: int

    @abstractmethod
    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]: ...


@dataclass(frozen=True)
class ResultKey:
    """Identity of one metric result row."""

    dataset_id: str
    sample_id: str
    method_fingerprint: str
    metric: str
    metric_version: int


class ResultStore:
    """Append-only JSONL result store: LF newlines, existing keys are skipped."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def has(self, key: ResultKey) -> bool:
        raise NotImplementedError

    def append(self, key: ResultKey, values: dict[str, float]) -> None:
        raise NotImplementedError

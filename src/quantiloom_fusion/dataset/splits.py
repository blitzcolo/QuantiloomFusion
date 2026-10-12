"""split_group folds for leakage-free cross-validation.

Standard library only; reads the index, never sample manifests or products.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reader import ThreeBandDataset


def leave_one_group_out(
    dataset: ThreeBandDataset,
) -> list[tuple[tuple[str, ...], tuple[str, ...], str]]:
    """(train_ids, test_ids, group) per split_group in first-appearance order."""
    groups = dict.fromkeys(e.split_group for e in dataset.index.entries)
    entries = dataset.index.entries
    return [
        (
            tuple(e.sample_id for e in entries if e.split_group != g),
            tuple(e.sample_id for e in entries if e.split_group == g),
            g,
        )
        for g in groups
    ]

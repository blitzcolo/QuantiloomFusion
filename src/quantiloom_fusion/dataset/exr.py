"""EXR pixel IO for dataset products.

This is the only consumer module allowed to import OpenEXR (enforced by
tests/test_boundaries.py). Channels are always read as separate (H, W)
arrays in their stored dtype; write_image exists for test fixtures and
tools that produce dataset-shaped products.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import OpenEXR


def read_channels(path: Path, names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Read channels of one EXR file as {name: (H, W) array}.

    All channels are returned when names is None; a missing name raises
    KeyError listing the available channels. Arrays keep the stored dtype.
    """
    with OpenEXR.File(str(path), separate_channels=True) as f:
        have = f.channels()
        want = list(have) if names is None else list(names)
        missing = [n for n in want if n not in have]
        if missing:
            raise KeyError(f"{missing} not in {path}; available: {sorted(have)}")
        return {n: np.asarray(have[n].pixels) for n in want}


def read_image(path: Path, channels: Sequence[str] | None = None) -> np.ndarray:
    """Read one EXR file as (H, W) for a single channel or (H, W, C) stacked.

    channels=None uses the file's channel order, which is deterministic:
    OpenEXR stores the header channel list in lexicographic order.
    """
    data = read_channels(path, channels)
    if len(data) == 1:
        return next(iter(data.values()))
    order = list(data) if channels is None else list(channels)
    return np.stack([data[n] for n in order], axis=-1)


def read_header(path: Path) -> dict:
    """Header attributes of a single-part EXR file as a plain dict."""
    with OpenEXR.File(str(path), header_only=True) as f:
        return dict(f.header())


def write_image(path: Path, channels: Mapping[str, np.ndarray], header: Mapping | None = None) -> None:
    """Write {name: (H, W) array} channels to an EXR file (float16/float32/uint32)."""
    parts = [OpenEXR.Part(dict(header or {}), {n: np.asarray(a) for n, a in channels.items()})]
    OpenEXR.File(parts).write(str(path))

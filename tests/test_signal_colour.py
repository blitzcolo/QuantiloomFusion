"""Colour helpers: BT.601 YCbCr round-trip and bilinear Bayer demosaic."""

import numpy as np
import pytest

from quantiloom_fusion.signal.colour import (
    demosaic_bilinear,
    luminance,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


def test_ycbcr_roundtrip():
    rng = np.random.default_rng(0)
    rgb = rng.random((16, 12, 3))
    ycc = rgb_to_ycbcr(rgb)
    assert np.all(np.abs(ycc - 0.5) <= 0.5 + 1e-9)
    assert np.max(np.abs(ycbcr_to_rgb(ycc) - rgb)) < 1e-6


def test_ycbcr_known_values():
    ycc = rgb_to_ycbcr(np.array([[[1.0, 0.0, 0.0], [0.5, 0.5, 0.5]]]))
    assert ycc[0, 0, 0] == pytest.approx(0.299)  # red luma
    assert ycc[0, 0, 2] == pytest.approx(1.0)  # Cr saturates on pure red
    assert np.allclose(ycc[0, 1], [0.5, 0.5, 0.5])  # grey: neutral chroma
    assert np.allclose(luminance(np.array([[[1.0, 0.0, 0.0]]])), 0.299)


def _mosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """Sample an RGB image down to a Bayer mosaic of the given pattern."""
    blocks = {"rggb": ((0, 1), (1, 2)), "bggr": ((2, 1), (1, 0)),
              "grbg": ((1, 0), (2, 1)), "gbrg": ((1, 2), (0, 1))}
    block = blocks[pattern]
    h, w, _ = rgb.shape
    raw = np.zeros((h, w))
    for i in range(2):
        for j in range(2):
            raw[i::2, j::2] = rgb[i::2, j::2, block[i][j]]
    return raw


@pytest.mark.parametrize("pattern", ["rggb", "bggr", "grbg", "gbrg"])
def test_demosaic_constant_colour(pattern):
    colour = np.full((10, 8, 3), [0.4, 0.5, 0.6])
    out = demosaic_bilinear(_mosaic(colour, pattern), pattern)
    assert out.shape == (10, 8, 3)
    assert np.allclose(out, colour)


def test_demosaic_horizontal_ramp():
    w, h = 12, 8
    ramp = np.tile(np.linspace(0.0, 1.0, w), (h, 1))
    rgb = np.dstack([ramp, 0.5 * ramp, 1.0 - ramp])
    out = demosaic_bilinear(_mosaic(rgb, "rggb"), "rggb")
    # A linear ramp is reproduced exactly away from the border columns.
    assert np.allclose(out[1:-1, 2:-2], rgb[1:-1, 2:-2], atol=1e-12)


def test_demosaic_rejects_bad_input():
    with pytest.raises(ValueError, match="pattern"):
        demosaic_bilinear(np.zeros((4, 4)), "rgbr")
    with pytest.raises(ValueError, match="mosaic"):
        demosaic_bilinear(np.zeros((4, 4, 3)))

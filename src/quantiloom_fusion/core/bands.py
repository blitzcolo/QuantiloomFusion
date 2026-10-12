"""Canonical spectral band identifiers shared by every layer."""

from __future__ import annotations

import enum


class Band(enum.StrEnum):
    """The three sensor bands of a dataset sample."""

    VIS = "vis"
    MWIR = "mwir"
    LWIR = "lwir"


BANDS: tuple[Band, ...] = (Band.VIS, Band.MWIR, Band.LWIR)

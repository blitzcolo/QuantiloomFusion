"""Typed view of the published dataset contract (sample.json and products).

Standard library only: these types describe shapes; parsing JSON into them is
done by the dataset reader, and EXR IO lives in dataset/exr.py.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass

from ..core.bands import BANDS, Band


class Condition(enum.StrEnum):
    """Illumination condition of a scene, derived from the scene id suffix."""

    DAY = "day"
    NIGHT = "night"
    MOONLIGHT = "moonlight"


def condition_of(scene_id: str) -> Condition:
    """The single place a scene id maps to its illumination condition."""
    if scene_id.endswith("_night"):
        return Condition.NIGHT
    if scene_id.endswith("_moonlight"):
        return Condition.MOONLIGHT
    return Condition.DAY


@dataclass(frozen=True)
class CameraGeometry:
    """provenance.geometry of one product: intrinsics, distortion, poses."""

    width: int
    height: int
    intrinsics: tuple[float, ...]  # 9, row-major
    distortion: tuple[float, ...]  # 5: k1, k2, p1, p2, k3 (Brown-Conrady)
    camera_to_world: tuple[float, ...]  # 16, row-major
    world_to_camera: tuple[float, ...]  # 16, row-major
    max_theta_radians: float | None = None
    camera_axes: str = "right_down_forward"
    pixel_center_offset: tuple[float, float] = (0.5, 0.5)
    pixel_origin: str = "top_left"
    world_units_to_meters: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"non-positive image size {self.width}x{self.height}")
        for name, want in (
            ("intrinsics", 9),
            ("distortion", 5),
            ("camera_to_world", 16),
            ("world_to_camera", 16),
        ):
            if len(getattr(self, name)) != want:
                raise ValueError(f"{name} must have {want} elements, got {len(getattr(self, name))}")

    @property
    def fx(self) -> float:
        return self.intrinsics[0]

    @property
    def fy(self) -> float:
        return self.intrinsics[4]

    @property
    def cx(self) -> float:
        return self.intrinsics[2]

    @property
    def cy(self) -> float:
        return self.intrinsics[5]


@dataclass(frozen=True)
class Readout:
    """Sensor readout parameters of one observation (camera_config_toml)."""

    cfa: str  # "mono" | "rggb" (sensor.cfa)
    adc_bits: int
    black_level_dn: float
    electrons_per_dn: float
    exposure_s: float
    white_balance: tuple[float, float, float]
    device_to_linear_srgb: tuple[float, ...]  # 9, row-major

    def __post_init__(self) -> None:
        if len(self.white_balance) != 3:
            raise ValueError(f"white_balance must have 3 elements, got {len(self.white_balance)}")
        if len(self.device_to_linear_srgb) != 9:
            raise ValueError(
                f"device_to_linear_srgb must have 9 elements, got {len(self.device_to_linear_srgb)}"
            )


@dataclass(frozen=True)
class SignalSemantics:
    """What the values in a product image mean."""

    kind: str
    unit: str
    transfer: str
    cfa: str  # "0" mono, "1" RGGB
    channels: tuple[str, ...]
    colour_space: str
    storage: str

    @property
    def is_linear(self) -> bool:
        return self.transfer == "linear"


@dataclass(frozen=True)
class Product:
    """One dataset product: a manifest name plus a dataset-relative path."""

    name: str
    path: str  # relative to dataset root, POSIX separators
    signal: SignalSemantics | None = None  # None for truth/reference products
    channels: tuple[str, ...] = ()  # channel names declared by the producer


@dataclass(frozen=True)
class Observation:
    """One band's view of a sample: geometry, products, provenance record."""

    band: Band
    geometry: CameraGeometry
    readout: Readout
    spectral_mode: str
    products: Mapping[str, Product]
    record: str

    def product(self, name: str) -> Product:
        try:
            return self.products[name]
        except KeyError:
            raise KeyError(f"{name!r} not in {self.band} products; available: {sorted(self.products)}") from None


@dataclass(frozen=True)
class SampleParameters:
    """Sampling cell parameters of a sample (orbit, scale, repeat)."""

    azimuth_deg: float
    elevation_deg: float
    radius_scale: float
    repeat: int


@dataclass(frozen=True)
class Sample:
    """One dataset sample: scene + rig pose + the per-band observations."""

    dataset_id: str
    sample_id: str
    scene_id: str
    split_group: str
    parameters: SampleParameters
    rig_to_world: tuple[float, ...]  # 16, row-major
    observations: Mapping[Band, Observation]

    @property
    def condition(self) -> Condition:
        return condition_of(self.scene_id)

    @property
    def bands(self) -> tuple[Band, ...]:
        return tuple(b for b in BANDS if b in self.observations)

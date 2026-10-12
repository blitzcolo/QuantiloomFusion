"""Interface tests for Capabilities, FusionInput/Output, FusionMethod, registry."""

import numpy as np
import pytest

import quantiloom_fusion.fusion.registry as registry_mod
from quantiloom_fusion.core.bands import BANDS, Band
from quantiloom_fusion.dataset import CameraGeometry
from quantiloom_fusion.fusion import (
    Capabilities,
    FusionInput,
    FusionMethod,
    FusionOutput,
    available,
    create,
    register,
)
from quantiloom_fusion.geometry import Warp

ALL_BANDS = Capabilities(
    bands=frozenset(BANDS),
    needs_registration=False,
    heterogeneous_resolution=True,
    color_output=False,
    trainable=False,
)

IDENTITY16 = (1.0,) + (0.0,) * 15


@pytest.fixture
def registry():
    """Snapshot and restore the registry so tests do not leak methods."""
    snapshot = dict(registry_mod._REGISTRY)
    yield registry_mod
    registry_mod._REGISTRY.clear()
    registry_mod._REGISTRY.update(snapshot)


def make_geometry() -> CameraGeometry:
    return CameraGeometry(
        width=4,
        height=4,
        intrinsics=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        distortion=(0.0,) * 5,
        camera_to_world=IDENTITY16,
        world_to_camera=IDENTITY16,
    )


def make_warp(ref=Band.VIS, src=Band.LWIR) -> Warp:
    return Warp(ref, src, np.zeros((4, 4, 2)), np.ones((4, 4), bool))


def make_input(warps=None) -> FusionInput:
    images = {b: np.zeros((4, 4), dtype=np.float32) for b in BANDS}
    geometry = {b: make_geometry() for b in BANDS}
    return FusionInput(reference=Band.VIS, images=images, geometry=geometry, warps=warps)


class Passthrough(FusionMethod):
    name = "passthrough"
    capabilities = ALL_BANDS

    def fuse(self, x: FusionInput) -> FusionOutput:
        return FusionOutput(image=x.images[x.reference])


def test_fusion_input_requires_reference():
    x = make_input()
    del x.images[Band.VIS]
    with pytest.raises(ValueError):
        FusionInput(reference=Band.VIS, images=x.images, geometry=x.geometry)


def test_fusion_input_images_subset_of_geometry():
    x = make_input()
    del x.geometry[Band.LWIR]
    with pytest.raises(ValueError):
        FusionInput(reference=Band.VIS, images=x.images, geometry=x.geometry)


def test_fusion_method_is_abstract():
    with pytest.raises(TypeError):
        FusionMethod()


def test_check_input_rejects_unsupported_band():
    method = Passthrough()
    method.capabilities = Capabilities(
        bands=frozenset({Band.VIS, Band.LWIR}),
        needs_registration=False,
        heterogeneous_resolution=True,
        color_output=False,
        trainable=False,
    )
    with pytest.raises(ValueError, match="mwir"):
        method.check_input(make_input())


def test_check_input_requires_warps():
    class NeedsReg(Passthrough):
        capabilities = Capabilities(
            bands=frozenset(BANDS),
            needs_registration=True,
            heterogeneous_resolution=True,
            color_output=False,
            trainable=False,
        )

    method = NeedsReg()
    with pytest.raises(ValueError):
        method.check_input(make_input(warps=None))
    warps = {Band.LWIR: make_warp()}  # MWIR warp missing
    with pytest.raises(ValueError):
        method.check_input(make_input(warps=warps))
    warps[Band.MWIR] = make_warp(Band.VIS, Band.MWIR)
    method.check_input(make_input(warps=warps))  # no raise


def test_register_create_available(registry):
    @register("passthrough")
    class M(Passthrough):
        def __init__(self, scale=1.0):
            self.scale = scale

    assert "passthrough" in available()
    method = create("passthrough", scale=2.0)
    assert isinstance(method, Passthrough)
    assert method.name == "passthrough"
    assert method.scale == 2.0


def test_register_rejects_duplicates_and_non_methods(registry):
    @register("dup")
    class M(Passthrough):
        pass

    with pytest.raises(ValueError, match="duplicate"):
        @register("dup")
        class M2(Passthrough):
            pass

    with pytest.raises(ValueError, match="FusionMethod"):
        @register("notamethod")
        class Nope:
            pass


def test_create_unknown_raises_keyerror(registry):
    with pytest.raises(KeyError, match="known"):
        create("nope")


def test_fuse_roundtrip():
    x = make_input()
    out = Passthrough().fuse(x)
    assert isinstance(out, FusionOutput)
    assert out.image is x.images[Band.VIS]
    assert out.extras == {}

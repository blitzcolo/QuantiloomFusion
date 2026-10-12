"""Interface tests for the Registrar ABC and registration_error."""

import numpy as np
import pytest

from quantiloom_fusion.core.bands import Band
from quantiloom_fusion.geometry import Warp
from quantiloom_fusion.registration import Registrar, registration_error


class IdentityRegistrar(Registrar):
    name = "identity"

    def estimate(self, ref, src, ref_image, src_image):
        return None


def make_warp() -> Warp:
    return Warp(
        ref_band=Band.VIS,
        src_band=Band.LWIR,
        map_xy=np.zeros((2, 2, 2), dtype=np.float32),
        valid=np.ones((2, 2), dtype=bool),
    )


def test_registrar_is_abstract():
    with pytest.raises(TypeError):
        Registrar()


def test_subclass_instantiates():
    assert IdentityRegistrar().name == "identity"


def test_registration_error_stub():
    with pytest.raises(NotImplementedError):
        registration_error(make_warp(), make_warp())

"""Interface tests for the Registrar ABC."""

import pytest

from quantiloom_fusion.registration import Registrar


class IdentityRegistrar(Registrar):
    name = "identity"

    def estimate(self, ref, src, ref_image, src_image):
        return None


def test_registrar_is_abstract():
    with pytest.raises(TypeError):
        Registrar()


def test_subclass_instantiates():
    assert IdentityRegistrar().name == "identity"

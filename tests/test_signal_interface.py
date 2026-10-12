"""Interface tests for the Normalizer protocol."""

from quantiloom_fusion.signal import Normalizer


class DummyNormalizer:
    name = "dummy"

    def params(self) -> dict:
        return {"scale": 1.0}

    def __call__(self, image, semantics):
        return image


def test_dummy_satisfies_protocol():
    assert isinstance(DummyNormalizer(), Normalizer)


def test_incomplete_object_is_not_normalizer():
    assert not isinstance(object(), Normalizer)

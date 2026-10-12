"""Name-based registry of FusionMethod implementations."""

from __future__ import annotations

from .base import FusionMethod

_REGISTRY: dict[str, type[FusionMethod]] = {}


def register(name: str):
    """Class decorator: register a FusionMethod under name."""

    def decorate(cls: type[FusionMethod]) -> type[FusionMethod]:
        if not (isinstance(cls, type) and issubclass(cls, FusionMethod)):
            raise ValueError(f"{cls!r} is not a FusionMethod subclass")
        if name in _REGISTRY:
            raise ValueError(f"duplicate fusion method name {name!r}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorate


def create(name: str, **config) -> FusionMethod:
    """Instantiate a registered method; config goes to its constructor."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown fusion method {name!r}; known: {sorted(_REGISTRY)}") from None
    return cls(**config)


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))

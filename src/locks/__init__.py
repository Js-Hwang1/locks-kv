from __future__ import annotations
from .config import LocksConfig

__all__ = ["LocksConfig", "register"]
__version__ = "0.1.5.dev20261108"


def register(overrides=None) -> None:
    from .backend.register import register as _register

    _register(overrides)

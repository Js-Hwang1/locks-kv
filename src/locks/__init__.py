"""LOCKS — decode-time certified page selection for KV-cache attention, with an
optional DRAM offload tier. Two build targets share one algorithm core:

  * LOCKS-fast  — K+V resident (the speed play).
  * LOCKS-mem   — V (or K+V) offloaded to DRAM (default; the memory play).

Public API:
  from locks import LocksConfig, register        # register() = vLLM plugin hook

Architecture (ours_doc/LOCKS_DESIGN.md):
  config.py     LocksConfig — single source of truth (no scattered env vars)
  selection/    Stage A: decode-time page selection (SHARED by both variants)
  attention/    Stage B: sparse paged attention (V source injected as a seam)
  tier/         DRAM offload state machine (mem variants only)
  backend/      vLLM integration: attention impl, metadata builder, register()
"""
from __future__ import annotations

from .config import LocksConfig  # noqa: F401

__all__ = ["LocksConfig", "register"]

__version__ = "0.1.1"


def register() -> None:
    """vLLM general-plugin entry point. Thin re-export of
    :func:`locks.backend.register.register` so the entry point is stable while
    the backend is ported."""
    from .backend.register import register as _register
    _register()

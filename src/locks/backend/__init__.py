"""vLLM integration — the plugin boundary.

Registers LOCKS as a vLLM general-plugin (entry point
``vllm.general_plugins`` -> ``locks.backend.register:register``): patches
``CudaPlatform.get_attn_backend_cls`` so the FlashAttention backend is replaced
by :class:`LocksAttentionBackend`, and (mem variants) patches the KV-cache spec
to K-only. All behaviour is driven by a single :class:`locks.LocksConfig`.

Modules:
  attn.py       LocksAttentionImpl — dispatch fast|mem, decode/prefill forwards
  builder.py    LocksMetadataBuilder — persistent buffers + per-step refresh
  register.py   register() + get_kv_cache_spec patch

Decode is a subclass of FlashAttentionImpl: every non-pure-decode path (prefill,
mixed, cascade, profiling) inherits stock FA; only pure single-token decode runs
the LOCKS Stage-A + Stage-B pipeline (+ tier for mem).
"""
from __future__ import annotations

from .attn import LocksAttentionImpl  # noqa: F401
from .builder import LocksMetadataBuilder  # noqa: F401
from .register import LocksAttentionBackend, register  # noqa: F401

__all__ = [
    "register",
    "LocksAttentionBackend",
    "LocksAttentionImpl",
    "LocksMetadataBuilder",
]

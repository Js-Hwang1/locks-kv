from __future__ import annotations
import os
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from ..config import LocksConfig
from . import _runtime
from .attn import LocksAttentionImpl
from .builder import LocksMetadataBuilder

_APPLIED = False


class LocksAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[LocksAttentionImpl]:
        return LocksAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[LocksMetadataBuilder]:
        return LocksMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        cfg = _runtime.get_config()
        if (
            cfg is not None
            and cfg.is_mem
            and cfg.mem_k_only
            and _runtime.mem_stageb_wired()
        ):
            if block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")
            if getattr(cfg, "mem_summary_cache", False):
                return (num_blocks, 1, block_size, num_kv_heads, head_size)
            return (num_blocks, 1, block_size, num_kv_heads, head_size)
        return FlashAttentionBackend.get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )


def _summary_head_size(cfg, block_size: int, head_size: int = 128) -> int:
    if cfg.is_mem and cfg.mem_k_only and cfg.offload_k and cfg.mem_summary_cache:
        if not (
            cfg.is_mem and cfg.mem_k_only and cfg.offload_k and cfg.mem_summary_cache
        ):
            raise ValueError(
                "Minimal engine handles require the complete host-backed K/V path"
            )
        return 1
    elem = 2
    from ..selection.rki4_state import record_bytes_padded

    rec_bytes = record_bytes_padded(d=head_size, page=block_size, elem=elem)
    if rec_bytes % (block_size * elem) != 0:
        raise ValueError(
            f"summary cache: rec_bytes {rec_bytes} not a multiple of block*elem {block_size * elem} -- cannot tile the engine page"
        )
    return rec_bytes // (block_size * elem)


_OUR_PATH = f"{__name__}.LocksAttentionBackend"
_MLA_PATH = __name__.rsplit(".", 1)[0] + ".mla_attn.LocksMLABackend"
_ANNOUNCED = False


def register(overrides: dict | None = None) -> None:
    global _APPLIED
    if _APPLIED or os.environ.get("LOCKS_DISABLE"):
        return
    if overrides is None and (not os.environ.get("LOCKS_CONFIG")):
        print(
            "[locks] register: no LOCKS_CONFIG and no overrides -> plugin INERT (stock vLLM). Set LOCKS_CONFIG to activate.",
            flush=True,
        )
        return
    cfg = LocksConfig.load(overrides)
    _runtime.set_config(cfg)
    from vllm.platforms import cuda

    target = next(
        (k for k in cuda.CudaPlatform.__mro__ if "get_attn_backend_cls" in k.__dict__)
    )
    orig = target.__dict__["get_attn_backend_cls"].__func__

    def get_attn_backend_cls(cls, *args, **kwargs):
        path = orig(cls, *args, **kwargs)
        if path == "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend":
            global _ANNOUNCED
            if not _ANNOUNCED:
                _ANNOUNCED = True
                print(
                    f"[locks] ACTIVE variant={cfg.variant} budget={cfg.budget} coverage={cfg.coverage} sink={cfg.sink_pages} window={cfg.window_pages} -> {_OUR_PATH}",
                    flush=True,
                )
            return _OUR_PATH
        if isinstance(path, str) and ".mla." in path:
            if cfg.is_mem:
                raise ValueError(
                    "MLA currently supports resident KV; host offloading is supported for GQA"
                )
            _disable_dsa_indexer()
            return _MLA_PATH
        return path

    target.get_attn_backend_cls = classmethod(get_attn_backend_cls)
    _APPLIED = True
    print("[locks] registered (patched CudaPlatform.get_attn_backend_cls)", flush=True)
    if os.environ.get("LOCKS_K5_STATS", "0") != "1":
        _wire_k5_early()
    if cfg.is_mem and cfg.mem_k_only and _runtime.mem_stageb_wired():
        _patch_kv_cache_spec_k_only()
    if cfg.is_mem:
        from .memplan import patch_available_memory

        patch_available_memory()
        _patch_profiling_window()


def _disable_dsa_indexer():
    from vllm.model_executor.models import deepseek_v2

    indexer = getattr(deepseek_v2, "Indexer", None)
    if indexer is None or getattr(indexer.forward, "_locks_replaced", False):
        return

    def forward(self, *args, **kwargs):
        return None

    forward._locks_replaced = True
    indexer.forward = forward


def _patch_profiling_window() -> None:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _MR
    except Exception as exc:
        print(f"[locks] mem: profiling-window patch SKIPPED ({exc})", flush=True)
        return
    orig = getattr(_MR, "profile_cudagraph_memory", None)
    if orig is None or getattr(orig, "_locks_wrapped", False):
        if orig is None:
            print(
                "[locks] mem: profiling-window patch SKIPPED (GPUModelRunner.profile_cudagraph_memory absent)",
                flush=True,
            )
        return

    def profile_cudagraph_memory(self, *args, **kwargs):
        _runtime.set_profiling(True)
        try:
            return orig(self, *args, **kwargs)
        finally:
            _runtime.set_profiling(False)

    profile_cudagraph_memory._locks_wrapped = True
    _MR.profile_cudagraph_memory = profile_cudagraph_memory
    print(
        "[locks] mem: patched GPUModelRunner.profile_cudagraph_memory (tier-less dummy-run window)",
        flush=True,
    )
    orig_dummy = getattr(_MR, "_dummy_run", None)
    if orig_dummy is not None and (not getattr(orig_dummy, "_locks_wrapped", False)):

        def _dummy_run(self, *args, **kwargs):
            if not kwargs.get("is_profile"):
                return orig_dummy(self, *args, **kwargs)
            _runtime.set_profiling(True)
            try:
                return orig_dummy(self, *args, **kwargs)
            finally:
                _runtime.set_profiling(False)

        _dummy_run._locks_wrapped = True
        _MR._dummy_run = _dummy_run
        print(
            "[locks] mem: patched GPUModelRunner._dummy_run (is_profile => tier-less window)",
            flush=True,
        )
    elif orig_dummy is None:
        print("[locks] mem: _dummy_run patch SKIPPED (method absent)", flush=True)


def _wire_k5_early() -> None:
    if _runtime.get_config().mem_transport == "pcie":
        print("[locks] PCIe transport ACTIVE: selected misses filled before attention", flush=True)
        return
    from . import attnwire

    def _cb(model, vllm_config):
        orig_cl = model.compute_logits

        def compute_logits(*a, **kw):
            tier = _runtime.get_tier()
            if tier is not None and (not _runtime.in_profiling()):
                tier.early_service()
            return orig_cl(*a, **kw)

        model.compute_logits = compute_logits
        print(
            "[locks] K5_EARLY ACTIVE: service issued at the post-graph host point (compute_logits pre-hook)",
            flush=True,
        )

    attnwire.on_model_loaded(_cb)


def _patch_kv_cache_spec_k_only() -> None:
    import dataclasses
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    try:
        from vllm.attention.layer import Attention
    except Exception:
        from vllm.model_executor.layers.attention.attention import Attention
    orig = Attention.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = orig(self, vllm_config)
        if spec is None:
            return spec
        if type(spec) is not FullAttentionSpec:
            raise RuntimeError(
                f"locks mem_k_only requires plain FullAttentionSpec layers; got {type(spec).__name__} for {getattr(self, 'layer_name', '?')!r}"
            )
        if spec.head_size_v != spec.head_size:
            raise RuntimeError(
                f"locks mem_k_only: layer has asymmetric head_size_v ({spec.head_size_v} != {spec.head_size}) -- unsupported"
            )
        cfg = _runtime.get_config()
        if cfg is not None and getattr(cfg, "mem_summary_cache", False):
            hs = _summary_head_size(cfg, spec.block_size, spec.head_size)
            if cfg.is_mem and cfg.mem_k_only and cfg.offload_k:
                return dataclasses.replace(
                    spec, head_size=hs, head_size_v=0, num_kv_heads=1
                )
            return dataclasses.replace(spec, head_size=hs, head_size_v=0)
        return dataclasses.replace(spec, head_size_v=0)

    Attention.get_kv_cache_spec = get_kv_cache_spec
    print(
        "[locks] mem_k_only: patched Attention.get_kv_cache_spec (head_size_v=0; summary_cache -> head_size=records)",
        flush=True,
    )

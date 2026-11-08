from __future__ import annotations
from typing import Optional
from ..config import LocksConfig

_config: Optional[LocksConfig] = None
_tier = None
_in_profiling = False
_PPSEL_ANNOUNCED = False


def mem_stageb_wired():
    return True


def set_config(cfg: LocksConfig) -> None:
    global _config
    _config = cfg


def get_config() -> Optional[LocksConfig]:
    return _config


def set_tier(tier) -> None:
    global _tier
    _tier = tier


def get_tier():
    return _tier


def set_profiling(on: bool) -> None:
    global _in_profiling
    _in_profiling = bool(on)


def in_profiling() -> bool:
    return _in_profiling


def tier_layer_index(tier, kv_cache) -> int:
    layer_of = getattr(tier, "_layer_of", None)
    if layer_of is None:
        raise RuntimeError("locks mem: tier has no _layer_of map (builder wiring)")
    lidx = layer_of.get(kv_cache.data_ptr())
    if lidx is None:
        raise RuntimeError(
            f"locks mem: kv_cache pointer not in the tier layer map (ptr={kv_cache.data_ptr():#x}); check static_forward_context wiring"
        )
    return lidx


def _layer_index(st, kv_cache) -> int:
    layer_of = getattr(st, "_layer_of", None)
    if layer_of is None:
        raise RuntimeError(
            "locks r8: R8State has no _layer_of map (builder did not wire it)"
        )
    lidx = layer_of.get(kv_cache.data_ptr())
    if lidx is None:
        raise RuntimeError(
            f"locks r8: kv_cache pointer not in the builder's layer map (ptr={kv_cache.data_ptr():#x}); the r8 state is L-major and needs the physical layer index. Check static_forward_context wiring."
        )
    return lidx


def _rki4_select_adapter(
    q, kv_cache, block_table, seq_lens, st, n_req, scale, cfg
) -> None:
    from ..selection import rki4_score_cuda, select_cuda
    from ..selection.shared_average import validate_policy_environment

    validate_policy_environment()
    lidx = _layer_index(st, kv_cache)
    if q.ndim == 2:
        q = q.view(q.shape[0], st.n_kv * st.G, st.d)
    rki4_score_cuda.rki4_score_state(
        st, lidx, q, block_table, seq_lens, n_req, scale, kv_cache=kv_cache
    )
    global _PPSEL_ANNOUNCED
    if not _PPSEL_ANNOUNCED:
        _PPSEL_ANNOUNCED = True
        print(
            "[locks] all-visible-page shared-average GQA ACTIVE: one page set per KV head, mandatory pages included in normalization and budget",
            flush=True,
        )
    select_cuda.topb_select_cuda(st, n_req)


def decode_dispatch(q, kv_cache, block_table, seq_lens, st, out, vsource, scale, cfg):
    from ..attention.decode import sparse_paged_decode_batched

    sparse_paged_decode_batched(
        q, kv_cache, block_table, seq_lens, st, out, vsource, scale=scale
    )


def resolve_selection():
    from ..selection import Rki4State, derive_page_params

    return (Rki4State, derive_page_params, _rki4_select_adapter, True)

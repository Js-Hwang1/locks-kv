"""The CO / FIX-C lane: the QKV projection + rope as ONE op, nt >= 1.

``torch.ops.locks_co.qkv`` replaces [fused QKV GEMM + rotary] wholesale for
the wired attention forward.  On a pure-decode step of ANY row count it runs
the q projection, then the DEPLOYED-TU score (whose staging prelude ropes q
and publishes ``st.q_rope``), and DEFERS the kv projection + rope + KV write
to ``_runtime.fixc_flush``, which launches them behind the select so they ride
its cover window.  Everything else -- prefill, mixed steps, metadata-less
capture passes -- takes the stock fused ``F.linear`` plus vLLM's in-place
rotary inside the op body.

UNION KERNEL RETIRED (2026-07-22, BATCHED_CO_DESIGN 2.7).  This file used to
also carry a fat ``coqkv_kernel`` that ran [kv-GEMV || rki4 score] in one
launch by compiling ``rki4_score_cuda._SRC`` with ``-DRKI4_DEVICE_BODY``.  It
is gone, for two independent reasons:

  * it was NOT LAUNCHED in the deployed set -- with ``LOCKS_QFIRST_FIXC=1``
    the score routes through the deployed TU and the union kernel never ran
    (MODEL_AGNOSTIC_QFIRST 7.3), which is also why FIX-C's own banner calls
    itself "union-kernel retirement";
  * generalizing it to nt > 1 would lose on registers, and the repo refuted
    that exact mechanism PRE-BUILD: "score occupancy cliff: fused union
    reg-bound at 4 CTA/SM vs score's 8" (REFUTED_ARMS_INDEX.md:128, arm M2).

Deleting it is itself the coherence win the unification is for: one lane, one
kernel per stage, ``nt = 1`` special-cased nowhere.  ``LOCKS_QFIRST_CO=1``
without ``LOCKS_QFIRST_FIXC=1`` is now a loud boot failure (register.py) --
it used to be the only configuration that launched the union kernel, and a
silent no-op is exactly what this work removes.
"""
from __future__ import annotations

from typing import Optional

import torch

_OP_READY = False

# Bound ONCE (not imported per call): the op reads attn_metadata every layer to
# get max_query_len, and at bs=1 the decode step is launch-bound, so a per-call
# import on the critical path shows up in the bs=1 wall.
try:
    from vllm.forward_context import get_forward_context as _get_fwd_ctx
except Exception:  # noqa: BLE001
    _get_fwd_ctx = None


def ensure_co_op() -> None:
    """Register torch.ops.locks_co.qkv (opaque, PURE, single output).

    UNIFIED nt >= 1 (2026-07-22, ours_doc/BATCHED_CO_DESIGN).  The predicate
    contains NO SHAPE.  It is::

        attn_metadata is live  AND  max_query_len == 1  AND  the select
        adapter's refs stash is fresh for THIS call

    i.e. "is this a pure-decode step" -- a statement about the WORK (rows that
    have a selection kernel vs rows that do not), the same class of dispatch
    vLLM itself makes between decode and prefill attention.  It is NOT a
    quality split: there is one implementation of the projection per step
    type, not two racing on speed.

      pure decode, any nt : q-GEMV over nt rows -> the DEPLOYED-TU score over
                nt rows (its staging prelude ropes q and publishes st.q_rope);
                the kv projection + rope + KV write are DEFERRED to
                _runtime.fixc_flush, which launches them behind the select so
                they ride its cover window.  Returns UNROPED q/k.
      otherwise (prefill / mixed / no metadata) : stock fused F.linear +
                IN-PLACE vllm rotary on the q/k slices INSIDE the op body
                (eager context, invisible to functionalization -> no clones;
                bitwise == the inductor rope, gate_norope_p3pre) -> returns
                ROPED qkv, prefill bytes stock.

    The op's rope decision is a function of ``max_query_len`` ONLY (absorb on
    pure decode, rope on mixed), so the attention impl can reach the SAME
    decision from its own metadata without any per-step op state -- which is
    what survives CUDA-graph replay.  See _runtime's block above _qf2_refs for
    why a receipt (op stamps, impl reads) does NOT survive replay."""
    global _OP_READY
    if _OP_READY:
        return
    import torch
    from torch.library import Library
    from vllm.utils.torch_utils import direct_register_custom_op

    lib = Library("locks_co", "FRAGMENT")

    def qkv(hidden: torch.Tensor, weight: torch.Tensor,
            bias: Optional[torch.Tensor], positions: torch.Tensor,
            q_size: int, lidx: int) -> torch.Tensor:
        from . import _runtime, gemv_cuda
        qs = int(q_size)
        nt = hidden.shape[0]
        # BIAS-FREE ARCHS (model-agnostic, 2026-07-22): Llama/Qwen have
        # attention_bias=False, so bias is None.  The hand GEMV kernels fuse
        # the bias add unconditionally (no branch, per the no-fallback rule),
        # so the DECODE halves get a real all-zeros row -- bitwise (`v + 0.0f
        # == v`).  The PREFILL fallthrough below still uses the true `bias`
        # (None), so prefill bytes remain stock.
        gb = bias if bias is not None else _runtime.zero_bias(weight)
        # PURE-DECODE test from the CURRENT metadata (NOT the stash, NOT a
        # shape).  The op's rope decision must be a function of metadata only,
        # so that under CUDA-graph replay -- where this python does not re-run
        # but the impl's does -- the impl can reach the SAME decision from the
        # same metadata.  ``max_query_len == 1`` IS the op's absorbed condition.
        md = _get_fwd_ctx().attn_metadata if _get_fwd_ctx else None
        if isinstance(md, dict):
            md = next(iter(md.values()), None)
        pure_decode = md is not None and getattr(md, "max_query_len", None) == 1
        if pure_decode and _runtime._FIXC:
            # Refs from the CURRENT metadata, never the stash: the stash is a
            # previous call's tuple that the batch-split drops, and reading it
            # created the batch-composition-stale hazard.  st carries the scale
            # (a scalar the adapter persisted).  On a pure-decode step every
            # layer's md.locks is the same, current selection state.
            st = getattr(md, "locks", None)
            bt = getattr(md, "block_table", None)
            sl = getattr(md, "seq_lens", None)
            scale = getattr(st, "_scale", None) if st is not None else None
            if (st is not None and bt is not None and sl is not None
                    and scale is not None and sl.shape[0] == nt):
                gv = gemv_cuda.get_mod()
                assert gv is not None and hasattr(gv, "kv_gemv_nf"), (
                    "LOCKS_QFIRST_FIXC: gemv extension lacks kv_gemv_nf"
                    " (built without -DLOCKS_FIXC?). Prebuild offline "
                    "with LOCKS_QFIRST_FIXC=1 -- no silent fallback.")
                from ..selection import rki4_score_cuda as rsc
                out = torch.empty(nt, weight.shape[0],
                                  dtype=hidden.dtype,
                                  device=hidden.device)
                gv.q_gemv(hidden, weight, gb, out[:, :qs], qs)
                q3 = out[:, :qs].view(nt, st.n_kv * st.G, st.d)
                # ABSORBED: q3 is UNROPED, the scorer's staging prelude ropes
                # it and publishes st.q_rope.  st._q_preroped=False tells the
                # score kernel to rope; set it here (the op runs first) because
                # a previous mixed step may have left it True.
                st._q_preroped = False
                rsc.rki4_score_only(st, int(lidx), q3, bt, sl, nt,
                                    float(scale))
                _runtime.fixc_arm_proj(int(lidx), hidden, weight, gb,
                                       out[:, qs:], qs, positions)
                _runtime._co_mark[int(lidx)] = _runtime._qf2_step
                return out
        # NON-pure-decode (mixed / prefill / no metadata), OR pure-decode with
        # the selection state not yet built (boot): STOCK projection + rope,
        # returns ROPED q/k.  The impl's `absorbed` is False here (its
        # max_query_len test agrees), so it reads this roped q directly.  A
        # pure-decode step ALWAYS has md.locks once the state is built, so the
        # only pure-decode rows that reach here are boot/profile passes whose
        # output is discarded.
        assert _runtime._NOROPE_CSC is not None, (
            "locks_co.qkv: the rotary cos_sin_cache was never published "
            "(attnwire.publish_rope did not run). No fallback.")
        full = torch.nn.functional.linear(hidden, weight, bias)
        from vllm import _custom_ops as ops
        q = full[:, :qs]
        k = full[:, qs:qs + (weight.shape[0] - qs) // 2]
        ops.rotary_embedding(positions, q, k, _runtime._NOROPE_HEAD,
                             _runtime._NOROPE_CSC, _runtime._NOROPE_NEOX)
        return full

    def qkv_fake(hidden: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor], positions: torch.Tensor,
                 q_size: int, lidx: int) -> torch.Tensor:
        return hidden.new_empty(hidden.shape[0], weight.shape[0])

    _ann = {"hidden": torch.Tensor, "weight": torch.Tensor,
            "bias": Optional[torch.Tensor], "positions": torch.Tensor,
            "q_size": int, "lidx": int, "return": torch.Tensor}
    qkv.__annotations__ = dict(_ann)
    qkv_fake.__annotations__ = dict(_ann)
    direct_register_custom_op(op_name="qkv", op_func=qkv, mutates_args=[],
                              fake_impl=qkv_fake, target_lib=lib)
    ensure_co_op._lib = lib
    _OP_READY = True

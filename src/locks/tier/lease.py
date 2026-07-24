"""PrefillLease -- leased-KV chunked prefill (LOCKS_KERNEL_REDESIGN.md v2 3.4).

THE PROBLEM (kills the old ``_prefill_konly``): with a K-only engine cache
(mem_k_only) chunk k's prefill attention must read the ENTIRE prefix K/V, but
the V of chunks 1..k-1 has already left the engine (staged -> flushed to the
pinned host pool).  Re-reading it from host DRAM is ~ctx^2/(2 chunk) bytes over
PCIe (67 GB at 128K): dead.  The old code therefore FORCED single-shot prefill
(max_num_batched_tokens >= ctx), which (a) makes mem unbenchmarkable under the
canonical engine config, (b) blows the staging pools up to O(ctx), and (c)
carries a ``.item()`` that invalidates CUDA-graph capture when the dense path
is reached under capture (both measured: nsys_H200_AMM_ctx16384_p16_b1.log).

THE FIX: the prefix K/V of the request CURRENTLY PREFILLING is held in a
transient, VRAM-resident LEASE for the duration of its prefill:

  K_lease / V_lease : (L, T, n_kv, d) bf16, T = max_model_len tokens,
                      token-major -- EXACTLY the layout stock FlashAttention
                      varlen consumes, so chunk attention is
                      flash_attn_varlen_func(q_chunk, K_lease[l][:sl],
                      V_lease[l][:sl], causal=True) with the standard
                      bottom-right rectangular mask.  No custom prefill kernel.

The lease is a pure READ-SIDE addition: the tier's staging/flush state machine
(begin_step, valloc, scatter, flush_collect/copy, I1-I4) is UNTOUCHED and runs
for prefill chunks exactly as it always did.  The lease is filled per layer,
per step, from the tier's own pools:

  pages [wm, sl) of the leased request:
    staged  (vbo >= 0)      -> VRAM->VRAM copy from v_pool/k_pool  (the normal
                               path: this step's chunk was just scattered)
    flushed (pool_valid)    -> UVA read from the pinned host pool  (RECOVERY
                               only: a lease picked up mid-prefill, e.g. the
                               first chunk was not leased or a preemption
                               resume; one-time, chunk-sized)

Bit-exact by construction (int16 bit copies end to end, same bytes the decode
tier serves).  At most ONE lease exists (engine asserts
max_num_partial_prefills == 1); it is freed the moment its request leaves
prefill.  Sizing: one lease of max_model_len tokens, K+V = the request's own
FullKV -- transient, O(one request), reported honestly in the VRAM plan
(backend/memplan.py reserves it; section 5 peak row).

I1 note: during prefill a complete written page is staged OR flushed exactly
as before (the lease holds a byte-identical COPY, not the primary); the
invariant contract is unchanged.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lease_fill_kernel(bt_ptr, lease_v_ptr, lease_k_ptr,
                       vp_ptr, kp_ptr, vbo_ptr,
                       poolv_ptr, poolk_ptr, valid_ptr,
                       err_ptr,
                       p0, lidx, NB,
                       svb, svt, svh,
                       lt, lh,
                       PAGE: tl.constexpr, D: tl.constexpr,
                       NKV: tl.constexpr, HAS_K: tl.constexpr):
    """Fill lease pages [p0, p0+grid) of ONE layer from staged/pinned.

    Grid (n_pages,). For page p (logical page index of the request), physical
    block blk = block_table_row[p]; per kv-head the source is the staged pool
    (token-major, same order as the lease) if ``vbo[blk] >= 0`` else the pinned
    pool (kv-major page, transposed on the fly). Pure int16 bit copies. If a
    kv-head of a page is neither staged nor flushed (an upstream I1 violation)
    the error latch is set and the page is left unwritten (loud, never wrong).
    """
    p = p0 + tl.program_id(0)
    blk = tl.load(bt_ptr + p).to(tl.int64)
    vb = tl.load(vbo_ptr + blk)
    offs_t = tl.arange(0, PAGE)
    offs_d = tl.arange(0, D)
    for kv in range(NKV):
        # lease destination: [l, p*PAGE + t, kv, d] (token-major)
        dst = (p * PAGE + offs_t[:, None]).to(tl.int64) * lt + kv * lh \
            + offs_d[None, :]
        if vb >= 0:
            src = vb.to(tl.int64) * svb + offs_t[:, None] * svt + kv * svh \
                + offs_d[None, :]
            v16 = tl.load(vp_ptr + src).to(tl.int16, bitcast=True)
            tl.store(lease_v_ptr + dst, v16)
            if HAS_K:
                k16 = tl.load(kp_ptr + src).to(tl.int16, bitcast=True)
                tl.store(lease_k_ptr + dst, k16)
        else:
            vld = tl.load(valid_ptr + (lidx * NKV + kv) * NB + blk)
            if vld > 0:
                pool_off = (((lidx * NB + blk) * NKV + kv).to(tl.int64)
                            * (PAGE * D)) + offs_t[:, None] * D + offs_d[None, :]
                tl.store(lease_v_ptr + dst, tl.load(poolv_ptr + pool_off))
                if HAS_K:
                    tl.store(lease_k_ptr + dst, tl.load(poolk_ptr + pool_off))
            else:
                tl.store(err_ptr, 1)


class PrefillLease:
    """One transient leased-KV buffer (the single mid-prefill request)."""

    def __init__(self, tier, max_tokens: int):
        self._tier = tier
        t = tier
        self.T = int(max_tokens)
        dev = t.device
        # token-major (L, T, n_kv, d): FA-varlen-ready per layer.
        self.v = torch.empty((t.L, self.T, t.n_kv, t.d), dtype=t.dtype,
                             device=dev)
        self.k = torch.empty((t.L, self.T, t.n_kv, t.d), dtype=t.dtype,
                             device=dev)
        self._v16 = self.v.view(torch.int16)
        self._k16 = self.k.view(torch.int16)
        self.err = torch.zeros(1, dtype=torch.int32, device=dev)
        self.wm = 0                     # tokens materialized: [0, wm)
        self.t1 = 0                     # this step's fill target (== sl)
        self.fill_lo = 0                # this step's fill range [lo, t1)
        self.gib = 2 * self.v.numel() * self.v.element_size() / 2**30

    # ---- per-step host plan (builder, outside the graph) ----------------- #
    def plan_step(self, ctx: int, sl: int) -> None:
        """Called once per prefill step for the leased row: freeze this step's
        fill range. ``ctx`` = tokens computed before this step, ``sl`` = after.
        wm > ctx (stale lease from an aborted request) resets to a full refill."""
        if self.wm > ctx:
            self.wm = 0
        self.fill_lo = self.wm
        self.t1 = sl
        self.wm = sl

    # ---- per-layer fill (impl.forward, after this layer's KV scatter) ----- #
    def fill_layer(self, lidx: int, bt_row: torch.Tensor) -> None:
        """Materialize lease[lidx, fill_lo:t1) from the tier's pools."""
        lo, t1 = self.fill_lo, self.t1
        if t1 <= lo:
            return
        t = self._tier
        page = t.page
        pg0, pg1 = lo // page, (t1 - 1) // page
        n_pg = pg1 - pg0 + 1
        stg = t.staging
        vp16 = stg.v_pool[lidx].view(torch.int16)
        has_k = t.offload_k
        kp16 = stg.k_pool[lidx].view(torch.int16) if has_k else vp16
        poolk = t.pool_k.dev_view if has_k else t.pool.dev_view
        lv = self._v16[lidx]
        _lease_fill_kernel[(n_pg,)](
            bt_row, lv, self._k16[lidx],
            vp16, kp16, stg.vbo,
            t.pool.dev_view, poolk, t.residency.pool_valid,
            self.err,
            pg0, lidx, t.NB,
            vp16.stride(0), vp16.stride(1), vp16.stride(2),
            lv.stride(0), lv.stride(1),
            PAGE=page, D=t.d, NKV=t.n_kv, HAS_K=has_k, num_warps=4)

    def release(self) -> None:
        self.v = self.k = self._v16 = self._k16 = None

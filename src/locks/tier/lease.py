from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _lease_fill_kernel(
    bt_ptr,
    lease_v_ptr,
    lease_k_ptr,
    vp_ptr,
    kp_ptr,
    vbo_ptr,
    poolv_ptr,
    poolk_ptr,
    valid_ptr,
    err_ptr,
    p0,
    lidx,
    NB,
    svb,
    svt,
    svh,
    lt,
    lh,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    NKV: tl.constexpr,
    HAS_K: tl.constexpr,
):
    p = p0 + tl.program_id(0)
    blk = tl.load(bt_ptr + p).to(tl.int64)
    vb = tl.load(vbo_ptr + blk)
    offs_t = tl.arange(0, PAGE)
    offs_d = tl.arange(0, D)
    for kv in range(NKV):
        dst = (p * PAGE + offs_t[:, None]).to(tl.int64) * lt + kv * lh + offs_d[None, :]
        if vb >= 0:
            src = (
                vb.to(tl.int64) * svb
                + offs_t[:, None] * svt
                + kv * svh
                + offs_d[None, :]
            )
            v16 = tl.load(vp_ptr + src).to(tl.int16, bitcast=True)
            tl.store(lease_v_ptr + dst, v16)
            if HAS_K:
                k16 = tl.load(kp_ptr + src).to(tl.int16, bitcast=True)
                tl.store(lease_k_ptr + dst, k16)
        else:
            vld = tl.load(valid_ptr + (lidx * NKV + kv) * NB + blk)
            if vld > 0:
                pool_off = (
                    ((lidx * NB + blk) * NKV + kv).to(tl.int64) * (PAGE * D)
                    + offs_t[:, None] * D
                    + offs_d[None, :]
                )
                tl.store(lease_v_ptr + dst, tl.load(poolv_ptr + pool_off))
                if HAS_K:
                    tl.store(lease_k_ptr + dst, tl.load(poolk_ptr + pool_off))
            else:
                tl.store(err_ptr, 1)


class PrefillLease:
    def __init__(self, tier, max_tokens: int):
        self._tier = tier
        t = tier
        self.T = int(max_tokens)
        dev = t.device
        self.v = torch.empty((t.L, self.T, t.n_kv, t.d), dtype=t.dtype, device=dev)
        self.k = torch.empty((t.L, self.T, t.n_kv, t.d), dtype=t.dtype, device=dev)
        self._v16 = self.v.view(torch.int16)
        self._k16 = self.k.view(torch.int16)
        self.err = torch.zeros(1, dtype=torch.int32, device=dev)
        self.wm = 0
        self.t1 = 0
        self.fill_lo = 0
        self.gib = 2 * self.v.numel() * self.v.element_size() / 2**30

    def plan_step(self, ctx: int, sl: int) -> None:
        if self.wm > ctx:
            self.wm = 0
        self.fill_lo = self.wm
        self.t1 = sl
        self.wm = sl

    def fill_layer(self, lidx: int, bt_row: torch.Tensor) -> None:
        (lo, t1) = (self.fill_lo, self.t1)
        if t1 <= lo:
            return
        t = self._tier
        page = t.page
        (pg0, pg1) = (lo // page, (t1 - 1) // page)
        n_pg = pg1 - pg0 + 1
        stg = t.staging
        vp16 = stg.v_pool[lidx].view(torch.int16)
        has_k = t.offload_k
        kp16 = stg.k_pool[lidx].view(torch.int16) if has_k else vp16
        poolk = t.pool_k.dev_view if has_k else t.pool.dev_view
        lv = self._v16[lidx]
        _lease_fill_kernel[n_pg,](
            bt_row,
            lv,
            self._k16[lidx],
            vp16,
            kp16,
            stg.vbo,
            t.pool.dev_view,
            poolk,
            t.residency.pool_valid,
            self.err,
            pg0,
            lidx,
            t.NB,
            vp16.stride(0),
            vp16.stride(1),
            vp16.stride(2),
            lv.stride(0),
            lv.stride(1),
            PAGE=page,
            D=t.d,
            NKV=t.n_kv,
            HAS_K=has_k,
            num_warps=4,
        )

    def release(self) -> None:
        self.v = self.k = self._v16 = self._k16 = None

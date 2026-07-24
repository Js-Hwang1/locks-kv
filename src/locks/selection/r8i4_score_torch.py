"""r8i4 page-LSE score -- plain-torch CORE LOGIC for sm_120 (Blackwell).

The hand-CUDA r8i4 score kernel (``r8i4_score_cuda.r8i4_score_bt``) is
Hopper-only and its sm_120 dispatch lane was removed (commit 2404b45,
``_runtime.py:797-800``). This module is the arch-gated sm_120 realization of
the SAME core logic: it reads the identical packed summary slabs
(``pp_v4/pp_vs/pp_c8/pp_cs/pp_mu8/pp_mus``) and writes the identical per-head
page LSE into ``st.score_h`` (R, n_kv, G, MP). The downstream GQA combine
(Triton ``_nrm_launch`` via ``LOCKS_FUSE_NRMTOPB=0``) and top-b
(``LOCKS_TOPB_TRITON=1``) then run unchanged, so page selection is bit-for-bit
the shared reference path.

It is the byte-faithful torch twin of the fp32 reference the CUDA kernel is
unit-tested against (``LOCKS-test2/kernel/r8score_ab.py::torch_ref``, the G1
gate), INCLUDING the per-(kv-head, group) int8 round-trip of q the kernel does
(i8*i8 tensor-core path). Math (per kv-head h, group g, page p; scale s):

    q~[h,g,r] = sum_d Vref[h,p,d,r] * qref[h,g,d]              (Vref = Vq * vs)
    tok[h,g,t] = s * ( sum_r Cref[h,p,t,r] * q~[h,g,r] + muref[h,p]·qref[h,g] )
    score_h[h,g,p] = logsumexp_t tok[h,g,t]                    (Cref=c8*cs, muref=mu8*mus)

with ``qref = round(q/qs)*qs``, ``qs = |q|.amax(-1)/127`` per (h,g).
The int4 basis is unpacked column-major (lo nibble = even-d row, hi = odd-d),
4-bit two's-complement sign-extended ``((n^8)-8)`` -- exactly ``_write_post``'s
packing in reverse. Pure torch (einsum/logsumexp), no CUDA/Triton.
"""
from __future__ import annotations

import os as _os

import torch

_ANNOUNCED = False
_SCORE_MS = 0.0   # LOCKS_R8I4_TIME=1 profiling accumulators (GPU-event ms)
_SCORE_N = 0


@torch.no_grad()
def r8i4_score_torch(st, layer: int, q, block_table, seq_lens, n_req: int,
                     scale: float, page_chunk: int = 8192) -> None:
    """Fill ``st.score_h`` (R, n_kv, G, MP) with the per-head page LSE for the
    selectable region [0, n_sel_hi[r]) of each request. sm_120 core-logic twin
    of ``r8i4_score_cuda.r8i4_score_only``'s six-slab BT launch. seq_lens is
    unused (every scored page is finalized, as in the kernel)."""
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print("[locks] r8i4 SCORE: sm_120 torch core-logic path ACTIVE "
              "(fp32, bitwise == fp32 r8i4 reference; Hopper CUDA lane removed)",
              flush=True)
    n_kv, G, d, page = int(st.n_kv), int(st.G), int(st.d), int(st.page)
    v4, vs, c8, cs, mu8, mus, _tag = st.layer_state(layer)
    RNK = int(v4.shape[-2])
    NB = int(v4.shape[0])

    # Engine memory-profile pass: the placeholder block table is narrower than
    # the selectable range (the kernel zeros score_h and lets combine/select run
    # deterministic; outputs discarded). Mirror it exactly.
    if int(block_table.shape[1]) < int(st.max_pages):
        st.score_h.zero_()
        return
    st.score_h.zero_()

    # FULL fp32: TF32 (10-bit mantissa) injects ~1e-3 error into the einsums and
    # flips near-tie pages at the top-b boundary; the r8i4 score is a real-valued
    # estimator, so compute it in true fp32 -> bitwise-identical to the fp32
    # ground-truth reference (gate: maxabs 0, page-set Jaccard 1.0). Disable TF32
    # only for THIS computation (single-stream decode; restored immediately) so the
    # model forward's TF32 speed is untouched.
    _tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    _prof = _os.environ.get("LOCKS_R8I4_TIME") == "1"
    if _prof:
        global _SCORE_MS, _SCORE_N
        _e0 = torch.cuda.Event(enable_timing=True); _e1 = torch.cuda.Event(enable_timing=True)
        _e0.record()
    try:
        _r8i4_score_torch_fp32(st, layer, q, block_table, n_req, scale, page_chunk,
                               n_kv, G, d, page, v4, vs, c8, cs, mu8, mus)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32
    if _prof:
        _e1.record(); torch.cuda.synchronize()
        _SCORE_MS += _e0.elapsed_time(_e1); _SCORE_N += 1
        if _SCORE_N % 240 == 0:
            print(f"[locks] r8i4 torch score PROFILE: {_SCORE_N} calls, "
                  f"{_SCORE_MS/_SCORE_N:.2f} ms/call, {_SCORE_MS/1000:.1f}s total",
                  flush=True)


def _r8i4_score_torch_fp32(st, layer, q, block_table, n_req, scale, page_chunk,
                           n_kv, G, d, page, v4, vs, c8, cs, mu8, mus) -> None:
    NB = int(v4.shape[0])
    RNK = int(v4.shape[-2])
    # q int8 round-trip per (kv-head, group), matching torch_ref / the i8-q kernel.
    # q is the CALLER's (max_reqs, n_kv*G, d) buffer: on a PADDED batch its row
    # count is max_reqs, not n_req (the CUDA kernel indexes rows [0, n_req) and
    # ignores the padding). Slice before the reshape -- a bare
    # q.reshape(n_req, ...) raised "shape '[n_req,n_kv,G,d]' is invalid for
    # input of size max_reqs*n_kv*G*d" the moment max_reqs > n_req.
    qv = q[:n_req].reshape(n_req, n_kv, G, d).float()
    qs = qv.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
    qref = torch.round(qv / qs) * qs                      # (R, n_kv, G, d)

    nsh = st.n_sel_hi[:n_req].to(torch.int64)
    s = float(scale)
    max_hi = int(nsh.max().item())
    if max_hi <= 0:
        return
    # OPT: batch over ALL requests + a large page tile so each decode step issues
    # a handful of big GPU kernels instead of O(n_req * n_chunks) small ones
    # (the smoke was launch-overhead bound). Bitwise-identical fp32 math: each
    # score_h[r,k,g,p] is an independent LSE over page p's tokens, so batching the
    # r/p dims changes nothing. Short requests are padded to max_hi (blocks
    # clamped in-range; their [nsh[r]:max_hi) scores are outside the selectable
    # region -> zeroed below to match the kernel's write range exactly).
    bt_all = block_table[:n_req, :max_hi].to(torch.int64).clamp_(0, NB - 1)  # (R,max_hi)
    chunk = page_chunk if (page_chunk and page_chunk > 0) else max_hi
    for c0 in range(0, max_hi, chunk):
        c1 = min(c0 + chunk, max_hi)
        blk = bt_all[:, c0:c1]                            # (R, P)
        v4b = v4[blk].to(torch.int16)                     # (R, P, n_kv, RNK, d//2)
        vsb = vs[blk].float()                             # (R, P, n_kv, RNK)
        Cref = c8[blk].float() * cs[blk].float()[..., None]     # (R,P,n_kv,page,RNK)
        muref = mu8[blk].float() * mus[blk].float()[..., None]  # (R,P,n_kv,d)
        R, P = int(blk.shape[0]), int(blk.shape[1])
        # unpack int4 column-major: lo nibble -> even-d row, hi -> odd-d row;
        # 4-bit two's-complement sign-extend ((n^8)-8).
        lo = ((v4b & 0xF) ^ 8) - 8                        # (R,P,n_kv,RNK,d//2)
        hn = (((v4b >> 4) & 0xF) ^ 8) - 8
        Vq = torch.empty(R, P, n_kv, RNK, d, device=v4b.device, dtype=torch.float32)
        Vq[..., 0::2] = lo.float()
        Vq[..., 1::2] = hn.float()
        Vref = (Vq * vsb[..., None]).transpose(-1, -2)             # (R,P,n_kv,d,RNK)
        qt = torch.einsum("rpkdn,rkgd->rpkgn", Vref, qref)         # (R,P,n_kv,G,RNK)
        tok = torch.einsum("rpktn,rpkgn->rpkgt", Cref, qt)         # (R,P,n_kv,G,page)
        mud = torch.einsum("rpkd,rkgd->rpkg", muref, qref)         # (R,P,n_kv,G)
        S = torch.logsumexp((tok + mud[..., None]) * s, dim=-1)    # (R,P,n_kv,G)
        st.score_h[:n_req, :, :, c0:c1] = S.permute(0, 2, 3, 1)    # (R,n_kv,G,P)
    # match the kernel's per-request write range: zero the padded tail [nsh[r]:max_hi)
    for r in range(n_req):
        hr = int(nsh[r].item())
        if hr < max_hi:
            st.score_h[r, :, :, hr:max_hi] = 0.0

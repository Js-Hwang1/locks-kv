"""topb_select — HOT Triton static top-b page selection (kv-union group-max).

The r8 score is already group-maxed across the GQA group in ``r8_score`` (the
kv-union), so selection is a plain per-(request, kv-head) top-b over the
SELECTABLE region [sink_pages, n_sel_hi), plus always-keeping the sink pages,
the recent window and (subsumed by the window) the partial tail.  STATIC budget:
``b`` is the same for all heads and all layers of a step (derived once from the
per-step page count + config budget); there is NO adaptive per-head coverage.

Reuses the rank-threshold / union-pack structure of
``vllm/kernels/select_pages.py`` (``_rank_threshold_kernel_b`` MODE 0 +
``_union_pack_kernel_b``), collapsed into ONE kernel per (req, kv) because the
group union is already folded into the score.  Grid (n_req, n_kv); fixed launch
shapes, no host sync, no allocation -> full-CUDA-graph safe.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



# --------------------------------------------------------------------------- #
# derive_page_params: per-request (n_pages, n_sel_hi, static b) once per step. #
# ceil(budget * n_selectable) in fp64 for host-parity; clamped to [1, S].      #
# --------------------------------------------------------------------------- #
@triton.jit
def _derive_params_kernel(sl_ptr, budget_ptr, npg_ptr, nsh_ptr, b_ptr,
                          n_reqs, PAGE: tl.constexpr, WIN: tl.constexpr,
                          SINK: tl.constexpr, BUDGET_ABS: tl.constexpr,
                          R_PAD: tl.constexpr):
    offs = tl.arange(0, R_PAD)
    mask = offs < n_reqs
    sl = tl.load(sl_ptr + offs, mask=mask, other=0)
    npg = (sl + (PAGE - 1)) // PAGE
    nsh = npg - WIN
    s = nsh - SINK                               # n_selectable
    if BUDGET_ABS > 0:                           # FIXED-B static budget (pages)
        b = tl.minimum(BUDGET_ABS, tl.maximum(s, 1))
        b = tl.maximum(b, 1)
    else:                                        # fixed fraction (1-cr)
        budget = tl.load(budget_ptr)             # fp64
        bf = tl.math.ceil(s.to(tl.float64) * budget)
        b = tl.minimum(tl.maximum(bf.to(tl.int32), 1), tl.maximum(s, 1))
    tl.store(npg_ptr + offs, npg, mask=mask)
    tl.store(nsh_ptr + offs, nsh, mask=mask)
    tl.store(b_ptr + offs, b, mask=mask)


def derive_page_params(st, seq_lens, n_req: int) -> None:
    """Refresh st.{n_pages, n_sel_hi, b_fix} for this decode step (one launch)."""
    assert n_req <= st.max_reqs
    _derive_params_kernel[(1,)](
        seq_lens, st.budget64, st.n_pages, st.n_sel_hi, st.b_fix, n_req,
        PAGE=st.page, WIN=st.window_pages, SINK=st.sink_pages,
        BUDGET_ABS=int(getattr(st, "budget_abs", 0)),
        R_PAD=triton.next_power_of_2(max(st.max_reqs, 2)), num_warps=1)


# --------------------------------------------------------------------------- #
# topb_select: static top-b on the group-maxed score, kv-union already folded. #
# --------------------------------------------------------------------------- #
@triton.jit
def _topb_select_kernel(score_ptr, npg_ptr, nsh_ptr, b_ptr,
                        tab_ptr, cnt_ptr, n_sink,
                        stride_sr, stride_sh, mp,
                        stride_tabr, stride_tabh, stride_cntr,
                        P_PAD: tl.constexpr, S_PAD: tl.constexpr):
    r = tl.program_id(0)
    kh = tl.program_id(1)
    n_pages = tl.load(npg_ptr + r)               # 0 for padded request rows
    n_sel_hi = tl.load(nsh_ptr + r)
    b = tl.load(b_ptr + r)
    base = r * stride_sr + kh * stride_sh
    offs_p = tl.arange(0, P_PAD)
    pmask = offs_p < n_pages
    score = tl.load(score_ptr + base + offs_p, mask=pmask, other=float("-inf"))

    selmask = (offs_p >= n_sink) & (offs_p < n_sel_hi)
    keep_all = (n_sel_hi - n_sink) <= 1          # too short: attend all (exact)

    # rank threshold over the SELECTABLE slice only.
    offs_s = tl.arange(0, S_PAD)
    smask = offs_s < (n_sel_hi - n_sink)
    sv = tl.load(score_ptr + base + n_sink + offs_s, mask=smask,
                 other=float("-inf"))
    srt = tl.sort(sv, descending=True)
    keepvec = offs_s < b                         # static b (per request)
    tau = tl.min(tl.where(keepvec, srt, float("inf")), axis=0)

    always = (offs_p < n_sink) | (offs_p >= n_sel_hi)   # sinks + window + tail
    # DETERMINISTIC EXACT-B (user ruling 2026-07-20: threshold-tie
    # semantics abolished).  Keep every selectable page STRICTLY above tau
    # (g of them; g <= b-1 since tau is the b-th largest) plus exactly
    # quota = b - g pages EQUAL to tau, lowest page index first (cumsum in
    # ascending page order).  Selectable kept == b by construction; fp32
    # collisions at tau can no longer inflate the count.
    gtmask = selmask & (score > tau)
    g = tl.sum(gtmask.to(tl.int32), axis=0)
    quota = b - g
    eqmask = selmask & (score == tau)
    eqrank = tl.cumsum(eqmask.to(tl.int32), axis=0)     # inclusive, asc page
    keep = (gtmask | (eqmask & (eqrank <= quota)) | always | keep_all) \
        & pmask

    # in-kernel cumsum compaction -> ascending page ids, -1 padded.
    ki = keep.to(tl.int32)
    pos = tl.cumsum(ki, axis=0)                  # 1-based rank at kept pages
    cnt = tl.sum(ki, axis=0)
    tl.store(cnt_ptr + r * stride_cntr + kh, cnt)
    tabbase = tab_ptr + r * stride_tabr + kh * stride_tabh
    for c0 in range(0, mp, P_PAD):               # -1 pads: [cnt, mp)
        offs_c = c0 + offs_p
        tl.store(tabbase + offs_c, tl.full([P_PAD], -1, tl.int32),
                 mask=(offs_c >= cnt) & (offs_c < mp))
    tl.store(tabbase + (pos - 1), offs_p.to(tl.int32), mask=keep)


_TOPB_TORCH_SAID = False   # one-shot: capacity dispatch to the torch twin


def _topb_select_torch(st, n_req: int) -> None:
    """EXACT top-b, PAGE-COUNT-INDEPENDENT (torch.topk; no shared-memory sort).

    The Triton kernel sorts the whole selectable slice in SMEM
    (``S_PAD = next_pow2(n_pages)`` fp32), i.e. smem = O(total pages), which
    hard-caps the reachable context (sm_120 99 KB -> 16384 pages -> 262144
    tokens; a 1M-context model needs 65536 pages = 256 KB). Exact top-b only
    needs O(b), so this path uses ``torch.topk`` and is unbounded in page count.

    Semantics are bit-for-bit the kernel's (gated):
        tau  = b-th largest score over the SELECTABLE slice [sink, n_sel_hi)
        keep = {selectable: score > tau}
             U {selectable: score == tau} taken LOWEST PAGE INDEX FIRST until
               exactly b selectable pages are kept          (deterministic exact-b)
             U [0, sink) U [n_sel_hi, n_pages)              (sinks + window/tail)
        (n_sel_hi - sink) <= 1  ->  keep every page in [0, n_pages)
    ``page_table`` is ascending page ids, -1 padded; ``page_cnt`` the count.
    """
    n_kv, MP = st.n_kv, st.max_pages
    sink = int(st.sink_pages)
    dev = st.score.device
    idx_row = torch.arange(MP, device=dev, dtype=torch.int32)
    rows_full = torch.arange(n_kv, device=dev)[:, None].expand(n_kv, MP)
    for r in range(n_req):
        npg = int(st.n_pages[r].item())
        nsh = int(st.n_sel_hi[r].item())
        b = int(st.b_fix[r].item())
        S = st.score[r]                                   # (n_kv, MP)
        keep = torch.zeros(n_kv, MP, dtype=torch.bool, device=dev)
        lo = min(sink, max(npg, 0))
        hi = max(min(nsh, npg), lo)
        L = hi - lo
        if L <= 1:                                        # too short: attend all
            if npg > 0:
                keep[:, :npg] = True
        else:
            if lo > 0:
                keep[:, :lo] = True                       # sink pages
            if hi < npg:
                keep[:, hi:npg] = True                    # recent window + tail
            sel = S[:, lo:hi]                             # (n_kv, L)
            if b >= L:
                keep[:, lo:hi] = True
            else:
                tau = torch.topk(sel, b, dim=-1, largest=True,
                                 sorted=True).values[:, -1]        # b-th largest
                gt = sel > tau[:, None]
                quota = (b - gt.sum(-1)).clamp_min(0)              # ties to fill
                eq = sel == tau[:, None]
                eqrank = eq.to(torch.int32).cumsum(-1)             # asc page order
                keep[:, lo:hi] = gt | (eq & (eqrank <= quota[:, None]))
        if npg < MP:
            keep[:, npg:] = False
        st.page_cnt[r, :] = keep.sum(-1).to(st.page_cnt.dtype)
        pos = keep.to(torch.int32).cumsum(-1)
        st.page_table[r].fill_(-1)
        st.page_table[r][rows_full[keep], (pos - 1)[keep].long()] = \
            idx_row[None, :].expand(n_kv, MP)[keep]


def topb_select(st, n_req: int) -> None:
    """HOT Stage-A.2: write st.page_table (R, n_kv, MP) / st.page_cnt (R, n_kv)
    from st.score.  Static budget from st.b_fix (derive_page_params)."""
    n_kv, MP = st.n_kv, st.max_pages
    P_PAD = triton.next_power_of_2(MP)
    S_PAD = triton.next_power_of_2(
        max(MP - st.window_pages - st.sink_pages, 2))
    # CAPACITY DISPATCH, EXPLICIT AND LOUD (no-fallback rule, 2026-07-22).
    # The Triton kernel sorts the selectable slice in SMEM (S_PAD fp32 =
    # O(total pages)), so it cannot launch past the device's opt-in cap
    # (sm_120 99 KB -> 16384 pages -> 262144 tokens; 1M ctx needs 256 KB).
    # The torch twin computes the SAME selected set (same tau, same
    # count-and-demote tie rule, same ascending compaction) in O(b).
    # Two things it is NOT: it is not bitwise-gated in-engine yet (no
    # EXACT_TOPB_VERIFICATION artifact on disk at the time of this edit), and
    # it is NOT capture-safe -- it host-syncs per request (.item()), so
    # letting it run inside a CUDA-graph capture would poison the graph.
    # Assert that precondition instead of discovering it as a corrupt replay.
    from .. import arch as _arch
    if S_PAD * 4 > _arch.smem_cap():
        assert not torch.cuda.is_current_stream_capturing(), (
            "locks topb_select: the Triton selectable-slice sort needs "
            f"{S_PAD * 4}B of shared memory > the device cap "
            f"{_arch.smem_cap()}B, and the page-count-independent torch "
            "top-b twin host-syncs (not capture-safe). Use the hand-CUDA "
            "selector (unset LOCKS_TOPB_TRITON) at this context length.")
        global _TOPB_TORCH_SAID
        if not _TOPB_TORCH_SAID:
            _TOPB_TORCH_SAID = True
            print(f"[locks] topb_select: TORCH exact top-b (Triton sort "
                  f"needs {S_PAD * 4}B smem > cap {_arch.smem_cap()}B) -- "
                  "same selected set, eager only", flush=True)
        _topb_select_torch(st, n_req)
        return
    _topb_select_kernel[(n_req, n_kv)](
        st.score, st.n_pages, st.n_sel_hi, st.b_fix,
        st.page_table, st.page_cnt, st.sink_pages,
        st.score.stride(0), st.score.stride(1), MP,
        st.page_table.stride(0), st.page_table.stride(1), st.page_cnt.stride(0),
        P_PAD=P_PAD, S_PAD=S_PAD, num_warps=4)

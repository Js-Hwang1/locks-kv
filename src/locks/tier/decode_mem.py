"""decode_mem -- the DYNAMIC-budget MEM tiered decode (Stage B for mem-v/mem-kv).

This module WIRES the previously-shadow mem tiered decode: it consumes the
Stage-A selection interface

    st.page_table   (R, n_kv, MP) int32   ascending logical page ids, -1 padded
    st.page_cnt     (R, n_kv)     int32   VARIABLE per (layer, kv-head) unit

where the DYNAMIC budget makes ``page_cnt`` differ per unit (Sum_u b_u
conserved by the selector; sink + window + partial tail forced into every
unit's set, exactly like the static selector), and decodes over the tier's
3-way KV union:

    hot buffer   (VRAM, exact-LRU residency)        complete + resident pages
    staging pool (VRAM, bounded working set)        partial tail / un-flushed
    pinned pool  (host DRAM, UVA zero-copy)         flushed pages (fallback)

per the ``V_SRC==1`` / ``K_SRC==1`` seam already compiled into the Triton
split-K decode (attention/decode.py). Two launches per layer:

  1. residency gather for THIS layer's selection: ``gather_plan`` (miss
     classification + exact-LRU victim assignment) then the miss FETCH
     (pinned -> hot over PCIe, staged -> hot in-VRAM). Both are graph-safe
     and VARIABLE-COUNT NATIVE: every kernel guards on the per-unit
     ``page_cnt``/``miss_cnt`` device values; no launch shape depends on b_u.
  2. the tiered split-K decode + merge (V from hot|staged|pinned; K likewise
     for mem-kv), reading the SAME page_table/page_cnt.

Fetch transports (``fetch=``):

  * ``"kernel"``  the UVA gather kernel: SM-resident zero-copy reads.
                  GRAPH-SAFE (fixed grid, no host sync) -> the only transport
                  legal inside CUDA-graph capture. Holds SMs while stalled on
                  PCIe (measured ~13 GB/s effective on scattered pages).
  * ``"dma"``     copy-engine DMA (cudaMemcpy2DAsync over coalesced physical-
                  block runs) for the FLUSHED misses + the kernel for STAGED
                  misses. Host-planned (device->host miss list sync) -> NOT
                  graph-capturable, but the transfer itself runs on the copy
                  engine at ~2-3x the UVA kernel's bandwidth and does not
                  steal SMs from the decode.
  * ``"off"``     no gather. Decode still correct (complete non-resident
                  pages fall through to pinned zero-copy reads) but every
                  cold page pays the UVA latency -- the hot buffer is the
                  performance play, not a correctness requirement.

The decode kernel is the golden Triton ``sparse_decode_split_kernel`` --
bitwise-equal output to the resident (V_SRC==0) path given equal bytes, which
is exactly what the tier guarantees (pure int16 bit copies end to end). The
hand-CUDA decode (decode_cuda.py) now carries a matching ``V_SRC==1`` tiered arm
(``tier_page()``), reached via ``mem_dynamic_decode(impl="cuda")`` and
bitwise-verified against this Triton reference plus the resident CUDA decode
(tests/test_mem_cuda.py), so ``ptrsel`` and ``cuda`` are interchangeable.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from ..attention.decode import _pps_min, sparse_paged_decode_batched
from ..attention.merge import (_ORDER2, _tree_sum_rows, _tree_sum_vec,
                               merge_splits_kernel)

_FETCH_MODES = ("kernel", "dma", "off")

# FMERGE-v2 (2026-07-19): fuse the split-K merge INTO the split kernel
# (last-CTA ticket merge).  The K8a layout refutation is resolved by the
# ORDER2 golden merge (merge.py): both the standalone reference and the
# in-kernel twin use the same LOGICAL halving op sequence => bitwise-equal
# across kernels by construction.  Requires LOCKS_MERGE_ORDER2=1 process-
# wide (asserted below) so every resident reference agrees.
_FMERGE = os.environ.get("LOCKS_MEM_FMERGE", "0") == "1"
if _FMERGE and not _ORDER2:
    raise RuntimeError("LOCKS_MEM_FMERGE=1 requires LOCKS_MERGE_ORDER2=1 "
                       "(the golden-order merge; K8a layout refutation)")

# K9 (2026-07-19): programmatic dependent launch (PDL) on the Triton
# split->merge chain.  Each armed kernel executes griddepcontrol.wait at its
# TOP (before any load) -- the exact visibility guarantee of a kernel
# boundary, so bytes and math are untouched -- and triggers its dependents
# right after, so the successor's launch/ramp overlaps the predecessor's
# body/drain instead of serializing behind it.  The CUDA score/select
# kernels have been PDL-armed since the chain campaigns (LOCKS_PDL=1); this
# closes the remaining un-armed boundaries (select->split, split->merge).
# Requires the K8 schedule (the SIDE fork's event-record nodes otherwise sit
# between select and split in the capture and break the programmatic edge).
# Kill switch: LOCKS_MEM_PDL=0.
_PDL = os.environ.get("LOCKS_MEM_PDL", "0") == "1"
# K9-v2 pre-wait prologue (LOCKS_MEM_PDLV2=0 restores the v1 wait-at-top
# placement -- the matched-pair ctl arm).
_PDLV2 = os.environ.get("LOCKS_MEM_PDLV2", "0") == "1"

# K8d (2026-07-19): absorb the per-layer KV-write scatter into the split
# kernel's tail-page CTA at bs=1 pure decode (mem-kv only: under mem-v the
# engine-K read/write aliasing inside one kernel is a compiler-reordering
# hazard; mem-kv reads K from the tier pools whose aliasing IS visible).
# The write hook stashes (lidx, key, value, K-engine-view, slot_mapping);
# the ptrsel launch consumes it.  An unconsumed stash raises (the stash
# condition mirrors _forward_mem's mql==1 split exactly; a miss means the
# proof failed and bytes would be wrong -- loud, never silent).
# Kill switch: LOCKS_MEM_K8D=0.
_K8D = os.environ.get("LOCKS_MEM_K8D", "0") == "1"
# NOROPE (round A): the write-hook stash holds UNROPED k at nt==1 (the
# forward's rope kernel is deleted there); the SCAT phase ropes it before
# both stores with the inductor kernel's exact math (U3-locked formula).
# q is NOT roped here: the impls pass st.q_rope (the scorer's publish).
_NOROPE = (os.environ.get("LOCKS_NOROPE", "0") == "1"
           or os.environ.get("LOCKS_QFIRST_CO", "0") == "1")

# FA1 (2026-07-19, user design): NO residency cache -- fetch the ENTIRE
# selection per layer into a selection-indexed landing buffer on a side
# stream (tier.fa1_step, forked after select), racy-published with epoch
# flags (clk*L + lidx).  The split RACES the fetch: per would-be-pinned
# page it acquire-checks the flag and reads landing (VRAM) if landed, else
# pinned in place -- identical bytes either way (I2), so output stays
# bitwise.  The K5 control plane (k5c claims, assign, transport, promote
# payloads) is INERT under FA1 at bs=1 (no appends; head keeps the clock
# tick).  Budget-fixed bytes => the offload cost is ctx-INVARIANT by
# design.  Kill switch: LOCKS_MEM_FA1=0.
_FA1 = os.environ.get("LOCKS_MEM_FA1", "0") == "1"
# FA1-v2: ONE join at the graph end instead of per-layer (the side
# stream serializes fetches; each fetch is covered by the following
# layers).  Requires FULL-graph decode capture (the deployed mode);
# per-layer/piecewise capture keeps the per-layer join (suite default).
_FA1_DEEP = os.environ.get("LOCKS_MEM_FA1_DEEP", "0") == "1"
# FA1-v3: {blk,gen} tag acceptance (int64 flags, per-layer landing sized to
# the R1 budget) -- the fetch skips landed+unchanged bytes, so per-step
# traffic drops to the true churn.  Split predicate: fl == (gen[blk]<<32|blk)
# (a stale or missing tag falls back to the racy pinned read, I2).
_FA1_V3 = os.environ.get("LOCKS_MEM_FA1_V3", "0") == "1"

# S3 (2026-07-19): the resolved (slot, vb, blk) step table (built at the K8
# graph head, tier.k8_head) replaces the split kernel's 3-deep dependent
# per-page load chain.  Same values by construction (step-stable maps; the
# in-step k5c -1->-2 claims are invisible to readers either way) => same
# source decisions, same bytes, bitwise output.  bs=1 K8 path only.
# Kill switch: LOCKS_MEM_S3=0.
_S3 = os.environ.get("LOCKS_MEM_S3", "0") == "1"


def k8d_active() -> bool:
    """Env-static half of the K8d consume condition (the hook's proof)."""
    from .k5 import _K5_STATS, _K5_V7, _K8
    return (_K8D and _K8 and _K5_V7 and not _K5_STATS
            and os.environ.get("LOCKS_TIER_SIDE", "0") == "1"
            and os.environ.get("LOCKS_MEM_DECODE", "auto") in ("auto",
                                                              "ptrsel"))


# K8b (2026-07-19): absorb the K7 _k5c control plane INTO the split kernel.
# Every input _k5c needs (pt, blk, complete, slot, vb) is ALREADY a register
# value of the split loop; the k5 side-band ops (hit epoch touch, miss CAS
# claim -1->-2, lost_v/staged reverts, pinned mailbox append, MCAP overflow
# latch) are added under runtime guards, so the decode's output math and its
# reads are untouched (a page this kernel claims was read as pinned either
# way; readers treat -1/-2 identically).  Deletes, per step at bs=1: 40
# _k5c + 40 _step_setup launches and all 160 SIDE fork/join graph edges; the
# clock advances once per step at the graph head (tier.k8_head) and the
# assign recency window rescales L -> 1 in k5.py (pure eviction policy,
# bitwise-free).  Dispatch mirrors K7's own gate: bs=1 + LOCKS_K5_V7 + the
# PLAN/SIDE schedule + impl=ptrsel, and never under LOCKS_K5_STATS (the
# diagnostic counters live in the legacy _k5c launch path).  The _K8 flag
# itself lives in k5.py (single source; the assign window reads it too).


# --------------------------------------------------------------------------- #
# PTR-SELECT tiered split-K decode: one K load + one V load per page.          #
#                                                                              #
# The shared kernel (attention/decode.py, V_SRC==1) issues THREE predicated    #
# loads + two tl.where merges per tensor per page (hot | staged | pinned).     #
# Measured in-graph on H200 that costs +49% over the resident decode (44.3 vs  #
# 29.8us, 4x16K reqs @10%). Here the 3-way select happens on the ADDRESS       #
# (scalar per page) instead of on the DATA: all three sources are addressed    #
# as int16 (pure bits; hot_i16 / v_pool.view(int16) / pinned pool int16), one  #
# load fetches the tile, one bitcast recovers bf16. Bit-identical values by    #
# construction (same bytes, same math), fewer memory instructions.             #
# --------------------------------------------------------------------------- #
@triton.jit
def _mem_decode_split_kernel(q_ptr, bt_ptr, tab_ptr, cnt_ptr, sl_ptr,
                             m_ptr, l_ptr, acc_ptr, sm_scale,
                             stride_qt, stride_qh,
                             kres_ptr, stride_kb, stride_kt, stride_kh,
                             hotv_ptr, hotk_ptr,
                             stgv_ptr, stgk_ptr, svb, svt, svh,
                             poolv_ptr, poolk_ptr,
                             p2s_ptr, vbo_ptr,
                             stride_btr, stride_tabr, stride_tabh,
                             stride_cntr, stride_pr,
                             t_lidx, t_NB, t_S,
                             o_ptr, stride_ot, stride_oh, ctr_ptr,
                             epoch_ptr, clock_ptr, valid_ptr, gen_ptr,
                             mail_ptr, mail_cnt_ptr, ovf_ptr, t_MCAP,
                             key_ptr, val_ptr, kc_ptr, slotmap_ptr, vsm_ptr,
                             skh_, svh2, kcb, kct, kch, csc_ptr,
                             s3_ptr, t_MP,
                             fa1_lv, fa1_lk, fa1_fl, fa1_so, t_L,
                             G: tl.constexpr, G_PAD: tl.constexpr,
                             PAGE: tl.constexpr, D: tl.constexpr,
                             SPLIT: tl.constexpr, PPS_MIN: tl.constexpr,
                             K_TIER: tl.constexpr, NKV: tl.constexpr,
                             SPLIT_PAD: tl.constexpr, FMERGE: tl.constexpr,
                             K5C: tl.constexpr, PDL: tl.constexpr = False,
                             SCAT: tl.constexpr = False,
                             S3: tl.constexpr = False,
                             PW: tl.constexpr = True,
                             FA1: tl.constexpr = False,
                             LCAP: tl.constexpr = 256,
                             NOROPE: tl.constexpr = False,
                             FA1V3: tl.constexpr = False):
    r = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    if PDL and PW:
        # K9-v2: pre-wait prologue.  Everything whose producer completed
        # BEFORE the predecessor (select) even started is loaded/computed
        # while select drains: seq_len (host prologue), the q tile (QKV gemm,
        # several kernels back), index frames.  The measured wait window
        # here is ~2.8us/layer (v1 spent it idle; nsys_r3 evidence).
        pw_sl = tl.load(sl_ptr + r)
        pw_og = tl.arange(0, G_PAD)
        pw_od = tl.arange(0, D)
        pw_gm = pw_og < G
        pw_q = tl.load(q_ptr + r * stride_qt
                       + (kh * G + pw_og)[:, None] * stride_qh
                       + pw_od[None, :], mask=pw_gm[:, None], other=0.0)
        gdc_wait()
        gdc_launch_dependents()
    elif PDL:
        gdc_wait()
        gdc_launch_dependents()
    cnt = tl.load(cnt_ptr + r * stride_cntr + kh)
    if SCAT:
        # K8d: the deferred KV write for THIS step's token (bs=1 decode),
        # absorbed from _scatter_kv_engine_kernel.  Executed by the CTA whose
        # j-range holds the TAIL page, BEFORE any load of that page (program
        # order); no other CTA reads the tail page, and nothing between the
        # write hook and this kernel reads raw K/V bytes (score reads
        # summaries, select reads score_h, k5c reads maps).  Same bytes,
        # same destinations as the standalone scatter: K -> engine K-only
        # cache at slot_mapping[0], V -> v_pool at v_slot_mapping[0]
        # (+ K -> k_pool under K_TIER).  Each CTA writes ITS kv-head slice.
        pps0 = tl.maximum(tl.cdiv(cnt, SPLIT), PPS_MIN)
        tail_mine = (cnt > 0) & (sp * pps0 <= cnt - 1) \
            & (cnt - 1 < sp * pps0 + pps0)
        if tail_mine:
            offs_d = tl.arange(0, D)
            slot = tl.load(slotmap_ptr)
            k_new = tl.load(key_ptr + kh * skh_ + offs_d)
            if NOROPE:
                # NOROPE (round A): the stash holds UNROPED k (the forward's
                # rope kernel is deleted at nt==1).  Apply the inductor
                # kernel's exact math before BOTH stores: partner via
                # offs_d^1, cos/sin from the bf16 cache row of pos = sl-1,
                # fp32 mul/mul + sub/add (separate ops, no fma), bf16
                # round; dims >= 64 pass through.  Byte-identical to
                # [triton rope -> stock scatter] (gate_norope_p1 U3).
                sl_v = tl.load(sl_ptr + r)
                crow = csc_ptr + 64 * (sl_v - 1).to(tl.int64)
                part = tl.load(key_ptr + kh * skh_ + (offs_d ^ 1))
                pr = offs_d // 2
                rot_m = offs_d < 64
                cv = tl.load(crow + pr, mask=rot_m,
                             other=0.0).to(tl.float32)
                sv = tl.load(crow + 32 + pr, mask=rot_m,
                             other=0.0).to(tl.float32)
                me = k_new.to(tl.float32)
                pa = part.to(tl.float32)
                even = (offs_d % 2) == 0
                rot = tl.where(even, me * cv - pa * sv, me * cv + pa * sv)
                kf = tl.where(rot_m, rot, me)
                k_new = kf.to(key_ptr.dtype.element_ty)
            if slot >= 0:
                blk_w = (slot // PAGE).to(tl.int64)
                off_w = slot % PAGE
                tl.store(kc_ptr + blk_w * kcb + off_w * kct + kh * kch
                         + offs_d, k_new)
            vs = tl.load(vsm_ptr)
            if vs >= 0:
                v_new = tl.load(val_ptr + kh * svh2 + offs_d)
                vb_w = (vs // PAGE).to(tl.int64)
                voff_w = vs % PAGE
                dstw = vb_w * svb + voff_w * svt + kh * svh + offs_d
                tl.store(stgv_ptr + dstw, v_new.to(tl.int16, bitcast=True))
                if K_TIER == 1:
                    tl.store(stgk_ptr + dstw, k_new.to(tl.int16, bitcast=True))
        # CTA-scope fence: the writes above must retire before this CTA's own
        # loop reads the tail page through the SAME pools (compiler ordering
        # guarantee; the pool aliasing is via identical pointer params but
        # the barrier removes any doubt at zero steady cost).
        tl.debug_barrier()
    pps = tl.maximum(tl.cdiv(cnt, SPLIT), PPS_MIN)
    j0 = sp * pps
    if j0 >= cnt:
        # padded request rows (cnt == 0 everywhere): the standalone merge
        # writes zeros for every head; under FMERGE sp==0 takes that duty.
        if FMERGE:
            if (cnt == 0) & (sp == 0):
                zd = tl.arange(0, D)
                for g in tl.range(0, G):
                    tl.store(o_ptr + r * stride_ot + (kh * G + g) * stride_oh
                             + zd,
                             tl.zeros([D], tl.float32).to(
                                 o_ptr.dtype.element_ty))
        return
    j1 = tl.minimum(j0 + pps, cnt)
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE)
    gmask = offs_g < G

    if PDL and PW:
        seq_len = pw_sl
        q = pw_q
    else:
        seq_len = tl.load(sl_ptr + r)
        q = tl.load(q_ptr + r * stride_qt
                    + (kh * G + offs_g)[:, None] * stride_qh
                    + offs_d[None, :], mask=gmask[:, None], other=0.0)

    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, D], tl.float32)
    if K5C:
        k5_clk = tl.load(clock_ptr)
    if FA1 and not FA1V3:
        fa_ep = tl.load(clock_ptr) * t_L + t_lidx

    for j in tl.range(j0, j1, num_stages=3):
        pt32 = tl.load(tab_ptr + r * stride_tabr + kh * stride_tabh + j)
        pt = pt32.to(tl.int64)
        if S3:
            # S3: one L2-warm 16B line replaces the 3-deep dependent chain
            # pt -> bt[pt] -> p2s/vbo[blk] (same values by construction:
            # the table is built from the step-stable maps at the head).
            tb = (kh * t_MP + pt) * 4
            slot = tl.load(s3_ptr + tb + 0)
            vb = tl.load(s3_ptr + tb + 1)
            blk32 = tl.load(s3_ptr + tb + 2)
            blk = blk32.to(tl.int64)
        else:
            blk32 = tl.load(bt_ptr + r * stride_btr + pt)
            blk = blk32.to(tl.int64)
        tok = pt * PAGE + offs_t
        tmask = tok < seq_len
        # ---- one 3-way source decision per PAGE (scalars) ----------------- #
        complete = ((pt + 1) * PAGE) <= seq_len
        if not S3:
            slot = tl.load(p2s_ptr + kh * t_NB + blk)
            vb = tl.load(vbo_ptr + blk)
        use_hot = complete & (slot >= 0)
        use_stg = (use_hot == 0) & (vb >= 0)
        if K5C:
            # ---- absorbed _k5c control plane (k5.py _k5c_kernel), one page
            # per iteration on values already in registers.  Side-band only:
            # epoch touch (policy), miss CAS claim -1->-2 (readers treat -1
            # and -2 identically), lost_v/staged reverts, pinned mailbox
            # append (consumed by next step's prologue assign), MCAP latch.
            # Claim order preserved: CAS -> classify -> revert.  The decode's
            # own source decision above used the PRE-claim slot value.
            if use_hot:
                tl.store(epoch_ptr + kh * t_S + slot, k5_clk)
            if complete & (slot == -1):
                oldp = tl.atomic_cas(p2s_ptr + kh * t_NB + blk, -1, -2)
                if oldp == -1:
                    vld = tl.load(valid_ptr + kh * t_NB + blk)
                    if (vb < 0) & (vld == 0):
                        # lost_v: revert the claim (I1 violation upstream;
                        # page stays unselectable from hot, decode reads
                        # pinned/staged in place exactly as before)
                        tl.store(p2s_ptr + kh * t_NB + blk, -1)
                    elif vb >= 0:
                        # STAGED miss: PLAN skip at source (readable staged
                        # in place, identical bytes)
                        tl.store(p2s_ptr + kh * t_NB + blk, -1)
                    else:
                        pos = tl.atomic_add(mail_cnt_ptr, 1, sem="relaxed")
                        if pos < t_MCAP:
                            gval = tl.load(gen_ptr + blk)
                            tl.store(mail_ptr + pos * 6 + 0, t_lidx)
                            tl.store(mail_ptr + pos * 6 + 1, kh)
                            tl.store(mail_ptr + pos * 6 + 2, blk32)
                            tl.store(mail_ptr + pos * 6 + 3, -1)
                            tl.store(mail_ptr + pos * 6 + 4, gval)
                        else:
                            # overflow: not cached this step; revert + latch
                            tl.store(p2s_ptr + kh * t_NB + blk, -1)
                            tl.store(ovf_ptr, 1)
        hot_off = ((kh * t_S + tl.maximum(slot, 0)).to(tl.int64) * (PAGE * D)
                   + offs_t[:, None] * D + offs_d[None, :])
        stg_off = (tl.maximum(vb, 0).to(tl.int64) * svb + kh * svh
                   + offs_t[:, None] * svt + offs_d[None, :])
        pin_off = ((((t_lidx * t_NB + blk) * NKV + kh).to(tl.int64)
                    * (PAGE * D)) + offs_t[:, None] * D + offs_d[None, :])
        if FA1:
            # racy landing check: acquire orders the tile loads below after
            # the fetch's release; a lost race reads pinned in place --
            # identical bytes (I2), bitwise-identical output.
            if FA1V3:
                # v3.2 acceptance: rank j's PERSISTENT slot via slot_of
                # (racy-safe indirection), verified content-keyed -- the
                # tag at that slot must equal (gen[blk]<<32)|blk at the
                # CURRENT generation (rewrites bump gen -> self-miss).
                sof32 = tl.load(fa1_so + kh * LCAP + j)
                sof = tl.maximum(sof32, 0).to(tl.int64)
                flv = tl.atomic_add(fa1_fl + kh * LCAP + sof, 0,
                                    sem="acquire")
                gen_v = tl.load(gen_ptr + blk).to(tl.int64)
                want = (gen_v << 32) | blk
                use_land = (use_hot == 0) & (use_stg == 0) & (sof32 >= 0) \
                    & (flv == want) & (j < LCAP)
                land_off = ((kh * LCAP + sof) * (PAGE * D)).to(tl.int64) \
                    + offs_t[:, None] * D + offs_d[None, :]
            else:
                fl = tl.atomic_add(fa1_fl + kh * LCAP + j, 0, sem="acquire")
                use_land = (use_hot == 0) & (use_stg == 0) & (fl == fa_ep) \
                    & (j < LCAP)
                land_off = ((kh * LCAP + j) * (PAGE * D)).to(tl.int64) \
                    + offs_t[:, None] * D + offs_d[None, :]
        # ---- K: resident (K_TIER=0) or address-selected tier read --------- #
        if K_TIER == 0:
            k = tl.load(kres_ptr + blk * stride_kb + kh * stride_kh
                        + offs_t[:, None] * stride_kt + offs_d[None, :],
                        mask=tmask[:, None], other=0.0)
        else:
            if FA1:
                kp = tl.where(use_hot, hotk_ptr + hot_off,
                              tl.where(use_stg, stgk_ptr + stg_off,
                                       tl.where(use_land, fa1_lk + land_off,
                                                poolk_ptr + pin_off)))
            else:
                kp = tl.where(use_hot, hotk_ptr + hot_off,
                              tl.where(use_stg, stgk_ptr + stg_off,
                                       poolk_ptr + pin_off))
            k = tl.load(kp, mask=tmask[:, None], other=0
                        ).to(tl.bfloat16, bitcast=True)
        # ---- V: address-selected tier read (always) ------------------------ #
        if FA1:
            vp = tl.where(use_hot, hotv_ptr + hot_off,
                          tl.where(use_stg, stgv_ptr + stg_off,
                                   tl.where(use_land, fa1_lv + land_off,
                                            poolv_ptr + pin_off)))
        else:
            vp = tl.where(use_hot, hotv_ptr + hot_off,
                          tl.where(use_stg, stgv_ptr + stg_off,
                                   poolv_ptr + pin_off))
        v = tl.load(vp, mask=tmask[:, None], other=0
                    ).to(tl.bfloat16, bitcast=True)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        s = tl.where(tmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.float16), v.to(tl.float16),
                     acc=acc * alpha[:, None])
        m_i = m_new

    base = r * stride_pr + (kh * SPLIT + sp) * G
    tl.store(m_ptr + base + offs_g, m_i, mask=gmask)
    tl.store(l_ptr + base + offs_g, l_i, mask=gmask)
    tl.store(acc_ptr + (base + offs_g)[:, None] * D + offs_d[None, :], acc,
             mask=gmask[:, None])
    if FMERGE:
        # FMERGE-v2: last-arriver merge ticket (acq_rel add releases this
        # CTA's partial stores and acquires every earlier CTA's); the CTA
        # that sees old == n_act-1 merges its unit using the ORDER2 golden
        # adjacent-pair halving -- the IDENTICAL logical op sequence as
        # merge.py's ORDER2 branch, so the standalone reference and this
        # in-kernel merge are bitwise-equal BY CONSTRUCTION (the K8a layout
        # refutation does not apply to logical reshape/trans/split + adds).
        n_act = tl.cdiv(cnt, pps)
        old = tl.atomic_add(ctr_ptr + r * NKV + kh, 1)
        if old == n_act - 1:
            tl.store(ctr_ptr + r * NKV + kh, 0)   # self-clean for next launch
            offs_s = tl.arange(0, SPLIT_PAD)
            smask = offs_s < n_act
            for g in tl.range(0, G):
                idx = r * stride_pr + (kh * SPLIT + offs_s) * G + g
                m = tl.load(m_ptr + idx, mask=smask, other=float("-inf"))
                l = tl.load(l_ptr + idx, mask=smask, other=0.0)
                a = tl.load(acc_ptr + idx[:, None] * D + offs_d[None, :],
                            mask=smask[:, None], other=0.0)
                m_max = tl.max(m, axis=0)
                w = tl.where(l > 0, tl.exp(m - m_max), 0.0)
                l_tot = tl.reshape(_tree_sum_vec(l * w, SPLIT_PAD), ())
                o = _tree_sum_rows(a * w[:, None], SPLIT_PAD, D)
                o = tl.where(l_tot > 0, o / l_tot, 0.0)
                tl.store(o_ptr + r * stride_ot + (kh * G + g) * stride_oh
                         + offs_d, o.to(o_ptr.dtype.element_ty))


def mem_decode_ptrsel(q: torch.Tensor, kv_cache: Optional[torch.Tensor],
                      block_table: torch.Tensor, seq_lens: torch.Tensor,
                      st, out: torch.Tensor, tier, lidx: int, *,
                      scale: float = None, k5c: bool = False,
                      pdl: bool = False, fa1: bool = False) -> None:
    """PTR-SELECT tiered Stage-B decode (split + merge), reading K/V straight
    from the tier's pools (no VSource indirection). Bit-identical output to
    the shared V_SRC==1 kernel (same bytes, same flash math)."""
    if scale is None:
        scale = st.scale
    n_req = seq_lens.shape[0]
    G = st.G
    split = st.split
    pps_min = _pps_min(n_req)
    r5 = tier.residency
    stg = tier.staging
    vp16 = stg.v_pool[lidx].view(torch.int16)
    has_k = tier.offload_k
    if has_k:
        kres = vp16                            # dead placeholder
        skb = skt = skh = 0
        kp16 = stg.k_pool[lidx].view(torch.int16)
        hotk = r5.hot_k_i16[lidx]
        poolk = tier.pool_k.dev_view
        n_kv = r5.n_kv
    else:
        from ..attention.decode import _split_kv
        kres, _ = _split_kv(kv_cache)
        skb, skt, skh = kres.stride(0), kres.stride(1), kres.stride(2)
        n_kv = kres.shape[2]
        kp16 = vp16
        hotk = r5.hot_i16[lidx]
        poolk = tier.pool.dev_view
    d = r5.d
    if k5c:
        from .k5 import _MCAP
        k5 = tier._k5_state()
        _epoch, _clock = r5.lru_clock[lidx], r5.clock
        _valid, _gen = r5.pool_valid[lidx], k5.gen
        _mail, _mcnt, _ovf, _mcap = k5.mail, k5.mail_cnt, r5.overflow, _MCAP
    else:
        # dead pointers (K5C=0 prunes every use); any int32 tensors do
        _epoch, _clock = r5.clock, r5.clock
        _valid, _gen = r5.clock, r5.clock
        if fa1 and _FA1_V3:
            # v3 acceptance needs the CURRENT per-block generation even
            # with the k5c plane off (gen bumps come from invalidate).
            _gen = r5.gen
        _mail, _mcnt, _ovf, _mcap = r5.clock, r5.clock, r5.clock, 0
    # K8d: consume the stashed KV write into the SCAT phase (mem-kv + K8 only;
    # SCAT is independent of the k5c control plane, so FA1 consumes it too --
    # an unconsumed stash raises loudly at the next hook)
    stash = getattr(tier, "_k8d_stash", None)
    scat = ((k5c or fa1) and has_k and stash is not None and _K8D
            and stash[0] == lidx)
    _norope_scat = _NOROPE and scat
    if _norope_scat:
        assert getattr(st, "_csc_cache", None) is not None, \
            "LOCKS_NOROPE=1: st._csc_cache not wired (attnwire.publish_rope)"
        # MODEL-AGNOSTIC audit (2026-07-22): the SCAT rope inside
        # _mem_decode_split_kernel is still HARDCODED to GLM-4 geometry
        # (rotary_dim 64, interleaved pairing -- see the `crow = csc_ptr + 64 *
        # ...` staging block).  The fast arm's rope kernels were generalized
        # (score prelude + rope_and_cache[_fixc]); this Triton one was NOT, so
        # it must REFUSE any other geometry rather than cache silently wrong K.
        from ..backend import _runtime as _rt
        assert int(_rt._NOROPE_ROT) == 64 and not _rt._NOROPE_NEOX, (
            "locks mem K8d SCAT: the Triton in-kernel rope is compiled for "
            f"rotary_dim 64 interleaved, model has rotary_dim {_rt._NOROPE_ROT}"
            f" neox={_rt._NOROPE_NEOX}. Generalize the SCAT rope (twin of "
            "locks_rope_elem) before running the mem arm on this arch.")
    if scat:
        _, s_key, s_val, s_keng, s_slot = stash
        tier._k8d_stash = None
        _kc, _slotm, _vsm = s_keng, s_slot, stg.v_slot_mapping
        _skh, _svh2 = s_key.stride(1), s_val.stride(1)
        _kcb, _kct, _kch = s_keng.stride(0), s_keng.stride(1), s_keng.stride(2)
        _keyp, _valp = s_key, s_val
    else:
        _keyp = _valp = _kc = r5.clock
        _slotm = _vsm = stg.v_slot_mapping     # int64 dummy (never loaded)
        _skh = _svh2 = _kcb = _kct = _kch = 0
    _mem_decode_split_kernel[(n_req, n_kv, split)](
        q, block_table, st.page_table, st.page_cnt, seq_lens,
        st.m_part, st.l_part, st.acc_part, scale,
        q.stride(0), q.stride(1),
        kres, skb, skt, skh,
        r5.hot_i16[lidx], hotk,
        vp16, kp16,
        vp16.stride(0), vp16.stride(1), vp16.stride(2),
        tier.pool.dev_view, poolk,
        r5.page2slot[lidx], stg.vbo,
        block_table.stride(0), st.page_table.stride(0),
        st.page_table.stride(1), st.page_cnt.stride(0), st.m_part.stride(0),
        lidx, r5.NB, r5.S,
        out, out.stride(0), out.stride(1), tier.fm_ctr,
        _epoch, _clock, _valid, _gen, _mail, _mcnt, _ovf, _mcap,
        _keyp, _valp, _kc, _slotm, _vsm,
        _skh, _svh2, _kcb, _kct, _kch,
        (st._csc_cache if _norope_scat else r5.clock),
        (tier.s3_tab[lidx] if (k5c and _S3) else r5.clock),
        (r5.MP if (k5c and _S3) else 0),
        (tier.fa1_land_v3[lidx] if (fa1 and _FA1_V3)
         else tier.fa1_land_v[lidx & 1] if fa1 else r5.clock),
        (tier.fa1_land_k3[lidx] if (fa1 and _FA1_V3)
         else tier.fa1_land_k[lidx & 1] if fa1 else r5.clock),
        (tier.fa1_flags3[lidx] if (fa1 and _FA1_V3)
         else tier.fa1_flags[lidx & 1] if fa1 else r5.clock),
        (tier.fa1_slot_of[lidx] if (fa1 and _FA1_V3) else r5.clock),
        (tier.L if fa1 else 0),
        G=G, G_PAD=max(16, triton.next_power_of_2(G)), PAGE=r5.page, D=d,
        SPLIT=split, PPS_MIN=pps_min, K_TIER=1 if has_k else 0, NKV=n_kv,
        SPLIT_PAD=triton.next_power_of_2(split), FMERGE=_FMERGE,
        K5C=bool(k5c), PDL=bool(pdl), SCAT=bool(scat),
        S3=bool(k5c and _S3), PW=_PDLV2, FA1=bool(fa1),
        LCAP=(tier.FA1_LCAP3 if (fa1 and _FA1_V3)
              else tier.FA1_LCAP if fa1 else 256),
        NOROPE=bool(_norope_scat), FA1V3=bool(fa1 and _FA1_V3),
        launch_pdl=bool(pdl), num_warps=4)
    if not _FMERGE:
        merge_splits_kernel[(n_req, n_kv * G)](
            st.m_part, st.l_part, st.acc_part, st.page_cnt, out,
            out.stride(0), out.stride(1), st.m_part.stride(0),
            st.page_cnt.stride(0), G=G, D=d, SPLIT=split,
            SPLIT_PAD=triton.next_power_of_2(split), PPS_MIN=pps_min,
            PDL=bool(pdl), ORDER2=_ORDER2, PW=_PDLV2, launch_pdl=bool(pdl))


def mem_dynamic_decode(q: torch.Tensor, kv_cache: Optional[torch.Tensor],
                       block_table: torch.Tensor, seq_lens: torch.Tensor,
                       st, out: torch.Tensor, tier, lidx: int, *,
                       scale: float = None, fetch: str = "kernel",
                       impl: str = "ptrsel", tier_stepped: bool = False) -> None:
    """One layer of the mem tiered decode with VARIABLE per-unit fetch.

    Reads ``st.page_table`` / ``st.page_cnt`` (this layer's Stage-A output;
    counts vary per (request, kv-head)), brings every selected complete page
    into the VRAM hot cache (miss-only fetch, exact-LRU victims), then runs
    the tiered split-K decode + merge into ``out`` rows [0, n_req).

    ``kv_cache`` is the resident engine cache; under mem-kv (``tier.offload_k``)
    it is never dereferenced for K/V (the tier serves both) and may be a K-only
    or dummy tensor. ``fetch`` selects the miss transport (see module doc);
    only ``"kernel"`` is CUDA-graph capturable.

    ``impl``: ``"ptrsel"`` (default) = the address-selected single-load tiered
    TRITON kernel in this module. Measured in-graph on H200 (4x16K reqs, 10%
    budget, 98% hot-hit): mem-kv 44.3 -> 31.6us (+6.8% over the resident
    decode's 29.6us, down from +49%), mem-v 35.8 -> 31.2us (+6.5%); output
    BITWISE equal to both the shared kernel and the resident decode.
    ``"cuda"`` = the hand-CUDA decode's ``V_SRC==1`` tiered arm (FINAL-OPT):
    the SAME per-page ptr-select (hot | staged | pinned, int16 pure-bit
    addressing) inside ``attention/decode_cuda.py``'s split/fused kernels --
    the fast path's kernel speed with tiered bytes, batch-dispatched like the
    resident CUDA decode, bitwise-equal to the resident CUDA decode on equal
    bytes.  ``"shared"`` = the ``V_SRC==1`` 3-way-data-select seam kernel in
    attention/decode.py (kept as the cross-check reference).
    """
    if fetch not in _FETCH_MODES:
        raise ValueError(f"fetch must be one of {_FETCH_MODES}, got {fetch!r}")
    n_req = seq_lens.shape[0]
    # ---- impl resolution FIRST (K8's fused control plane is ptrsel-only) -- #
    if impl == "auto":
        impl = os.environ.get("LOCKS_MEM_DECODE", "auto")
    if impl == "auto":
        # (The sm_120 "cuda at n_req >= 8" crossover was removed 2026-07-21
        # with the sm_120 lane; ptrsel is the sole auto route on every arch
        # (the sm_90-proven path).  See ours_doc/REFUTED_ARMS_INDEX.md.)
        impl = "ptrsel"
    from .k5 import _K5_STATS, _K5_V7, _K8
    use_k8 = (_K8 and _K5_V7 and not _K5_STATS and n_req == 1
              and impl == "ptrsel" and fetch == "kernel"
              and not tier_stepped
              and os.environ.get("LOCKS_TIER_SIDE", "0") == "1")
    side_forked = False
    fa1 = use_k8 and _FA1
    if use_k8:
        # K8: no per-layer control-plane launch.  The split kernel carries
        # the absorbed _k5c (non-FA1) or nothing (FA1: the landing race
        # replaces the residency protocol); evict/promote/clock run once per
        # step at the graph head (under FA1 the walkers see empty lists --
        # the head is kept for the CLOCK tick, the FA1 epoch source).
        if lidx == 0:
            tier.k8_head(block_table, seq_lens, s3=_S3)
        if fa1:
            tier.fa1_step(lidx, st.page_table, st.page_cnt, block_table,
                          seq_lens)
            side_forked = True
    elif fetch == "kernel":
        # tier_stepped: the lookahead side schedule already ran this layer's
        # K5 classify+claim+gather on the side stream (a layer early); the
        # ev_sel wait the caller enqueued covers it.  Schedule topology, not
        # a latency fallback.
        if not tier_stepped:
            # P15 overlap schedule (LOCKS_TIER_SIDE=1): classify+plan on a
            # side stream; image mutations at graph heads only.  The decode
            # below reads the step-stable page2slot either way.  The fork is
            # CLOSED at the bottom of this function (tier.side_join), i.e. at
            # the end of THIS layer and after the decode it overlaps -- so the
            # fork/join pair is balanced inside every capture unit (per-layer,
            # piecewise or whole-step) instead of dangling to the last layer.
            if os.environ.get("LOCKS_TIER_SIDE", "0") == "1":
                tier.side_step(lidx, st.page_table, st.page_cnt, block_table,
                               seq_lens, n_req)
                side_forked = True
            else:
                tier.step(lidx, st.page_table, st.page_cnt, block_table,
                          seq_lens, n_req)
    elif fetch == "dma":
        tier.step_dma(lidx, st.page_table, st.page_cnt, block_table, seq_lens,
                      n_req, wait=True)
    # (impl was resolved at the top of this function.  Measured dispatch,
    # kept for the record: sm_90 = PTRSEL at every bucket -- the P6 paired
    # in-situ race measured ptrsel 77.10 vs cuda 120.05 us/layer at 16K b8.
    # The sm_120 "cuda at n_req >= 8" crossover was removed 2026-07-21 with
    # the sm_120 lane.  Host-static inputs only -> graph-safe per bucket.
    # LOCKS_MEM_DECODE={ptrsel|cuda|shared} forces an arm (A/B races).)
    if impl == "ptrsel":
        mem_decode_ptrsel(q, kv_cache, block_table, seq_lens, st, out, tier,
                          lidx, scale=scale, k5c=(use_k8 and not _FA1),
                          pdl=(_PDL and use_k8), fa1=fa1)
    elif impl == "cuda":
        from ..attention.decode_cuda import sparse_paged_decode_batched_cuda
        sparse_paged_decode_batched_cuda(q, kv_cache, block_table, seq_lens,
                                         st, out, tier.vsource(lidx),
                                         scale=scale)
    else:
        sparse_paged_decode_batched(q, kv_cache, block_table, seq_lens, st,
                                    out, tier.vsource(lidx), scale=scale)
    if side_forked:
        if fa1 and _FA1_DEEP:
            # v2: single join at the graph end -- every earlier fetch is
            # ordered behind the last one on the single side stream.
            if lidx == tier.L - 1:
                tier.side_join(lidx)
        else:
            # close this layer's side fork AFTER the decode+merge.
            tier.side_join(lidx)

"""K5 -- the tier control plane: two grid-ordered kernels, atomic claims,
transport split (LOCKS_KERNEL_REDESIGN.md v2 section 4.5; red-team F2/F3 fixes).

Replaces the steady-path duty of ``_plan_fused_kernel`` (one serial program per
kv-head, a LOAD-BEARING ``tl.debug_barrier``, measured 45.5 us mean / 231.8 us
max per layer-step on sm_120) and ``_fetch_flat_kernel`` (SM-holding UVA reads
of pinned pages at ~13 GB/s).

The protocol (4.5.2), all graph-safe:

  K5a CLASSIFY+CLAIM   grid (n_req, n_kv): per selected COMPLETE page,
      HIT  (page2slot >= 0)  -> epoch[slot] = clock   (eviction protection)
      MISS (page2slot == -1) -> atomicCAS(-1 -> PENDING=-2); the ONE winner
           across all units sharing the kv-head appends (blk, source) to its
           per-unit miss list (cross-request dedup falls out of the CAS).
           Source: STAGED (vbo >= 0) or PINNED (pool_valid).  Losers and
           already-PENDING pages are left alone: THE DECODE READS THEM FROM
           staged/pinned IN PLACE this step (bitwise-free: identical bytes).

  K5b VICTIM+INSTALL   grid (n_req, n_kv), SEPARATE launch (grid-ordered after
      K5a): per miss, a bounded CLOCK probe (at most 2S):
          h = atomicAdd(hand) mod S;  skip if epoch[h] == clock;
          CLAIM owner[h] via atomicCAS(0 -> unit tag); RE-VERIFY epoch[h]
          (claim-then-verify; release and continue on race).
      Then evict (page2slot[old] = -1), install bookkeeping (slot2page[h] =
      blk, epoch[h] = clock), and by source:
        STAGED: copy the page in-kernel (VRAM->VRAM), publish page2slot[blk]=h,
                release owner.
        PINNED: append (l, kv, blk, h, gen) to the DMA MAILBOX; page2slot
                STAYS PENDING and owner STAYS HELD until the copy-engine bytes
                land (the fence): decode reads pinned in place meanwhile.
      Probe exhaustion / mailbox overflow: revert page2slot[blk] = -1 and latch
      (the page simply is not cached this step; still bitwise).

  K5-PROMOTE           1 launch at the graph head (before layer 0's K5a):
      installs the entries whose copy-engine transfer the host has FENCED:
      atomicCAS(page2slot[blk], PENDING -> slot).  A failed CAS means the block
      was invalidated (or re-installed) since the copy was issued -> the slot
      is abandoned (slot2page = -1) and released.  A per-block GENERATION
      (bumped by invalidate_written) additionally guards the ABA case where the
      same block goes PENDING again before a stale promote entry drains: the
      entry's generation no longer matches and it is abandoned.

Transport split (4.5.3): PINNED -> hot moves ONLY by copy-engine DMA
(``K5Service``: mailbox -> coalesced 2D runs -> cudaMemcpy2DAsync on a
dedicated NEVER-CAPTURED transfer stream -> event -> promote next step).
In-kernel UVA reads of pinned pages remain only in the decode's pinned arm
(pages not yet cached; identical bytes).  STAGED -> hot stays in-kernel.

Scheduling realized in P4 (no lookahead yet): K5a/K5b run per layer on the
main stream (the sync schedule); the mailbox accumulates over the step's
layers, the host services it at the NEXT step's prologue and the transfer
lands best-effort during that step, installed by the promote one step later
(miss -> hot latency: 2 steps).  Every intervening decode reads the pinned
copy in place, so residency is bitwise-free by construction.  Under P6's
lookahead side-stream the same mailbox is simply drained a layer early.

CLOCK replaces exact LRU (policy is bitwise-free: hot sizing/policy is a pure
VRAM<->fetch knob; hit-rate raced vs LRU in gate G5b).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# P7: the K5 stats (sel/pinned_miss/staged_miss/dropped/lost_v) are pure
# telemetry for stats_line(); reading them per step is a D2H sync on the
# critical path (measured ~1.7 ms/step, one of the three service() syncs).
# Off by default; LOCKS_K5_STATS=1 (or LOCKS_TIER_ASSERT) re-enables the
# per-step readback for the hit-rate/lost_v report + the I1 latch check.
_K5_STATS = (os.environ.get("LOCKS_K5_STATS") not in (None, "", "0")
             or os.environ.get("LOCKS_TIER_ASSERT") not in (None, "", "0"))

_MCAP = 65536                 # mailbox / promote-list capacity (entries)
_EW = 6                       # entry words: (l, kv, blk, slot, gen, pad)
_NEVER = tl.constexpr(2147483647)   # never-matching CAS compare (scratch lane)

# K7 (HERO_PLAN sec 22, GO 2026-07-19): at bs=1 under PLAN the per-layer
# victim+install kernel (_k5b) leaves the critical path entirely.  _k5c
# (classify + DIRECT mailbox append) replaces the k5a+k5b pair per layer;
# victim assignment happens ONCE per step in _k5_assign_kernel on the
# K5Service transfer stream (between the mailbox snapshot and the elist
# copy), off the critical path.  Deferability is measured fact (doc 18):
# a pinned miss's same-step decode consumes nothing the claim produces.
# LOCKS_K5_V7=0 restores the legacy k5a+k5b schedule (the matched-pair ctl
# arm).
_K5_V7 = os.environ.get("LOCKS_K5_V7", "1") not in ("0",)

# K7B (bs>1 campaign phase 2, 2026-07-21): the K7 protocol EXTENDED to
# n_req > 1.  The K7 kernels are already batch-shaped -- _k5c's grid carries
# the request dimension (r = program_id(0)), the mailbox is global with
# request-free entries (l, kv, blk, slot, gen; blk is a physical block id),
# and _k5_assign's grid (L, n_kv) walks per-(layer, kv) slot pools that ALL
# requests share (memplan fix: S = attended-set pages x n_seqs) -- so the
# per-step assignment on the service stream covers every sequence at once.
# The historical "k5a/k5b claim split is load-bearing at n_req > 1" concern
# is structurally void here: the split's value was classify-before-probe
# ordering, and K7 defers the probe to the NEXT prologue, a whole-step
# barrier strictly stronger than the k5a->k5b grid order.  What k5b did at
# batch instead was the measured collapse: per-(request, kv) serial CLOCK
# probes against an exactly-sized pool (S/b = 1 under R2) burn ~S probes
# per miss -- 915.4 ms/step = 97.2% of busy at mem@bs32/32K (phase-1
# profile, results/batch32/prof_mem_bs32.json).  Residency policy only:
# every arm stays bitwise (misses read staged/pinned in place).
# bs=1 is BIT-INERT by construction: the k5_step gate keeps `n_req == 1`
# as an independent disjunct, so single-request dispatch is unchanged in
# every flag state.  Kill switch LOCKS_K5_V7B=0 = the legacy k5a+k5b batch
# path exactly (the phase-1 ctl arm).
_K5_V7B = os.environ.get("LOCKS_K5_V7B", "1") not in ("0",)
_V7B_ANNOUNCED = False        # one-shot engagement sentinel (n_req > 1 only)

# K7C (bs>1 campaign phase 3, 2026-07-21): batch-capacity service plane.
# The phase-3 K5_STATS diagnostics at mem bs8/32K (results/batch32/phase3/
# diag{2,4}*) measured the batch steady state: demand ~127K new misses/step
# against sel ~162K (hit 0.60 cumulative, ~0.22 steady) with assign
# succeeding on only ~4K/step and ~123K/step dropping; slot census: fresh
# ~49%, owned ~25%, free ~24%.  THREE capacity bugs compose into a
# self-sustaining starvation loop (a dropped page stays selected and
# re-misses every step, inflating demand ~3.6x over the true ~25%/step
# selection churn -- the bs=1 flap physics, re-measured at batch):
#   (1) the GLOBAL mailbox (MCAP=65536) fills in LAYER ORDER inside a step,
#       so at batch demand (up to R*128*L*n_kv = 164K..655K appends/step at
#       bs8..32) the high layers' misses NEVER reach the mailbox -- ~half
#       the units are permanently transport-starved;
#   (2) _k5_assign's ticket loop issues only n_need (~1-2) tickets per
#       vectorized iteration and gives up at ONE lap (spent < S) -- at
#       batch free densities (~0.24) serving the unit's rows needs ~4
#       tickets/row > the lap budget, and the per-(unit, chunk) scalar
#       loop is ALSO the 2.6-3.7 ms/step assign burn from the phase-2
#       profiles;
#   (3) install latency 2 steps (see PROMO1 below) holds assigned slots
#       owner-locked ~2x longer than needed.
# K7C (n_req > 1 only): MCAP scales to the worst-case per-step demand
# (next_pow2(R*L*n_kv*128), R=1 keeps 65536 exactly); the assign consumes
# PER-UNIT BUCKETS built by a grid-strided bucketize over the snapshot
# (fairness across layers by construction; bucket overflow takes the
# give-up path at source) and draws tickets BLOCK-wide per iteration
# (free slots compacted through a per-unit scratch row, rank-matched to
# the needy rows; budget 2 laps).  Residency policy only: every miss
# still reads staged/pinned in place (identical bytes), pool SIZE is
# untouched (R2), transport bytes are the same pages the deployed path
# would move once capacity allows.  bs=1 dispatch verbatim (WIDE=False
# compiles the historical loop; R==1 sizes = historical constants).
# Kill switch LOCKS_K5_K7C=0 = the phase-2 batch service plane exactly.
_K5_K7C = os.environ.get("LOCKS_K5_K7C", "1") not in ("0",)
_K7C_ANNOUNCED = False        # one-shot engagement sentinel (n_req > 1 only)


def _mcap_for(max_reqs: int, n_layers: int, n_kv: int) -> int:
    """Mailbox capacity: worst-case per-step pinned-miss demand (every
    selection missing) with R2-exact budgets = R * L * n_kv * 128 pages,
    rounded up to a power of two.  R == 1 keeps the historical 65536
    (bs=1 byte-inert); the K7C kill switch also pins it."""
    if max_reqs <= 1 or not _K5_K7C:
        return _MCAP
    need = max_reqs * n_layers * n_kv * 128
    cap = 65536
    while cap < need:
        cap *= 2
    return min(cap, 2 ** 21)


# PROMO1 (bs>1 campaign phase 3, 2026-07-21): 1-STEP INSTALL -- consume the
# oldest FENCED transport batch at the NEXT prologue (main stream, pre-graph)
# instead of the next post-graph service.  Deployed batch schedule was: miss
# at step t (k5c append) -> assign+transport at end-of-t service (tr stream)
# -> fence fires during t+1 -> PUBLISH at end-of-t+1 service -> install at
# t+2's graph head = TWO steps of pinned-UVA split reads per miss AND two
# steps of owner-held in-flight slots.  Measured at mem bs8/32K (K5_STATS
# diag, results/batch32/phase3/diag2_k5stats_bs8): sustained churn-in
# ~65K pages/step (mailbox at MCAP cap) with ~40K assigned + ~23K give-ups
# per step and hot fill stuck at 0.34 -- the 2-step latency x ~40K/step
# holds ~80K of the 164K slots owner-held-in-transit, starving the assign
# (drops re-miss next step, inflating demand ~2x over the true selection
# churn).  PROMO1 publishes + APPLIES the fenced batch at the true prologue
# (fence.query() -> evict launch -> promote launch, kernel boundary between
# the phases because the SAME batch appears in both lists -- the fused
# head's phase-disjointness premise does not hold for a same-batch apply),
# then zeros ecnt/pcnt so the captured in-graph head walks empty lists.
# Install latency 2 -> 1 step: pinned window halves, in-flight hold halves,
# give-up starvation drains.  Residency/transport timing only -- misses
# still read staged/pinned in place (identical bytes), R2 pool size
# untouched, R3 hide-don't-hit.  n_req > 1 GATED (bs=1 keeps the deployed
# schedule verbatim).  Kill switch LOCKS_K5_PROMO1=0.
_K5_PROMO1 = os.environ.get("LOCKS_K5_PROMO1", "1") not in ("0",)
_PROMO1_ANNOUNCED = False     # one-shot engagement sentinel (n_req > 1 only)

# K8 (2026-07-19): the _k5c control plane is absorbed into the mem decode
# split kernel (decode_mem.py) and the clock advances ONCE per step at the
# graph head instead of once per layer.  The assign recency window rescales
# with it: one step = 1 tick (was L ticks).  Pure eviction policy, bitwise-
# free by design.  Kill switch: LOCKS_MEM_K8=0.
_K8 = os.environ.get("LOCKS_MEM_K8", "0") == "1"

# Z1 (2026-07-19): per-unit mailbox buckets.  The hr12/hr6 localization
# pair (results/nsys_s5_hr{12,6}) attributed the offload tax's ctx-growing
# term to _k5_assign's full-mailbox scan (+0.0293 of +0.042 busy at
# dm~7/unit).  The snapshot copy becomes ONE CUDA kernel (k5_cuda.py
# mail_snap_bucketize: copy + clamp + per-unit bucket index build + give-up
# overflow at source), and assign reads ONLY its unit's bucket ->
# O(own misses) instead of O(L*n_kv*mail).  Residency-policy-only,
# bitwise-free.  Kill switch: LOCKS_K5_Z1=0.
_Z1 = os.environ.get("LOCKS_K5_Z1", "0") == "1"
_Z1_CAP = 2048               # bucket capacity per (layer, kv) unit
# E3 (mail diet): replace service()'s three FULL-buffer mailbox copies
# (_MCAP x ENTW words of ~empty buffers) with cnt-bounded device copies
# (k5_cuda.mail_bcopy).  Read-equal by construction: every consumer walks
# [0, cnt) only.  Kill switch: 0.  Composes with LOCKS_K5_EARLY (the
# copies then run clock-warm under the LM-head span).
_MAIL_DIET = os.environ.get("LOCKS_MAIL_DIET", "0") == "1"


def _bcopy(src, dst, cnt, entw: int = 6, cap: int | None = None):
    """cnt-bounded mailbox copy; falls back to the full copy_ if the CUDA
    module is unavailable (behavior-equal either way)."""
    if _MAIL_DIET:
        from .k5_cuda import get_mod
        mod = get_mod()
        if mod is not None:
            mod.mail_bcopy(src.view(-1), dst.view(-1), cnt,
                           (_MCAP if cap is None else int(cap)), entw)
            return
    dst.copy_(src, non_blocking=True)


@triton.jit
def _k5a_kernel(tab_ptr, cnt_ptr, bt_ptr, sl_ptr,
                p2s_ptr, epoch_ptr, clock_ptr,
                vbo_ptr, valid_ptr, scratch_ptr,
                mpage_ptr, msrc_ptr, mcnt_ptr, stats_ptr,
                NB, S, MP, stride_btr, stride_tr, stride_cr,
                PAGE: tl.constexpr, BLOCK: tl.constexpr,
                STATS: tl.constexpr):
    """Classify+claim (see module doc). Per-(request, kv-head) program;
    vectorized over the selected pages; the only cross-program touch points are
    page2slot (atomicCAS) and the hit-epoch stores (same-value racy stores are
    benign: every writer stores the same clock).

    tl.atomic_cas takes no mask: inactive lanes are redirected to a scratch
    word with a never-matching compare value (an atomic read, no store)."""
    r = tl.program_id(0)
    kv = tl.program_id(1)
    n = tl.load(cnt_ptr + r * stride_cr + kv)
    sl = tl.load(sl_ptr + r)
    clk = tl.load(clock_ptr)
    n_miss = clk * 0
    n_sel = clk * 0
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        valid = idx < n
        pt = tl.load(tab_ptr + r * stride_tr + kv * MP + idx, mask=valid,
                     other=0)
        blk = tl.load(bt_ptr + r * stride_btr + pt, mask=valid, other=0)
        complete = ((pt + 1) * PAGE) <= sl
        act = valid & complete
        n_sel += tl.sum(act.to(tl.int32), axis=0)
        slot = tl.load(p2s_ptr + kv * NB + blk, mask=valid, other=-1)
        is_hit = act & (slot >= 0)
        tl.store(epoch_ptr + kv * S + slot, clk, mask=is_hit)
        cand = act & (slot == -1)
        cas_ptr = tl.where(cand, p2s_ptr + kv * NB + blk, scratch_ptr)
        cas_cmp = tl.where(cand, -1, _NEVER)
        old = tl.atomic_cas(cas_ptr, cas_cmp,
                            tl.full([BLOCK], -2, tl.int32))
        won = cand & (old == -1)
        vb = tl.load(vbo_ptr + blk, mask=won, other=-1)
        vld = tl.load(valid_ptr + kv * NB + blk, mask=won, other=0)
        lost_v = won & (vb < 0) & (vld == 0)      # I1 violation upstream
        # revert the claim on lost pages (loud latch; page left unselectable
        # from hot -- the decode's pinned arm would read zeros, so never map it)
        tl.store(p2s_ptr + kv * NB + blk, tl.full([BLOCK], -1, tl.int32),
                 mask=lost_v)
        won = won & (lost_v == 0)
        m32 = won.to(tl.int32)
        mpos = tl.cumsum(m32, axis=0) - m32
        tl.store(mpage_ptr + r * stride_tr + kv * MP + n_miss + mpos, blk,
                 mask=won)
        tl.store(msrc_ptr + r * stride_tr + kv * MP + n_miss + mpos,
                 (vb < 0).to(tl.int8), mask=won)   # 0 staged | 1 pinned
        n_miss += tl.sum(m32, axis=0)
        if STATS:
            n_lost = tl.sum(lost_v.to(tl.int32), axis=0)
            if n_lost > 0:
                tl.atomic_add(stats_ptr + 4, n_lost)
    tl.store(mcnt_ptr + r * stride_cr + kv, n_miss)
    if STATS:
        tl.atomic_add(stats_ptr + 0, n_sel)


@triton.jit
def _k5c_kernel(tab_ptr, cnt_ptr, bt_ptr, sl_ptr,
                p2s_ptr, epoch_ptr, clock_ptr,
                vbo_ptr, valid_ptr, scratch_ptr,
                gen_ptr, mail_ptr, mail_cnt_ptr, ovf_ptr, stats_ptr,
                lidx, NB, S, MP, MCAP, stride_btr, stride_tr, stride_cr,
                PAGE: tl.constexpr, BLOCK: tl.constexpr,
                STATS: tl.constexpr):
    """K7 thin classify (bs=1, PLAN schedule): k5a's classify+claim with the
    PINNED misses appended DIRECTLY to the DMA mailbox (slot = -1, assigned
    later by _k5_assign_kernel at the next prologue) and STAGED misses
    reverted AT SOURCE (the PLAN semantics k5b implemented as claim->revert;
    staying readable staged in place is bitwise-free).  _k5b is not launched.

    Order preserved from k5a: CAS claim -> source classify -> lost_v revert
    (the I1 latch depends on claiming before classifying).  The mailbox index
    is reserved ONCE per chunk with a warp-aggregated relaxed scalar
    atomic_add (SASS census 22b: no membar/no CTA barrier), then the entries
    scatter at base+cumsum -- no per-miss scalar atomics remain.  Per-lane
    MCAP guard reverts overflow lanes and latches ovf (the eviction test's
    contract).  No miss list / victim scratch is written: the mailbox IS the
    miss list under K7."""
    r = tl.program_id(0)
    kv = tl.program_id(1)
    n = tl.load(cnt_ptr + r * stride_cr + kv)
    sl = tl.load(sl_ptr + r)
    clk = tl.load(clock_ptr)
    n_sel = clk * 0
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        valid = idx < n
        pt = tl.load(tab_ptr + r * stride_tr + kv * MP + idx, mask=valid,
                     other=0)
        blk = tl.load(bt_ptr + r * stride_btr + pt, mask=valid, other=0)
        complete = ((pt + 1) * PAGE) <= sl
        act = valid & complete
        n_sel += tl.sum(act.to(tl.int32), axis=0)
        slot = tl.load(p2s_ptr + kv * NB + blk, mask=valid, other=-1)
        is_hit = act & (slot >= 0)
        tl.store(epoch_ptr + kv * S + slot, clk, mask=is_hit)
        cand = act & (slot == -1)
        cas_ptr = tl.where(cand, p2s_ptr + kv * NB + blk, scratch_ptr)
        cas_cmp = tl.where(cand, -1, _NEVER)
        old = tl.atomic_cas(cas_ptr, cas_cmp,
                            tl.full([BLOCK], -2, tl.int32))
        won = cand & (old == -1)
        vb = tl.load(vbo_ptr + blk, mask=won, other=-1)
        vld = tl.load(valid_ptr + kv * NB + blk, mask=won, other=0)
        lost_v = won & (vb < 0) & (vld == 0)      # I1 violation upstream
        tl.store(p2s_ptr + kv * NB + blk, tl.full([BLOCK], -1, tl.int32),
                 mask=lost_v)
        won = won & (lost_v == 0)
        staged = won & (vb >= 0)
        # PLAN staged-skip at source (was k5b's claim->revert round trip):
        # the page stays readable staged in place, identical bytes.
        tl.store(p2s_ptr + kv * NB + blk, tl.full([BLOCK], -1, tl.int32),
                 mask=staged)
        pin = won & (vb < 0)
        m32 = pin.to(tl.int32)
        n_pin = tl.sum(m32, axis=0)
        # RELAXED batched reservation: one warp-aggregated scalar atomic per
        # chunk; only the index's atomicity matters (payload visibility is by
        # kernel boundary, as with k5b's mailbox arm).
        base = tl.atomic_add(mail_cnt_ptr, n_pin, sem="relaxed")
        pos = base + tl.cumsum(m32, axis=0) - m32
        ok = pin & (pos < MCAP)
        g = tl.load(gen_ptr + blk, mask=ok, other=0)
        tl.store(mail_ptr + pos * 6 + 0, tl.full([BLOCK], 0, tl.int32) + lidx,
                 mask=ok)
        tl.store(mail_ptr + pos * 6 + 1, tl.full([BLOCK], 0, tl.int32) + kv,
                 mask=ok)
        tl.store(mail_ptr + pos * 6 + 2, blk, mask=ok)
        tl.store(mail_ptr + pos * 6 + 3, tl.full([BLOCK], -1, tl.int32),
                 mask=ok)
        tl.store(mail_ptr + pos * 6 + 4, g, mask=ok)
        over = pin & (pos >= MCAP)
        n_over = tl.sum(over.to(tl.int32), axis=0)
        if n_over > 0:
            # overflow: not cached this step; revert (reads pinned in place).
            tl.store(p2s_ptr + kv * NB + blk,
                     tl.full([BLOCK], -1, tl.int32), mask=over)
            tl.store(ovf_ptr, 1)
        if STATS:
            n_lost = tl.sum(lost_v.to(tl.int32), axis=0)
            if n_lost > 0:
                tl.atomic_add(stats_ptr + 4, n_lost)
            tl.atomic_add(stats_ptr + 1, n_pin - n_over)  # appended misses
            tl.atomic_add(stats_ptr + 2, tl.sum(staged.to(tl.int32), axis=0))
            if n_over > 0:
                tl.atomic_add(stats_ptr + 3, n_over)
    if STATS:
        tl.atomic_add(stats_ptr + 0, n_sel)


@triton.jit
def _k5_assign_kernel(mail_ptr, mcnt_ptr,
                      p2s_ptr, epoch_ptr, owner_ptr, hand_ptr, clock_ptr,
                      scratch_ptr, ovf_ptr, stats_ptr,
                      bcnt_ptr, bidx_ptr, CAP, fscr_ptr,
                      NB, S, L,
                      NKV: tl.constexpr, BLOCK: tl.constexpr,
                      STATS: tl.constexpr, BUCKETED: tl.constexpr = False,
                      WIDE: tl.constexpr = False):
    """K7 per-step victim assignment (OFF the critical path: launched by
    K5Service on the transfer stream between the mailbox snapshot-clamp and
    the elist copy, so the graph-head evict and the transport both see the
    assigned slots).  Grid (L, NKV): slot pools, hand and epoch are
    per-(layer, kv-head), so each program owns its unit's ticket draw
    outright (red-team change 2: per-unit grouping is what makes tickets
    distinct).

    Per unit: scan the snapshot for rows (l, kv, slot == -1); draw CLOCK
    tickets from a register cursor (hand is read once and stored once -- the
    assign is the ONLY prober under K7, red-team A7); a slot is refusable if
    touched during the previous step (epoch > clock - L: every layer's step
    advances the shared clock once, so one full step is exactly L ticks) or
    its owner is held (in-flight transfer from an older batch -- those
    owners release only at a LATER graph-head promote, which zev-ordering
    puts after this kernel).  The claim itself is the VECTOR per-lane
    atomic_cas (k5a's masked-redirect pattern; one membar chain per
    instruction across the whole batch, sec 22b) -- atomicity is required
    because the promote's owner release races this kernel across streams.
    Total tickets per unit are bounded by S (one lap); leftover rows GIVE UP:
    page2slot reverts to -1 (concurrent readers see -2 or -1, both "not
    hot"), the row stays slot = -1 (DEAD -- every walker skips it) and the
    overflow latch fires.  Rows with slot >= 0 (legacy-k5b claims) are
    skipped, so mixed-protocol batches are safe by construction."""
    l = tl.program_id(0)
    kv = tl.program_id(1)
    n = tl.load(mcnt_ptr)
    clk = tl.load(clock_ptr)
    unit = l * NKV + kv
    tag = kv + 1
    cur = tl.load(hand_ptr + unit)
    spent = clk * 0
    n_asn = clk * 0
    if BUCKETED:
        # Z1: this unit's bucket only (built by mail_snap_bucketize) --
        # O(own misses) instead of the full-mailbox filter scan.
        n = tl.load(bcnt_ptr + unit)
    for off in range(0, n, BLOCK):
        pos = off + tl.arange(0, BLOCK)
        inb = pos < n
        if BUCKETED:
            idx = tl.load(bidx_ptr + unit * CAP + pos, mask=inb, other=0)
            es = tl.load(mail_ptr + idx * 6 + 3, mask=inb, other=0)
            need = inb & (es == -1)
        else:
            idx = pos
            el = tl.load(mail_ptr + idx * 6 + 0, mask=inb, other=-1)
            ek = tl.load(mail_ptr + idx * 6 + 1, mask=inb, other=-1)
            es = tl.load(mail_ptr + idx * 6 + 3, mask=inb, other=0)
            need = inb & (el == l) & (ek == kv) & (es == -1)
        n_need = tl.sum(need.to(tl.int32), axis=0)
        if WIDE:
            # K7C batch arm: probe a FULL BLOCK of consecutive tickets per
            # iteration (the historical loop issues only n_need -- at batch
            # free densities that starves the unit inside its lap budget and
            # burns hundreds of serial vector iterations).  Free slots are
            # compacted through this unit's scratch row and rank-matched to
            # the needy rows; budget = TWO laps of the hand (one lap SEES
            # every slot; the second covers slots freed mid-lap).  Policy
            # order preserved: tickets in hand order, rows in list order.
            while (n_need > 0) & (spent < 2 * S):
                lane = tl.arange(0, BLOCK)
                c = (cur + lane) % S
                ep = tl.load(epoch_ptr + unit * S + c)
                own = tl.load(owner_ptr + unit * S + c)
                isf = (ep <= clk - L) & (own == 0)
                f32 = isf.to(tl.int32)
                rkf = tl.cumsum(f32, axis=0) - f32
                nfree = tl.sum(f32, axis=0)
                tl.store(fscr_ptr + unit * BLOCK + rkf, c, mask=isf)
                tl.debug_barrier()
                m32 = need.to(tl.int32)
                rkn = tl.cumsum(m32, axis=0) - m32
                take = need & (rkn < nfree)
                cs = tl.load(fscr_ptr + unit * BLOCK + rkn, mask=take,
                             other=0)
                cas_ptr = tl.where(take, owner_ptr + unit * S + cs,
                                   scratch_ptr)
                cas_cmp = tl.where(take, 0, _NEVER)
                old = tl.atomic_cas(cas_ptr, cas_cmp,
                                    tl.zeros([BLOCK], tl.int32) + tag)
                won = take & (old == 0)
                tl.store(mail_ptr + idx * 6 + 3, cs, mask=won)
                tl.store(epoch_ptr + unit * S + cs, clk, mask=won)
                n_asn += tl.sum(won.to(tl.int32), axis=0)
                need = need & (won == 0)
                n_need = tl.sum(need.to(tl.int32), axis=0)
                cur += BLOCK
                spent += BLOCK
                tl.debug_barrier()      # fscr reuse across iterations
        else:
            while (n_need > 0) & (spent < S):
                m32 = need.to(tl.int32)
                rk = tl.cumsum(m32, axis=0) - m32
                issued = tl.minimum(n_need, S - spent)
                live = need & (rk < issued)
                c = (cur + rk) % S
                ep = tl.load(epoch_ptr + unit * S + c, mask=live, other=0)
                own = tl.load(owner_ptr + unit * S + c, mask=live, other=1)
                free = live & (ep <= clk - L) & (own == 0)
                cas_ptr = tl.where(free, owner_ptr + unit * S + c,
                                   scratch_ptr)
                cas_cmp = tl.where(free, 0, _NEVER)
                old = tl.atomic_cas(cas_ptr, cas_cmp,
                                    tl.zeros([BLOCK], tl.int32) + tag)
                won = free & (old == 0)
                tl.store(mail_ptr + idx * 6 + 3, c, mask=won)
                tl.store(epoch_ptr + unit * S + c, clk, mask=won)
                n_asn += tl.sum(won.to(tl.int32), axis=0)
                need = (live & (won == 0)) | (need & (rk >= issued))
                n_need = tl.sum(need.to(tl.int32), axis=0)
                cur += issued
                spent += issued
        n_left = tl.sum(need.to(tl.int32), axis=0)
        if n_left > 0:
            blkl = tl.load(mail_ptr + idx * 6 + 2, mask=need, other=0)
            tl.store(p2s_ptr + unit * NB + blkl,
                     tl.full([BLOCK], -1, tl.int32), mask=need)
            tl.store(ovf_ptr, 1)
            if STATS:
                tl.atomic_add(stats_ptr + 3, n_left)
    tl.store(hand_ptr + unit, cur)
    if STATS:
        tl.atomic_add(stats_ptr + 5, n_asn)   # assigned = will-be-transported


@triton.jit
def _k7c_bucketize_kernel(snap_ptr, scnt_ptr, p2s_ptr,
                          bcnt_ptr, bidx_ptr, ovf_ptr,
                          CAP, NB,
                          NKV: tl.constexpr, BLOCK: tl.constexpr,
                          GRID: tl.constexpr):
    """K7C: per-unit bucket index build over the SNAPSHOT (grid-strided,
    multi-CTA -- the Z1 single-block CUDA bucketize serialized at batch
    mailbox sizes).  PENDING rows (slot == -1) append their snapshot index
    to bucket (l, kv); bucket overflow takes the assign give-up path at
    source (revert the p2s claim -> the page reads pinned in place,
    bitwise-free; ovf latch).  Bucket order is atomic-append order --
    residency policy only.  Caller zeroes bcnt first (same stream)."""
    pid = tl.program_id(0)
    n = tl.load(scnt_ptr)
    one = tl.full([BLOCK], 1, tl.int32)
    for i0 in range(pid * BLOCK, n, GRID * BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        inb = idx < n
        el = tl.load(snap_ptr + idx * 6 + 0, mask=inb, other=0)
        ek = tl.load(snap_ptr + idx * 6 + 1, mask=inb, other=0)
        es = tl.load(snap_ptr + idx * 6 + 3, mask=inb, other=0)
        pend = inb & (es == -1)
        unit = el * NKV + ek
        pos = tl.atomic_add(bcnt_ptr + unit, one, mask=pend)
        ok = pend & (pos < CAP)
        tl.store(bidx_ptr + unit * CAP + pos, idx, mask=ok)
        over = pend & (pos >= CAP)
        n_over = tl.sum(over.to(tl.int32), axis=0)
        if n_over > 0:
            blk = tl.load(snap_ptr + idx * 6 + 2, mask=over, other=0)
            tl.store(p2s_ptr + unit * NB + blk,
                     tl.full([BLOCK], -1, tl.int32), mask=over)
            tl.store(ovf_ptr, 1)


@triton.jit
def _k5b_kernel(mpage_ptr, msrc_ptr, mcnt_ptr,
                p2s_ptr, s2p_ptr, epoch_ptr, owner_ptr, hand_ptr, clock_ptr,
                gen_ptr, mail_ptr, mail_cnt_ptr, ovf_ptr, stats_ptr,
                hot_ptr, hotk_ptr, vp_ptr, kp_ptr, vbo_ptr,
                poolv_ptr, poolk_ptr,
                lidx, NB, S, MP, MCAP, stride_tr, stride_cr,
                svb, svt, svh, NKV: tl.constexpr, W: tl.constexpr,
                PAGE: tl.constexpr, D: tl.constexpr, HAS_K: tl.constexpr,
                EAGER_PIN: tl.constexpr, PLAN: tl.constexpr,
                STATS: tl.constexpr):
    """Victim+install (see module doc). Grid (n_req, n_kv, W): W worker
    programs per unit stride the unit's miss list -- every miss claims its own
    slot through the owner CAS, so misses are independent and the walk is
    parallel (the first cut walked each unit serially from one program:
    95.6 us/layer mean at 16K bs1, 1.23 ms cold). Shared cells -- hand, owner,
    epoch, the mailbox counter -- are atomics/claim-guarded. No cross-CTA
    sync anywhere."""
    r = tl.program_id(0)
    kv = tl.program_id(1)
    w = tl.program_id(2)
    m = tl.load(mcnt_ptr + r * stride_cr + kv)
    clk = tl.load(clock_ptr)
    tag = r * NKV + kv + 1
    offs_t = tl.arange(0, PAGE)
    offs_d = tl.arange(0, D)
    tile = offs_t[:, None] * D + offs_d[None, :]
    for i in range(w, m, W):
        blk = tl.load(mpage_ptr + r * stride_tr + kv * MP + i).to(tl.int64)
        src = tl.load(msrc_ptr + r * stride_tr + kv * MP + i)
        # ---- bounded CLOCK probe + claim-then-verify ----------------------- #
        # Bound = one lap + margin: a second full lap can only succeed if a
        # slot freed mid-lap (rare); when demand exceeds capacity the give-up
        # (revert to in-place pinned reads, bitwise-free) must be FAST -- at
        # 2S the exhaustion path burned 4096 atomic probes per miss and one
        # capacity-thrashed b8 corner measured 23.8 ms/layer in K5b.
        h = clk * 0 - 1
        t = clk * 0
        # P14 probe-storm guard: zero probe budget past the slot count; the
        # miss then takes the exhaustion path below (revert -> pinned read,
        # identical bytes).  See module doc; measured 1.53 s/layer k5b at
        # 128K-sync capacity thrash without this.
        plim = tl.where(i < S, S + 64, 0)
        while (h < 0) & (t < plim):
            # RELAXED: the hand is a pure ticket dispenser -- its value only
            # NOMINATES a candidate slot; the owner CAS below is what confers
            # the claim, so no other memory operation may be ordered against
            # this one.  Relaxed drops the MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR
            # + CCTL.IVALL that STRONG.GPU lowering emits per atomic (the
            # long_sb 33.61% / membar 7.61% chain in the 2026-07-18 census).
            c = tl.atomic_add(hand_ptr + kv, 1, sem="relaxed") % S
            ep = tl.load(epoch_ptr + kv * S + c)
            if ep != clk:
                prev = tl.atomic_cas(owner_ptr + kv * S + c, 0, tag)
                if prev == 0:
                    ep2 = tl.load(epoch_ptr + kv * S + c)
                    if ep2 == clk:
                        tl.store(owner_ptr + kv * S + c, 0)   # raced: release
                    else:
                        h = c
            t += 1
        if h < 0:
            tl.store(ovf_ptr, 1)
            tl.store(p2s_ptr + kv * NB + blk, -1)   # revert PENDING
            if STATS:
                tl.atomic_add(stats_ptr + 3, 1)
        else:
            # P15 PLAN mode: the claim marks ONLY k5-private state (owner via
            # the CAS above, epoch for intra-step dedupe).  The image mutation
            # (evict p2s[old], reserve s2p[h]) moves to the GRAPH-HEAD evict
            # kernel -- a concurrent main-stream decode can therefore never
            # observe a mid-step eviction or a mid-write slot.
            if not PLAN:
                old = tl.load(s2p_ptr + kv * S + h)
                if old >= 0:
                    tl.store(p2s_ptr + kv * NB + old, -1)
                tl.store(s2p_ptr + kv * S + h, blk)
            tl.store(epoch_ptr + kv * S + h, clk)
            if src == 0 and PLAN:
                # staged miss under PLAN: revert (stays readable staged/pinned
                # in place -- identical bytes; re-missable after flush).  The
                # staged in-kernel copy is exactly the torn-write hazard the
                # plan protocol exists to remove.
                tl.store(p2s_ptr + kv * NB + blk, -1)
                tl.store(owner_ptr + kv * S + h, 0)
                if STATS:
                    tl.atomic_add(stats_ptr + 3, 1)
            elif src == 0:
                # STAGED: in-kernel VRAM->VRAM page copy, publish, release.
                vb = tl.load(vbo_ptr + blk).to(tl.int64)
                stg = vb * svb + kv * svh + offs_t[:, None] * svt \
                    + offs_d[None, :]
                hot_off = (kv * S + h).to(tl.int64) * (PAGE * D) + tile
                tl.store(hot_ptr + hot_off,
                         tl.load(vp_ptr + stg).to(tl.int16, bitcast=True))
                if HAS_K:
                    tl.store(hotk_ptr + hot_off,
                             tl.load(kp_ptr + stg).to(tl.int16, bitcast=True))
                tl.store(p2s_ptr + kv * NB + blk, h)
                tl.store(owner_ptr + kv * S + h, 0)
                if STATS:
                    tl.atomic_add(stats_ptr + 2, 1)
            else:
                if EAGER_PIN:
                    # PINNED, LAYER-AHEAD (P10): copy the host page -> hot IN
                    # KERNEL and install page2slot immediately, mirroring the
                    # STAGED arm above (identical int16 bytes to what the decode
                    # would otherwise read in place from the pinned pool).  This
                    # runs on the lookahead SIDE stream a layer early and is
                    # serialized before ev_sel[L]; the layer-L decode waits that
                    # event and reads the page hot, so there is NO PENDING /
                    # mailbox / fenced-promote round trip (that 2-step latency
                    # existed only because the copy-engine transport ran on an
                    # unwaited stream).  page2slot may flip now because the copy
                    # completed on this same stream before the event.
                    pin_off = (((lidx * NB + blk) * NKV + kv) * (PAGE * D)
                               + tile)
                    hot_off = (kv * S + h).to(tl.int64) * (PAGE * D) + tile
                    tl.store(hot_ptr + hot_off, tl.load(poolv_ptr + pin_off))
                    if HAS_K:
                        tl.store(hotk_ptr + hot_off,
                                 tl.load(poolk_ptr + pin_off))
                    tl.store(p2s_ptr + kv * NB + blk, h)
                    tl.store(owner_ptr + kv * S + h, 0)
                    if STATS:
                        tl.atomic_add(stats_ptr + 1, 1)
                else:
                    # PINNED: mailbox for the copy engine; PENDING + owner held.
                    # RELAXED: this reserves a mailbox INDEX; only its
                    # atomicity matters.  The payload stores below are made
                    # visible to the transport by the KERNEL BOUNDARY (the
                    # transport is a separate launch at the next prologue),
                    # never by this atomic's ordering, and nothing reads
                    # mail[] inside this kernel.
                    mi = tl.atomic_add(mail_cnt_ptr, 1, sem="relaxed")
                    if mi < MCAP:
                        g = tl.load(gen_ptr + blk)
                        tl.store(mail_ptr + mi * 6 + 0, lidx)
                        tl.store(mail_ptr + mi * 6 + 1, kv)
                        tl.store(mail_ptr + mi * 6 + 2, blk.to(tl.int32))
                        tl.store(mail_ptr + mi * 6 + 3, h)
                        tl.store(mail_ptr + mi * 6 + 4, g)
                        if STATS:
                            # RELAXED: pure counter, read only after the
                            # kernel completes (host D2H under LOCKS_K5_STATS).
                            tl.atomic_add(stats_ptr + 1, 1, sem="relaxed")
                            # legacy claims transport at append time, so the
                            # assigned counter (stats[5]) tracks it here too:
                            # assigned == transported holds under EVERY
                            # protocol/flag combination (review D1).
                            tl.atomic_add(stats_ptr + 5, 1, sem="relaxed")
                    else:
                        # overflow: not cached this step; revert + release.
                        # Under PLAN, s2p[h] was never written (it still
                        # holds the OLD page's record until the graph-head
                        # evict) -- clobbering it would leave the old page's
                        # p2s dangling at h.  Only the legacy in-stream claim
                        # needs the s2p revert.
                        tl.store(ovf_ptr, 1)
                        tl.store(p2s_ptr + kv * NB + blk, -1)
                        if not PLAN:
                            tl.store(s2p_ptr + kv * S + h, -1)
                        tl.store(owner_ptr + kv * S + h, 0)
                        if STATS:
                            tl.atomic_add(stats_ptr + 3, 1)


@triton.jit
def _k5_evict_kernel(elist_ptr, ecnt_ptr,
                     p2s_ptr, s2p_ptr,
                     NB, S, NKV: tl.constexpr, GRID: tl.constexpr):
    """P15 graph-head phase-1: EVICT the victims of the batch snapshotted at
    this step's prologue, BEFORE any decode of the step reads page2slot.  The
    transport (kicked at the same prologue, off-graph stream) then writes the
    slots' bytes with no live mapping pointing at them; phase-2 (the existing
    install promote) publishes them once the transfer fence has passed, next
    step.  Entries are unique in h (owner CAS at plan time) -> parallel."""
    pid = tl.program_id(0)
    n = tl.load(ecnt_ptr)
    for i in range(pid, n, GRID):
        h = tl.load(elist_ptr + i * 6 + 3)
        if h >= 0:      # K7: skip DEAD rows (red-team change 1)
            l = tl.load(elist_ptr + i * 6 + 0)
            kv = tl.load(elist_ptr + i * 6 + 1)
            blk = tl.load(elist_ptr + i * 6 + 2)
            base = (l * NKV + kv).to(tl.int64)
            old = tl.load(s2p_ptr + base * S + h)
            if old >= 0:
                if old != blk:
                    # only unmap if the old page still maps to THIS slot
                    tl.atomic_cas(p2s_ptr + base * NB + old, h, -1)
            tl.store(s2p_ptr + base * S + h, blk)


def k5_evict(res, k5) -> None:
    """Graph-head phase-1 launch (P15 side schedule; call before k5_promote)."""
    _k5_evict_kernel[(_PGRID,)](
        k5.elist_dev, k5.ecnt_dev,
        res.page2slot, res.slot2page,
        res.NB, res.S, NKV=res.n_kv, GRID=_PGRID, num_warps=1)


@triton.jit
def _k5_promote_kernel(plist_ptr, pcnt_ptr,
                       p2s_ptr, s2p_ptr, owner_ptr, gen_ptr,
                       NB, S, NKV: tl.constexpr, GRID: tl.constexpr):
    """Graph-head promote: install fenced copy-engine pages (see module doc).
    Grid-stride over the (device-resident, host-refreshed) promote list;
    entries are independent (distinct (l, kv, blk, slot)), so the walk is
    embarrassingly parallel. The first cut read the PINNED list serially from
    ONE program: 2.59 ms/step mean at 16K bs1 (19.4 ms cold) -- the serial
    UVA load chain. The list is now H2D-copied into a device buffer by the
    prologue (stream-ordered before the replay) and walked by GRID programs."""
    pid = tl.program_id(0)
    n = tl.load(pcnt_ptr)
    for i in range(pid, n, GRID):
        h = tl.load(plist_ptr + i * 6 + 3)
        if h >= 0:      # K7: DEAD rows hold no slot, no owner -- skip
            l = tl.load(plist_ptr + i * 6 + 0)
            kv = tl.load(plist_ptr + i * 6 + 1)
            blk = tl.load(plist_ptr + i * 6 + 2)
            g = tl.load(plist_ptr + i * 6 + 4)
            base = (l * NKV + kv).to(tl.int64)
            cur_g = tl.load(gen_ptr + blk)
            installed = 0
            if cur_g == g:
                old = tl.atomic_cas(p2s_ptr + base * NB + blk, -2, h)
                if old == -2:
                    installed = 1
            if installed == 0:
                # invalidated / re-installed since issue: abandon the slot.
                tl.store(s2p_ptr + base * S + h, -1)
            tl.store(owner_ptr + base * S + h, 0)


@triton.jit
def _k5_headv_kernel(elist_ptr, ecnt_ptr, plist_ptr, pcnt_ptr,
                     p2s_ptr, s2p_ptr, owner_ptr, gen_ptr,
                     scratch_ptr, clock_ptr, fetch_ptr,
                     NB, S, NKV: tl.constexpr, GRID: tl.constexpr,
                     BLOCK: tl.constexpr, TICK: tl.constexpr):
    """K10 (2026-07-19): the graph-head evict + promote walks, VECTORIZED and
    FUSED into one launch (true-exposure profile: the two scalar walkers cost
    27-64us EACH per step on the in-graph critical path -- 4-6 dependent
    scalar loads per entry per program).  Same ops, same masks, BLOCK-wide:
    entry fields load as vectors, claims use the k5a masked-CAS redirect (the
    cheap per-lane predicated lowering, SASS census 22b.3).  The two phases
    are disjoint by protocol -- evict slots hold this prologue's owners,
    promote slots the fenced batch's owners (owner CAS at assign); evict
    CASes p2s[old] for RESIDENT old pages, promote CASes p2s[blk] for
    PENDING (-2) pages, and one (kv, blk) cell cannot be both -- so no
    barrier is needed between them.  TICK folds the K8 once-per-step clock
    advance (bitwise-free policy state).  Residency policy only: no output
    bytes depend on any of this within a step (readers treat -1/-2 alike;
    hot bytes install strictly before their mapping publishes)."""
    pid = tl.program_id(0)
    if TICK:
        if pid == 0:
            c = tl.load(clock_ptr)
            tl.store(clock_ptr, c + 1)
            tl.store(fetch_ptr, 0)
    en = tl.load(ecnt_ptr)
    pn = tl.load(pcnt_ptr)
    nv = _NEVER
    # ---- phase 1: evict (this prologue's snapshotted batch) --------------- #
    for i0 in range(pid * BLOCK, en, GRID * BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        inb = idx < en
        h = tl.load(elist_ptr + idx * 6 + 3, mask=inb, other=-1)
        act = inb & (h >= 0)                 # skip DEAD rows
        l = tl.load(elist_ptr + idx * 6 + 0, mask=act, other=0)
        kv = tl.load(elist_ptr + idx * 6 + 1, mask=act, other=0)
        blk = tl.load(elist_ptr + idx * 6 + 2, mask=act, other=0)
        base = (l * NKV + kv).to(tl.int64)
        old = tl.load(s2p_ptr + base * S + h, mask=act, other=-1)
        # unmap the old page iff it still maps to THIS slot
        um = act & (old >= 0) & (old != blk)
        cas_ptr = tl.where(um, p2s_ptr + base * NB + old, scratch_ptr)
        cas_cmp = tl.where(um, h, nv)
        tl.atomic_cas(cas_ptr, cas_cmp, tl.full([BLOCK], -1, tl.int32))
        tl.store(s2p_ptr + base * S + h, blk, mask=act)
    # ---- phase 2: promote (the fenced batch) ------------------------------ #
    for i0 in range(pid * BLOCK, pn, GRID * BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        inb = idx < pn
        h = tl.load(plist_ptr + idx * 6 + 3, mask=inb, other=-1)
        act = inb & (h >= 0)
        l = tl.load(plist_ptr + idx * 6 + 0, mask=act, other=0)
        kv = tl.load(plist_ptr + idx * 6 + 1, mask=act, other=0)
        blk = tl.load(plist_ptr + idx * 6 + 2, mask=act, other=0)
        g = tl.load(plist_ptr + idx * 6 + 4, mask=act, other=0)
        base = (l * NKV + kv).to(tl.int64)
        cur_g = tl.load(gen_ptr + blk, mask=act, other=-1)
        fresh = act & (cur_g == g)
        cas_ptr = tl.where(fresh, p2s_ptr + base * NB + blk, scratch_ptr)
        cas_cmp = tl.where(fresh, -2, nv)
        old = tl.atomic_cas(cas_ptr, cas_cmp,
                            tl.where(fresh, h, tl.full([BLOCK], -1,
                                                       tl.int32)))
        installed = fresh & (old == -2)
        aband = act & (installed == 0)
        tl.store(s2p_ptr + base * S + h, tl.full([BLOCK], -1, tl.int32),
                 mask=aband)
        tl.store(owner_ptr + base * S + h, tl.zeros([BLOCK], tl.int32),
                 mask=act)


_HEADV = os.environ.get("LOCKS_K5_HEADV", "0") == "1"
_HV_GRID = 16
_HV_BLOCK = 128


def k5_headv(res, k5, tick: bool) -> None:
    """K10 fused graph-head launch (replaces k5_evict + k5_promote [+ the K8
    once-per-step _step_setup when ``tick``]).  Hand-CUDA (k5_cuda.py, the
    low-level port -- plain device atomics, no Triton lowering tax); the
    Triton twin below is the build-failure fallback only."""
    from .k5_cuda import get_mod
    mod = get_mod()
    if mod is not None:
        mod.k5_headv(k5.elist_dev, k5.ecnt_dev, k5.plist_dev, k5.pcnt_dev,
                     res.page2slot, res.slot2page, k5.owner, k5.gen,
                     res.clock, res.fetch_cnt,
                     res.NB, res.S, res.n_kv, bool(tick))
        return
    _k5_headv_kernel[(_HV_GRID,)](
        k5.elist_dev, k5.ecnt_dev, k5.plist_dev, k5.pcnt_dev,
        res.page2slot, res.slot2page, k5.owner, k5.gen,
        k5.scratch, res.clock, res.fetch_cnt,
        res.NB, res.S, NKV=res.n_kv, GRID=_HV_GRID, BLOCK=_HV_BLOCK,
        TICK=bool(tick), num_warps=1)


class K5State:
    """Device+pinned state for the K5 protocol (hangs off the Tier)."""

    def __init__(self, residency, device):
        r = residency
        i32 = torch.int32
        dev = torch.device(device)
        # owner / hand / gen live on the Residency (the invariant checks and
        # invalidate need them); aliased here for the kernel launches.
        self.owner = r.owner
        self.hand = r.hand
        self.gen = r.gen
        # miss source classification parallel to residency.miss_pages.
        self.msrc = torch.zeros((r.R, r.n_kv, r.MP), dtype=torch.int8,
                                device=dev)
        # K7C: capacity scales with the batch's worst-case per-step demand
        # (R == 1 -> the historical _MCAP verbatim).
        L = r.page2slot.shape[0]
        self.mcap = _mcap_for(r.R, L, r.n_kv)
        # DMA mailbox (device; drained by the host each prologue).
        self.mail = torch.zeros((self.mcap, _EW), dtype=i32, device=dev)
        self.mail_cnt = torch.zeros((1,), dtype=i32, device=dev)
        # P15 side schedule: graph-head phase-1 evict list = the batch
        # snapshotted THIS prologue (published by K5Service; zero when none).
        self.elist_dev = torch.zeros((self.mcap, _EW), dtype=i32, device=dev)
        self.ecnt_dev = torch.zeros((1,), dtype=i32, device=dev)
        # scratch word for inactive-lane CAS redirection (K5a).
        self.scratch = torch.zeros((1,), dtype=i32, device=dev)
        # PROMO1: permanent zero count -- passed as the OTHER phase's count to
        # k5_headv so one launch runs evict-only / promote-only (the fused
        # kernel reads counts from device memory; a zero count is a no-op).
        self.zcnt = torch.zeros((1,), dtype=i32, device=dev)
        # stats [sel_complete, pinned_miss, staged_miss, dropped, lost_v,
        # assigned] i32, reset each prologue (host accumulates).  assigned
        # (index 5) counts entries that WILL be transported: K7's assign
        # successes plus legacy k5b's mailbox appends -- the ce_bytes
        # identity's counter under every protocol.
        self.stats = torch.zeros((8,), dtype=i32, device=dev)
        # promote list: host-side staging (pinned) + the DEVICE buffer the
        # graph-head promote reads (H2D-refreshed by the prologue, address
        # stable across replays; device reads, never UVA).
        self.plist_host = torch.zeros((self.mcap, _EW), dtype=i32,
                                      pin_memory=True)
        self.plist_dev = torch.zeros((self.mcap, _EW), dtype=i32, device=dev)
        self.pcnt_dev = torch.zeros((1,), dtype=i32, device=dev)
        # P7: double-buffered DEVICE mailbox snapshots -- the transport kernel
        # reads k5.mail in place, and each snapshot feeds the graph-head promote
        # ONE step later (device->device publish, no host round-trip).
        self.snap_mail = [torch.zeros((self.mcap, _EW), dtype=i32, device=dev)
                          for _ in range(2)]
        self.snap_cnt = [torch.zeros((1,), dtype=i32, device=dev)
                         for _ in range(2)]
        # Z1/K7C per-unit bucket index lists (bucketize -> assign).  K7C
        # sizes the per-unit cap to the worst case (one unit's misses <=
        # R * 128); Z1's historical 2048 is the floor (R == 1 unchanged).
        self.bkt_cap = max(_Z1_CAP, 128 * r.R) if _K5_K7C else _Z1_CAP
        self.bkt_cnt = torch.zeros((L * r.n_kv,), dtype=i32, device=dev)
        self.bkt_idx = torch.zeros((L * r.n_kv * self.bkt_cap,), dtype=i32,
                                   device=dev)
        # K7C WIDE-draw scratch: one BLOCK-wide row per unit for the free-
        # slot compaction (assign kernel, WIDE arm).
        self.fscr = torch.zeros((L * r.n_kv * 256,), dtype=i32, device=dev)
        self.bytes_gib = (self.owner.numel() * 4 + self.hand.numel() * 4
                          + self.gen.numel() * 4 + self.msrc.numel()
                          + self.mail.numel() * 4 + self.stats.numel() * 4
                          ) / 2**30


_PGRID = 64          # promote walk parallelism (independent entries)
_K5B_W = 8           # K5b worker programs per (request, kv-head) unit


def k5_promote(res, k5) -> None:
    """Graph-head promote launch (call once per step, before layer 0's K5a)."""
    _k5_promote_kernel[(_PGRID,)](
        k5.plist_dev, k5.pcnt_dev,
        res.page2slot, res.slot2page, k5.owner, k5.gen,
        res.NB, res.S, NKV=res.n_kv, GRID=_PGRID, num_warps=1)


def k5_step(res, k5, staging, lidx, page_table, page_cnt, block_table,
            seq_lens, n_req, poolv=None, poolk=None,
            eager_pin: bool = False, plan: bool = False) -> None:
    """One layer's K5a + K5b (grid-ordered launches; the setup advances the
    clock exactly like the old chain: once per (layer, step)).

    ``eager_pin`` (P10): the caller is the lookahead SIDE fork running a layer
    early, so K5b copies PINNED misses host->hot in kernel and installs
    page2slot NOW (see the EAGER_PIN arm) instead of posting them to the copy-
    engine mailbox; ``poolv``/``poolk`` are the pinned pool dev views the copy
    reads.  Off the lookahead path ``eager_pin`` is False and the mailbox +
    fenced-promote transport is unchanged."""
    from .residency import _step_setup_kernel
    n_kv, S, NB, MP = res.n_kv, res.S, res.NB, res.MP
    _step_setup_kernel[(1,)](res.clock, res.fetch_cnt, num_warps=1)
    if plan and _K5_V7 and not eager_pin and (n_req == 1 or _K5_V7B):
        if n_req > 1:
            global _V7B_ANNOUNCED
            if not _V7B_ANNOUNCED:
                _V7B_ANNOUNCED = True
                print(f"[locks] K7B ACTIVE: batch k5c protocol "
                      f"(n_req={n_req}, per-step assign on the service "
                      f"stream)", flush=True)
        # K7 protocol (PLAN): thin classify with direct mailbox append; NO
        # _k5b launch -- victim assignment happens once per step in
        # _k5_assign_kernel (K5Service prologue, transfer stream).  K7B
        # (phase 2) extends the dispatch to n_req > 1: the kernel grid
        # already carries the request dimension and the page2slot CAS gives
        # cross-request dedup exactly as k5a did; the classify-before-probe
        # order the k5a/k5b split enforced holds a fortiori (the probe is a
        # whole step later).  The `n_req == 1` disjunct keeps bs=1 dispatch
        # byte-identical in every flag state (bit-inert gate extension).
        _k5c_kernel[(n_req, n_kv)](
            page_table, page_cnt, block_table, seq_lens,
            res.page2slot[lidx], res.lru_clock[lidx], res.clock,
            staging.vbo, res.pool_valid[lidx], k5.scratch,
            k5.gen, k5.mail, k5.mail_cnt, res.overflow, k5.stats,
            lidx, NB, S, MP, getattr(k5, 'mcap', _MCAP), block_table.stride(0),
            page_table.stride(0), page_cnt.stride(0),
            PAGE=res.page, BLOCK=256, STATS=_K5_STATS, num_warps=4)
        return
    _k5a_kernel[(n_req, n_kv)](
        page_table, page_cnt, block_table, seq_lens,
        res.page2slot[lidx], res.lru_clock[lidx], res.clock,
        staging.vbo, res.pool_valid[lidx], k5.scratch,
        res.miss_pages, k5.msrc, res.miss_cnt, k5.stats,
        NB, S, MP, block_table.stride(0), page_table.stride(0),
        page_cnt.stride(0), PAGE=res.page, BLOCK=256, STATS=_K5_STATS,
        num_warps=4)
    vp = staging.v_pool[lidx]
    has_k = res.hot_k_i16 is not None
    kp = staging.k_pool[lidx] if has_k else vp
    hotk = res.hot_k_i16[lidx] if has_k else res.hot_i16[lidx]
    # pinned-pool dev views for the EAGER_PIN copy; dead pointers (never
    # dereferenced) under EAGER_PIN=0 -- pass the staging views as placeholders.
    pv = poolv if poolv is not None else vp
    pk = poolk if poolk is not None else (kp if has_k else vp)
    _k5b_kernel[(n_req, n_kv, _K5B_W)](
        res.miss_pages, k5.msrc, res.miss_cnt,
        res.page2slot[lidx], res.slot2page[lidx], res.lru_clock[lidx],
        k5.owner[lidx], k5.hand[lidx], res.clock,
        k5.gen, k5.mail, k5.mail_cnt, res.overflow, k5.stats,
        res.hot_i16[lidx], hotk, vp, kp, staging.vbo,
        pv, pk,
        lidx, NB, S, MP, getattr(k5, 'mcap', _MCAP), res.miss_pages.stride(0),
        res.miss_cnt.stride(0),
        vp.stride(0), vp.stride(1), vp.stride(2), NKV=n_kv, W=_K5B_W,
        PAGE=res.page, D=res.d, HAS_K=has_k, EAGER_PIN=bool(eager_pin),
        PLAN=bool(plan), STATS=_K5_STATS,
        num_warps=4)


_TR_GRID = 4096         # transport kernel grid-stride programs


@triton.jit
def _k5_transport_kernel(mail_ptr, mcnt_ptr, poolv_ptr, poolk_ptr,
                         hotv_ptr, hotk_ptr, NB, S, stride_hl,
                         GRID: tl.constexpr, NKV: tl.constexpr,
                         PAGE: tl.constexpr, D: tl.constexpr,
                         HAS_K: tl.constexpr):
    """Device-side pinned -> hot transport (P7): each program strides the mailbox
    and copies one page ``pool[l, blk, kv]`` (host DRAM, UVA int16) -> ``hot[l]
    [kv, slot]`` (VRAM int16).  Pure bit copies, byte-identical to the old
    copy-engine 2D transfer.  Reads mail_cnt ON DEVICE (guarded grid-stride), so
    the host issues NOTHING: no ``.cpu()`` sync, no numpy coalesce, no per-run
    ctypes ``cudaMemcpy2DAsync``.  page2slot is NOT touched here -- the graph-head
    promote installs it next step (identical mailbox -> transport -> promote 2-step
    semantics to the CE path), so residency stays bitwise-free."""
    pid = tl.program_id(0)
    n = tl.load(mcnt_ptr)
    offs_t = tl.arange(0, PAGE)
    offs_d = tl.arange(0, D)
    tile = offs_t[:, None] * D + offs_d[None, :]
    for i in range(pid, n, GRID):
        s32 = tl.load(mail_ptr + i * 6 + 3)
        # K7: DEAD rows (assign give-up, slot -1) are skipped -- an unguarded
        # -1 would alias the NEIGHBORING kv slice's hot bytes (in-bounds
        # corruption, not a fault).  Red-team change 1: load-bearing.
        if s32 >= 0:
            l = tl.load(mail_ptr + i * 6 + 0).to(tl.int64)
            kv = tl.load(mail_ptr + i * 6 + 1).to(tl.int64)
            blk = tl.load(mail_ptr + i * 6 + 2).to(tl.int64)
            slot = s32.to(tl.int64)
            page_off = (((l * NB + blk) * NKV + kv) * (PAGE * D)) + tile
            hot_off = l * stride_hl + (kv * S + slot) * (PAGE * D) + tile
            tl.store(hotv_ptr + hot_off, tl.load(poolv_ptr + page_off))
            if HAS_K:
                tl.store(hotk_ptr + hot_off, tl.load(poolk_ptr + page_off))


class K5Service:
    """Host side of the transport split (P7 device-driven rewrite): a single
    off-graph Triton kernel moves the PINNED misses to hot, reading the device
    mailbox in place.  The host issues one launch + a few event records per step
    and NEVER synchronizes on the device -- so the whole call pipelines under the
    GPU like the fast builder's prologue (was: three ``.cpu()`` D2H syncs + a
    per-entry numpy/ctypes copy-engine coalesce loop, ~13 ms/step host tax at
    16K bs1).

    Semantics UNCHANGED vs the copy-engine path: a pinned miss is left PENDING by
    K5b, its bytes are transported here on the transfer stream, and the graph-head
    promote installs page2slot next step once the transfer fence has passed.  The
    decode reads the pinned page in place until then (bitwise-free).

    Runs in the tier's begin_step (host prologue, outside the graph). Relies on
    vLLM's per-step host sync for mailbox coherence: by the time the prologue
    runs, the previous step's kernels have completed.
    """

    def __init__(self, tier, k5):
        self._tier = tier
        self._k5 = k5
        self._stream = torch.cuda.Stream(device=tier.device)
        self._inflight = []       # list[(fence_event, snap_buf_idx)]
        self._par = 0             # double-buffer parity for the promote snapshot
        # PROMO1 guard: service() published a batch to plist for the NEXT
        # in-graph head -> the prologue consume must not clobber pcnt that
        # step (it skips once and clears the flag; the head applies).
        self._pub_pending = False
        # PROMO1: which snapshot batch the device elist currently holds
        # (set at issue; None once its evict has been applied/zeroed).  In
        # the 2-in-flight transient the elist can hold a NEWER batch than
        # the one being consumed -- its evict then belongs to THIS step's
        # in-graph head and must not be zeroed away.
        self._elist_batch = None
        # cumulative stats (host; only maintained under LOCKS_K5_STATS)
        self.sel = 0
        self.pinned_miss = 0
        self.staged_miss = 0
        self.dropped = 0
        self.lost_v = 0
        self.assigned = 0         # K7: entries given a slot (== transported)
        self.ce_bytes = 0         # transported HOST->HOT bytes (K+V for mem-kv)
        self.fetch_us = 0.0       # side-stream transport GPU time (event pairs)
        self.steps = 0
        # per-(page, kv-head) tile bytes moved by one pinned-miss transport
        # (V, plus K for mem-kv); telemetry only.
        self._tile_bytes = tier.page * tier.d * 2 \
            * (2 if tier.pool_k is not None else 1)
        # (t0, fence) event pairs of not-yet-harvested transport launches;
        # harvested non-blockingly (fence.query()) on later service() calls.
        self._timing = []

    def service(self, defer_zev: bool = False) -> None:
        t, k5 = self._tier, self._k5
        res = t.residency
        self.steps += 1
        # ---- 1. promote publish: install the OLDEST fenced transport batch ---
        # (device -> device copy of the snapshot; the graph-head promote reads
        # plist_dev/pcnt_dev.  One batch/step -- transport lands in ~1 step, so
        # produce=consume=1 at steady state.)
        k5.ecnt_dev.zero_()                   # phase-1 list: fresh batch only
        published = False
        # PROMO1 (batch ENGINES, res.R > 1): the PROLOGUE owns the entire
        # apply -- consume_fenced_prologue evicts+promotes fenced batches
        # straight from their snapshots at every begin_step (prefill steps
        # included), and the in-graph head walks permanently-zero lists.
        # The deployed publish path is a WINDOW-BOUNDARY LEAK at batch: a
        # batch published at the last decode step of a generate has no
        # decode head after it, and the next (prefill) service's eager
        # pcnt/ecnt zero WIPES the unapplied lists -- owners leak (measured
        # 86-110K stale owners per boundary, diag7/8) and the lost EVICT
        # leaves the victims' old p2s mappings pointing at slots whose
        # bytes were overwritten (a wrong-bytes hazard the R2-exact batch
        # pool makes live; bs=1's huge pool almost always evicts EMPTY
        # slots, which is why the deployed path never showed it).
        # R == 1 engines run the deployed path verbatim.
        _prologue_owns = _K5_PROMO1 and int(res.R) > 1
        if not _prologue_owns:
            still = []
            for fence_ev, b in self._inflight:
                if (not published) and fence_ev.query():
                    _bcopy(k5.snap_mail[b], k5.plist_dev, k5.snap_cnt[b],
                           cap=k5.mcap)
                    k5.pcnt_dev.copy_(k5.snap_cnt[b], non_blocking=True)
                    published = True
                    self._pub_pending = True  # PROMO1 guard (head applies)
                else:
                    still.append((fence_ev, b))
            self._inflight = still
        if not published:
            k5.pcnt_dev.zero_()               # nothing to promote this step
        # stats are PURE TELEMETRY: read only under LOCKS_K5_STATS/_ASSERT so the
        # steady path never pays a D2H sync (P7).
        if _K5_STATS:
            stats = k5.stats.cpu()
            d_sel = int(stats[0])
            d_pin = int(stats[1])
            d_stg = int(stats[2])
            d_drop = int(stats[3])
            d_lost = int(stats[4])
            d_asn = int(stats[5])
            self.sel += d_sel
            self.pinned_miss += d_pin
            self.staged_miss += d_stg
            self.dropped += d_drop
            self.lost_v += d_lost
            self.assigned += d_asn
            # fetched bytes follow what the transport MOVES: under K7 that is
            # the ASSIGNED entries (stats[5], counted in _k5_assign; give-ups
            # stay pinned-in-place and are never transported), under the
            # legacy protocol every appended pinned miss (k5b claims at
            # append time, so appended == transported there).  The miss RATE
            # keeps counting appends (a give-up is still a miss).  Sec 22c
            # change 5: both identities restated, neither approximated.
            self.ce_bytes += (d_asn if _K5_V7 else d_pin) * self._tile_bytes
            k5.stats.zero_()
            # DIAGNOSTIC per-step line (S4 residency campaign).  The counter
            # atomic_adds in _k5a/_k5b_kernel are CONSTEXPR-GATED on STATS
            # (=_K5_STATS), so with LOCKS_K5_STATS unset they compile out and
            # the device pays nothing.  The earlier claim here -- that the
            # unconditional counters added "ZERO device work" -- was measurably
            # FALSE: one of the four atomics per miss was a counter nobody
            # reads, and each scalar atomic costs ~0.033 us in the k5b
            # per-launch fit (2026-07-18 census).  Under LOCKS_K5_STATS the
            # counters are fully live and cost the D2H sync plus one small
            # reduction -> diagnostic cells ONLY, never a timing cell.
            units = max(1, t.L * t.n_kv)
            occ = int((res.page2slot >= 0).sum())
            # phase-3 slot census (diagnostic): the assign's own free
            # predicate, evaluated at the service point.
            clk_v = int(res.clock.item())
            win = 1 if (_K8 and int(res.R) == 1) else t.L
            ep = res.lru_clock.view(-1, res.S)
            ow = k5.owner.view(-1, res.S)
            n_fresh = int((ep > clk_v - win).sum())
            n_owned = int((ow != 0).sum())
            n_free = int(((ep <= clk_v - win) & (ow == 0)).sum())
            # owner age (assign stamps epoch=clk at claim): older than ~4
            # steps = not plausibly in flight -> leaked.
            n_ow_stale = int(((ow != 0) & (ep <= clk_v - 4 * win)).sum())
            print(f"[k5census] step={self.steps} clk={clk_v} win={win} "
                  f"slots={ep.numel()} fresh={n_fresh} owned={n_owned} "
                  f"ow_stale={n_ow_stale} free={n_free} "
                  f"inflight={len(self._inflight)} "
                  f"pubpend={int(self._pub_pending)}", flush=True)
            print(f"[k5stats] step={self.steps} L={t.L} nkv={t.n_kv} "
                  f"units={units} S={res.S} NB={t.NB} "
                  f"sel={d_sel} sel/u={d_sel / units:.2f} "
                  f"pinned_miss={d_pin} staged_miss={d_stg} asn={d_asn} "
                  f"miss/u={(d_pin + d_stg) / units:.3f} "
                  f"pin/u={d_pin / units:.3f} stg/u={d_stg / units:.3f} "
                  f"dropped={d_drop} drop/u={d_drop / units:.3f} "
                  f"lost_v={d_lost} "
                  f"resident={occ} res/u={occ / units:.1f} "
                  f"fill={occ / (units * max(1, res.S)):.3f}", flush=True)
            if self.steps % 16 == 0:
                print(self.stats_line(), flush=True)
            # harvest completed transport timings without blocking: the fence
            # of a finished launch has fired; elapsed_time is then free.
            still_t = []
            for t0, fence in self._timing:
                if fence.query():
                    self.fetch_us += t0.elapsed_time(fence) * 1e3
                else:
                    still_t.append((t0, fence))
            self._timing = still_t
        # ---- 2. transport this step's mailbox (device kernel; no host sync) --
        b = self._par
        busy = any(bi == b for _, bi in self._inflight)
        if busy:
            # both snapshot buffers still in flight (transport slower than 1
            # step): leave the mailbox for next step (K5b keeps appending; pages
            # stay pinned meanwhile -- bitwise-free).  Do NOT zero mail_cnt.
            return
        self._par ^= 1
        has_k = t.pool_k is not None
        poolv = t.pool.dev_view
        poolk = t.pool_k.dev_view if has_k else poolv
        hotv = res.hot_i16
        hotk = res.hot_k_i16 if has_k else hotv
        main = torch.cuda.current_stream(device=t.device)
        gate = torch.cuda.Event()
        gate.record(main)                     # after the prev forward's K5b
        tr = self._stream
        tr.wait_event(gate)
        with torch.cuda.stream(tr):
            # snapshot the mailbox (device->device), then release mail_cnt EARLY:
            # the transport reads the SNAPSHOT, so the zero (which the next
            # forward's K5b waits on) does not depend on the ~0.9 ms kernel.
            if _Z1:
                # Z1: one kernel = snapshot + clamp + per-unit bucketize +
                # early mail_cnt zero (program-ordered inside the block).
                from .k5_cuda import get_mod as _z1_mod
                _z1_mod().mail_snap_bucketize(
                    k5.mail, k5.mail_cnt, k5.snap_mail[b], k5.snap_cnt[b],
                    k5.bkt_cnt, k5.bkt_idx, res.page2slot, res.overflow,
                    t.NB, t.n_kv, t.L * t.n_kv, k5.mcap, _Z1_CAP, True)
            else:
                _bcopy(k5.mail, k5.snap_mail[b], k5.mail_cnt,
                       cap=k5.mcap)
            # _k5b_kernel increments mail_cnt UNCONDITIONALLY (atomic_add before
            # the `mi < MCAP` store guard), so on a mailbox overflow -- working
            # set >> hot slots, i.e. heavy residency thrash at large batch /
            # small hot_slots -- mail_cnt runs PAST _MCAP while only the first
            # _MCAP entries were actually written (the rest were reverted to
            # pinned-in-place by K5b, correct + bitwise-free).  The device
            # transport and the graph-head promote both walk [0, n) over the
            # _MCAP-row snap_mail / plist buffers, so an unclamped n reads off
            # the end -> cudaErrorIllegalAddress.  Clamp to _MCAP: process
            # exactly the written entries; overflow pages stay pinned (read in
            # place).  The pre-P7 copy-engine path was implicitly safe because
            # `mail[:cnt].cpu()` clips a slice; the device kernel does not.
            if not _Z1:
                torch.clamp(k5.mail_cnt, max=k5.mcap, out=k5.snap_cnt[b])
            if _K5_V7:
                # K7 victim assignment: ONE launch per step, on this transfer
                # stream, BEFORE the elist copy (so the graph-head evict sees
                # the assigned slots) and before the transport (stream order
                # gives it the slots' bytes destinations).  Runs inside the
                # zev window: the main stream's step-head wait now covers it
                # (priced, sec 22e: 15-40us, hidden under the ~284us host
                # prologue gap).  Rows with slot >= 0 (legacy claims) are
                # skipped inside the kernel, so mixed batches are safe.
                # RECENCY WINDOW must match the TICK REGIME, not the K8 env
                # (K7B fix): the K8 absorbed path (clock ticks once per
                # step) is n_req==1-gated in decode_mem, so at batch the
                # per-layer k5_step ticks L times per step even with
                # LOCKS_MEM_K8=1 -- a window of 1 there would let the
                # assign evict slots touched THIS step (policy thrash;
                # still bitwise, but exactly the churn burn K7B removes).
                # n_req == 1 keeps the historical expression verbatim.
                # PHASE-3 FIX: keyed on the ENGINE'S max_reqs (res.R), not
                # the step's n_req -- a batch engine's n_req==1 TRANSITION
                # steps (window drains) carry batch-vintage epochs (clock
                # ticked L/step), and window=1 there mass-claims the whole
                # pool in one step (measured: owned 38K -> 109K stale at
                # the win=1 census step, diag7 -- the batch that claimed
                # them was then lost across the mode transition).  R == 1
                # engines evaluate the historical expression verbatim
                # (every step there has n_req == 1).
                _nreq = int(res.R)
                use_k7c = _K5_K7C and _nreq > 1
                if use_k7c:
                    # K7C: per-unit buckets over the snapshot (fairness
                    # across layers; kills the units x mailbox filter
                    # scan), then the WIDE-draw assign.
                    global _K7C_ANNOUNCED
                    if not _K7C_ANNOUNCED:
                        _K7C_ANNOUNCED = True
                        print(f"[locks] K7C ACTIVE: batch service capacity "
                              f"(mcap={k5.mcap} bkt_cap={k5.bkt_cap} "
                              f"wide-draw assign, n_req={_nreq})",
                              flush=True)
                    k5.bkt_cnt.zero_()
                    _k7c_bucketize_kernel[(64,)](
                        k5.snap_mail[b], k5.snap_cnt[b], res.page2slot,
                        k5.bkt_cnt, k5.bkt_idx, res.overflow,
                        k5.bkt_cap, t.NB,
                        NKV=t.n_kv, BLOCK=256, GRID=64, num_warps=4)
                _k5_assign_kernel[(t.L, t.n_kv)](
                    k5.snap_mail[b], k5.snap_cnt[b],
                    res.page2slot, res.lru_clock, k5.owner, k5.hand,
                    res.clock, k5.scratch, res.overflow, k5.stats,
                    k5.bkt_cnt, k5.bkt_idx,
                    (k5.bkt_cap if use_k7c else _Z1_CAP), k5.fscr,
                    t.NB, res.S, (1 if (_K8 and _nreq == 1) else t.L),
                    NKV=t.n_kv, BLOCK=256, STATS=_K5_STATS,
                    BUCKETED=(_Z1 or use_k7c), WIDE=use_k7c, num_warps=4)
            # P15: the SAME fresh batch drives the graph-head phase-1 evict
            # (before any decode of this step reads page2slot); bounded by
            # zev below, which main waits before the replay.  PROMO1 batch
            # engines skip the elist publish entirely -- the prologue
            # consume evicts straight from the snapshot, and the in-graph
            # head must walk a zero list (an elist published at the last
            # decode step of a window would otherwise be wiped unapplied;
            # see the publish-loop comment).
            if not _prologue_owns:
                _bcopy(k5.snap_mail[b], k5.elist_dev, k5.snap_cnt[b],
                       cap=k5.mcap)
                k5.ecnt_dev.copy_(k5.snap_cnt[b], non_blocking=True)
                self._elist_batch = b         # PROMO1 bookkeeping
            if not _Z1:                   # Z1: zeroed inside the bucketize
                k5.mail_cnt.zero_()
        zev = torch.cuda.Event()
        zev.record(tr)                        # after the zero, BEFORE the kernel
        if defer_zev:
            # K5_EARLY: the zero only gates the NEXT graph's K5b.  At the
            # post-graph issue point, waiting here would serialize the LM
            # head behind snapshot+assign (measured: assign 31.5us fully
            # exposed BEFORE the LM-head span, nsys_gt_k5e) -- defeating
            # the overlap that motivates early issue.  The wait is taken
            # in begin_step, right before the next graph, instead.
            self._zev_pending = zev
        else:
            main.wait_event(zev)              # next forward's K5b waits the zero
        t0 = None
        if _K5_STATS:
            t0 = torch.cuda.Event(enable_timing=True)
            t0.record(tr)
        with torch.cuda.stream(tr):
            _k5_transport_kernel[(_TR_GRID,)](
                k5.snap_mail[b], k5.snap_cnt[b], poolv, poolk, hotv, hotk,
                t.NB, res.S, hotv.stride(0),
                GRID=_TR_GRID, NKV=t.n_kv, PAGE=t.page, D=t.d,
                HAS_K=has_k, num_warps=4)
        fence = torch.cuda.Event()
        fence.record(tr)                      # transport done -> promote may read
        self._inflight.append((fence, b))
        if _K5_STATS:
            t1 = torch.cuda.Event(enable_timing=True)
            t1.record(tr)
            self._timing.append((t0, t1))

    def consume_fenced_prologue(self) -> bool:
        """PROMO1 (batch ENGINES, res.R > 1): apply the oldest FENCED
        transport batch at the true prologue, straight from its snapshot.

        Caller contract (tier.begin_step): main stream, pre-graph, runs at
        EVERY non-capture begin_step of an R > 1 engine (prefill and drain
        steps included -- that is what closes the window-boundary leak),
        AFTER the zev consume when one is pending (orders main behind the
        tr-stream assign that wrote the slots into the snapshot; the fence
        wait below subsumes it otherwise) and AFTER invalidate_written
        (deployed relative order; the gen guard covers either way).

        Both phases walk THE SNAPSHOT (no elist/plist copies -- at R > 1
        the service never publishes either list and the in-graph head walks
        permanently-zero counts).  Evict then promote as TWO launches: the
        same batch appears in both phases, so the fused head's phase-
        disjointness premise does not hold and a kernel boundary is the
        ordering.  Applies AT MOST ONE batch per call; unfenced batches
        stay for a later prologue.

        Returns True iff a batch was consumed."""
        t, k5 = self._tier, self._k5
        res = t.residency
        if not self._inflight:
            return False
        fence_ev, b = self._inflight[0]
        if not fence_ev.query():
            return False                      # transport still in flight
        self._inflight.pop(0)
        main = torch.cuda.current_stream(device=t.device)
        main.wait_event(fence_ev)             # formal order (already fired)
        from .k5_cuda import get_mod
        mod = get_mod()
        if mod is not None:
            # evict-only launch (pcnt := zero), then promote-only (ecnt :=
            # zero), both walking the snapshot.  tick=False (batch ticks
            # per layer in k5_step).
            mod.k5_headv(k5.snap_mail[b], k5.snap_cnt[b], k5.snap_mail[b],
                         k5.zcnt, res.page2slot, res.slot2page,
                         k5.owner, k5.gen, res.clock, res.fetch_cnt,
                         res.NB, res.S, res.n_kv, False)
            mod.k5_headv(k5.snap_mail[b], k5.zcnt, k5.snap_mail[b],
                         k5.snap_cnt[b], res.page2slot, res.slot2page,
                         k5.owner, k5.gen, res.clock, res.fetch_cnt,
                         res.NB, res.S, res.n_kv, False)
        else:
            # Triton fallback: point the standalone walkers at the
            # snapshot via the device list args they read.
            _k5_evict_kernel[(_PGRID,)](
                k5.snap_mail[b], k5.snap_cnt[b],
                res.page2slot, res.slot2page,
                res.NB, res.S, NKV=res.n_kv, GRID=_PGRID, num_warps=1)
            _k5_promote_kernel[(_PGRID,)](
                k5.snap_mail[b], k5.snap_cnt[b],
                res.page2slot, res.slot2page, k5.owner, k5.gen,
                res.NB, res.S, NKV=res.n_kv, GRID=_PGRID, num_warps=1)
        global _PROMO1_ANNOUNCED
        if not _PROMO1_ANNOUNCED:
            _PROMO1_ANNOUNCED = True
            print("[locks] PROMO1 ACTIVE: fenced batch applied at the "
                  "prologue (snapshot-direct, install latency 2 -> 1 "
                  "step)", flush=True)
        return True

    def stats_line(self) -> str:
        sel = max(self.sel, 1)
        steps = max(self.steps, 1)
        return (f"[locks k5] steps={self.steps} sel={self.sel} "
                f"pinned_miss={self.pinned_miss} staged_miss={self.staged_miss} "
                f"assigned={self.assigned} "
                f"dropped={self.dropped} lost_v={self.lost_v} "
                f"hit={1.0 - (self.pinned_miss + self.staged_miss) / sel:.4f} "
                f"fetched={self.ce_bytes / 2**20:.1f}MiB "
                f"({self.ce_bytes / steps / 2**20:.3f}MiB/step) "
                f"fetch_gpu={self.fetch_us / 1e3:.1f}ms "
                f"({self.fetch_us / steps:.1f}us/step side-stream)")

"""rki4 build / refresh -- the packed rank-8 page-projection summary (rki4).

Port of ``LOCKS-test2/kernel/r8score_cuda.py::build_summary`` into the mainline
build lifecycle (same page-finalize discipline as ``build.py`` /
``quad_build.py``: bulk at first decode / prefill overlap, sync-free tail on
page completion, tag-scan delta on composition changes).  Only COMPLETE pages
are summarized -- the partial tail lives inside the always-attended window (A0
tail-refresh semantics; the historical off-by-one is handled by the builder's
settled-length gate, unchanged).  Per (block, kv-head), for a finalized page:

    mu   = mean_t K            (d,)                page centroid
    dc   = K - mu              (page, d)           centered keys
    Gm   = dc @ dc^T           (page, page)        page-gram
    S2,U = eigh(Gm)            CHUNKED at 8192 grams (cusolver batched syev
                               fails above ~16k matrices)
    S8   = sqrt(clamp(S2[-8:]))                    top-8 singular values
    U8   = U[..., -8:]
    C    = U8 * S8             (page, 8)  int8, per-token bf16 scales
    V    = dc^T U8 / S8        (d, 8)     IBITS-bit, per-column bf16 scales
                               (scale = absmax/QMAX; i4 default = absmax/7),
                               packed COLUMN-major at 8//IBITS values/byte
                               (i4: 8 x d/2 bytes, lo nibble = even d row)
    mu                         int8, per-page bf16 scale (absmax/127)

The math AND the quantization are the standalone constructor's EXACT op
sequence (G2 gate: bitwise-equal tensors given identical K).  Two deliberate
deviations from ``build.py``:

  * eigh stays ``torch.linalg.eigh`` (chunked), NOT the 16x16 Jacobi kernel:
    the Jacobi eigenbasis differs in ulps and would break the G2 bitwise
    contract against the standalone constructor.
  * V is quantized ``round(V/(absmax/QMAX)).clamp(QMIN, QMAX)`` (the
    standalone's asymmetric range; i4 default ``/7, -8..7``), not the
    symmetric +-7 of ``build._quant``. The i-axis (LOCKS_RKI4_IBITS)
    moves ONLY this V quant + pack; C/mu stay int8.

NOT graph-safe by design (eigh + gathers); runs on page-finalize off the hot
path, exactly like ``r8_build_*`` / ``quad_build_*``.
"""
from __future__ import annotations

import os

import torch

from .build import (_build_blocks, _bulk_blocks, _delta_select,
                    _scatter_index, _tail_blocks)
from .rki4_state import IBITS, QMAX, QMIN, RNK, Rki4State

from .. import quant as _kvq   # KV-quant study seam; _kvq.KVQ None = inert

from .rki4_state import MUC as _MUC  # doc 25: default-on at RNK<8

# build_summary's eigh chunk: cusolver batched syev fails above ~16k matrices.
# LOCKS_EIGH_CHUNK overrides the chunk for the TTFT chunk-size lever.  This is
# BITWISE-INERT: torch's batched cusolver syevd solves each 16x16 page gram
# INDEPENDENTLY, so the chunk (batch) size cannot move any S2/U value -- the
# same grouping-invariance already measured at S/U delta 0.000e+00 for
# layer-group splits.  Smaller chunks shrink the ~2.4 GiB syevd workspace so
# the caching allocator can retain/reuse it across calls instead of re-mapping
# it (cuMem* churn) per call.  Default 8192 keeps the shipped path identical;
# any override is gated slab-bitwise + PTAB_EXACT before it lands.
_EIGH_CHUNK = int(os.environ.get("LOCKS_EIGH_CHUNK", "8192"))

# Eigensolver for the 16x16 page grams. Default "cusolver" (the standalone
# constructor's byte-identical G2 path). "jacobi"/"dual" routes to the fused
# dual-gram Jacobi kernel (jacobi_dual.eigh_dual) -- 2.80x over cusolver at 16K,
# LICENSED by the reconstruction-equivalence + K3 (e28 recall/mass) gates
# (recall/mass identical to <=1e-4; page-LSE within the int4/int8 quant floor).
# Off by default because the swap is value-equivalent but NOT bitwise (the
# top-8 eigenbasis differs by a harmless sign/rotation), so existing
# G2-bitwise regression checks stay on cusolver. See ours_doc/TTFT_16K_KERNEL.md.
_RKI4_EIGH = os.environ.get("LOCKS_RKI4_EIGH", "cusolver").strip().lower()

# LOCKS_BUILD_TIME=1: cumulative build-eigh profiler (mirrors the score
# profiler's LOCKS_RKI4_TIME). Prints total eigh wall + gram count per call.
_EIGH_MS = 0.0
_EIGH_N = 0
_EIGH_GRAMS = 0

# LOCKS_TAIL_GRAPH (2026-07-19): CUDA-graph the bs=1 page-crossing tail
# refresh -- two captured graphs around the (non-capturable) cusolver eigh,
# replayed with static buffers (rki4_tailgraph).  graphprof measured the
# eager chain at 3.6-4.0 ms inter-graph wall every 16th step (~176 launches,
# 2.7-3.1 ms GPU-idle); the standalone prototype (scratch_qgemv/
# refresh_probe.py) cut host-blocking 2118 -> 326 us with the slab bytes
# BITWISE equal (same ATen kernels, replayed).  Kill switch: unset/0.  Any
# guard miss (n_req != 1, capture failure, profiling pass) falls back to the
# eager path below.
_TAIL_GRAPH = os.environ.get("LOCKS_TAIL_GRAPH", "0") == "1"


def _eigh_chunked(Gf: torch.Tensor):
    """(M, n, n) fp32 grams -> (S2, U) ascending, chunked at ``_EIGH_CHUNK``
    (the standalone constructor's exact loop). LOCKS_RKI4_EIGH=jacobi swaps in
    the fused dual-gram Jacobi kernel for the 16x16 grams (gate-licensed);
    LOCKS_RKI4_EIGH=syevd routes to the direct cusolverDnXsyevBatched binding
    with a PERSISTENT workspace (rki4_syevd.eigh_syevd) -- BITWISE-identical to
    torch.linalg.eigh (same routine, S2/U max|d|=0 gate), but the ~2.4 GiB
    workspace is mapped once and reused, removing the per-build cuMem* churn
    that dominates the summary-build TTFT cost, and the call omits the info
    read-back that host-syncs ATen's eigh. See ours_doc/SYEVD_PERSISTENT."""
    _bp = os.environ.get("LOCKS_BUILD_TIME") == "1"
    if _bp:
        global _EIGH_MS, _EIGH_N, _EIGH_GRAMS
        _be0 = torch.cuda.Event(enable_timing=True)
        _be1 = torch.cuda.Event(enable_timing=True)
        _be0.record()
    if _RKI4_EIGH in ("jacobi", "dual") and Gf.shape[-1] == 16:
        from .jacobi_dual import eigh_dual
        res = eigh_dual(Gf)
    elif _RKI4_EIGH == "syevd":
        from .rki4_syevd import eigh_syevd
        res = eigh_syevd(Gf, _EIGH_CHUNK)
    else:
        n = Gf.shape[-1]
        S2 = torch.empty(Gf.shape[0], n, device=Gf.device)
        U = torch.empty_like(Gf)
        for i in range(0, Gf.shape[0], _EIGH_CHUNK):
            S2[i:i + _EIGH_CHUNK], U[i:i + _EIGH_CHUNK] = \
                torch.linalg.eigh(Gf[i:i + _EIGH_CHUNK])
        res = (S2, U)
    if _bp:
        _be1.record(); torch.cuda.synchronize()
        _EIGH_MS += _be0.elapsed_time(_be1); _EIGH_N += 1
        _EIGH_GRAMS += int(Gf.shape[0])
        print(f"[locks] eigh BUILD PROFILE: {_EIGH_N} calls, {_EIGH_GRAMS} "
              f"grams, {_EIGH_MS/1000:.2f}s total", flush=True)
    return res


# --------------------------------------------------------------------------- #
def _mask_blocks(st: Rki4State, K: torch.Tensor,
                 blocks: torch.Tensor) -> torch.Tensor:
    """Bound physical-block indices to BOTH the state slab and the K tensor
    (sync-free boolean mask). The engine's memory-profile pass hands a
    placeholder block table (narrow and/or garbage values): the OOB gathers/
    scatters silently corrupted the six-array dummy state by mapped-neighbor
    luck, and FAULT on the compact AOS record tensor. Real tables never
    produce OOB blocks, so this is the identity outside the profile pass."""
    nb_state = (st.pp_rec.shape[1] if getattr(st, "pp_rec", None) is not None
                else st.pp_v4.shape[1])
    nb = min(int(nb_state), int(K.shape[0]))
    return blocks[(blocks >= 0) & (blocks < nb)]


def _write_pre(st: Rki4State, Kp_flat: torch.Tensor):
    """Stage A of ``_rki4_write``: gather-side math up to the page grams.
    Split point = the eigh call, so the tail graph (``rki4_tailgraph``) can
    CUDA-graph A/B around the non-capturable cusolver solve.  Op order is
    ``_rki4_write``'s, unchanged (pure extraction)."""
    page, tagw = st.page, st.tagw
    tag_new = Kp_flat[:, page - 1, :, :tagw]
    K = Kp_flat.permute(0, 2, 1, 3).float().contiguous()   # (N, n_kv, page, d)
    mu = K.mean(2)                                         # (N, n_kv, d)
    dc = K - mu[:, :, None]
    Gm = dc @ dc.transpose(-1, -2)                         # (N, n_kv, page, page)
    return tag_new, mu, dc, Gm


def _pack_ibit(Vc: torch.Tensor) -> torch.Tensor:
    """Pack int-quantized V columns (N, n_kv, RNK, d) int8 -> uint8
    (N, n_kv, RNK, d*IBITS//8), column-major at PPB = 8//IBITS values/byte,
    field k of each byte = d row k::PPB (lowest field = lowest row). IBITS=4
    emits lo|(hi<<4) with lo = even d row -- op-for-op the shipped nibble
    pack (bitwise regression anchor); i8 = the raw int8 bytes; i2 = four
    2-bit fields."""
    if IBITS == 8:
        return Vc.view(torch.uint8).contiguous()
    ppb = 8 // IBITS
    mask = (1 << IBITS) - 1
    out = (Vc[..., 0::ppb] & mask).to(torch.uint8)
    for k in range(1, ppb):
        out = out | ((Vc[..., k::ppb] & mask).to(torch.uint8) << (k * IBITS))
    return out.contiguous()


def _write_post(st: Rki4State, mu, dc, S2, U, tag_new, lidx, bidx) -> None:
    """Stage B of ``_rki4_write``: eigh consumers -> quantize -> slab scatter.
    Op order unchanged (pure extraction; see ``_write_pre``)."""
    page = st.page
    N, n_kv = mu.shape[0], mu.shape[1]
    S2 = S2.reshape(N, n_kv, page)
    U = U.reshape(N, n_kv, page, page)
    S8 = S2[..., -RNK:].clamp_min(1e-10).sqrt()
    U8 = U[..., -RNK:]
    C = U8 * S8[..., None, :]                              # (N, n_kv, page, 8)
    V = dc.transpose(-1, -2) @ U8 / S8[..., None, :]       # (N, n_kv, d, 8)
    # IBITS-bit basis, per-column scales (absmax/QMAX; asymmetric clamp to
    # QMIN..QMAX -- the shipped i4 /7.0, -8..7 at the default)
    vsc = (V.abs().amax(2) / float(QMAX)).clamp_min(1e-8)  # (N, n_kv, RNK)
    Vq = torch.round(V / vsc[:, :, None]).clamp(QMIN, QMAX)
    # pack column-major: (N, n_kv, RNK, d*i/8), 8//IBITS values per byte
    Vc = Vq.permute(0, 1, 3, 2).contiguous().to(torch.int8)  # (N, n_kv, RNK, d)
    v4 = _pack_ibit(Vc)
    # int8 coeffs, per-token scales
    csc = (C.abs().amax(-1) / 127.0).clamp_min(1e-8)       # (N, n_kv, page)
    Cq = torch.round(C / csc[..., None]).clamp(-127, 127)
    # int8 mu, per-page scale
    msc = (mu.abs().amax(-1) / 127.0).clamp_min(1e-8)      # (N, n_kv)
    muq = torch.round(mu / msc[..., None]).clamp(-127, 127)

    st.pp_v4[lidx, bidx] = v4
    st.pp_vs[lidx, bidx] = vsc.to(torch.bfloat16)
    st.pp_c8[lidx, bidx] = Cq.to(torch.int8)
    st.pp_cs[lidx, bidx] = csc.to(torch.bfloat16)
    if _MUC and st.G == 8:
        # MUC (doc 25): store mu even/odd-d packed (the mma B-fragment order,
        # matching the q8e/q8o A staging); values unchanged, order only.
        # G==8 GATE (MUC_G4_RANK_BUG_2026-07-25): the even/odd layout is what
        # the G8 USE_MMA path consumes under #ifdef RKI4_MMA_MUC; the G4
        # non-MMA path has NO MUC handling and read the permuted mu as natural
        # order -- scrambling every page's centroid term (RULER-16K r4: 38.7
        # vs 93.4 with the permute off). G<8 states now store natural order,
        # which is exactly what the G4 kernel (and the torch twin) compute on.
        muq = torch.cat([muq[..., 0::2], muq[..., 1::2]], dim=-1)
    st.pp_mu8[lidx, bidx] = muq.to(torch.int8)
    st.pp_mus[lidx, bidx] = msc.to(torch.bfloat16)
    if getattr(st, "pp_nrmC", None) is not None:
        # SCREEN radius norm (doc 26): max_t cs_t * ||c8[t,:]||_1, rounded UP
        # (the 1+2^-7 pre-scale dominates bf16 round-to-nearest) so the
        # stored radius NEVER under-covers. Zero-cost when records absent.
        nrmC = (Cq.abs().sum(-1) * csc).amax(-1) * (1.0 + 2.0 ** -7)
        st.pp_nrmC[lidx, bidx] = nrmC.to(torch.bfloat16)
    st.pp_tag[lidx, bidx] = tag_new.to(st.pp_tag.dtype)


def _rki4_write(st: Rki4State, Kp_flat: torch.Tensor, lidx: torch.Tensor,
                bidx: torch.Tensor) -> None:
    """Summarize + quantize + scatter ``Kp_flat`` (N, page, n_kv, d) into the
    rki4 slabs at the flat (layer, block) index pairs.  Sync-free.  The op
    chain (``_write_pre`` -> eigh -> ``_write_post``) is line-for-line
    ``build_summary`` on (N, n_kv)-batched pages (the standalone batches
    (n_kv, P); per-matrix ops are batch-order independent, verified bitwise
    by the G2 gate)."""
    # Bound the scatter to the state's physical slab: the engine's memory-
    # profile pass hands a placeholder block table (narrow and/or garbage
    # values), and an OOB index_put into the slabs silently corrupted the
    # six-array dummy state (mapped-neighbor luck) but FAULTS on the compact
    # AOS record tensor. Real tables never produce OOB blocks, so this mask
    # is the identity outside the profile pass.
    nb = st.pp_v4.shape[1] if getattr(st, "pp_rec", None) is None \
        else st.pp_rec.shape[1]
    ok = (bidx >= 0) & (bidx < nb)
    Kp_flat, lidx, bidx = Kp_flat[ok], lidx[ok], bidx[ok]   # sync-free mask
    # KV-QUANT STUDY SEAM (locks.quant; LOCKS_KVQ unset/off = this branch is
    # one None check, no tensor touched).  The hook may (a) filter out pages
    # already quantized-and-unchanged (idempotency of the in-place cache
    # fake-quant), (b) swap the summary's K source (stage pre/selq), and
    # (c) override the stored tag so it always matches the cache bytes.
    tag_override = None
    if _kvq.KVQ is not None:
        hooked = _kvq.build_hook(st, Kp_flat, lidx, bidx)
        if hooked is None:
            return                         # all pages already processed
        Kp_flat, tag_override, lidx, bidx = hooked
    tag_new, mu, dc, Gm = _write_pre(st, Kp_flat)
    if tag_override is not None:
        tag_new = tag_override
    S2, U = _eigh_chunked(Gm.reshape(-1, st.page, st.page))
    _write_post(st, mu, dc, S2, U, tag_new, lidx, bidx)


def rki4_build_tail(st: Rki4State, K_layers, block_table: torch.Tensor,
                    seq_lens: torch.Tensor, n_req: int, rows=None) -> int:
    """Batched all-layer rebuild of the last finalized page of ``rows`` (or of
    every request).  rki4 twin of :func:`locks.selection.build.r8_build_tail`
    (rationale there): zero device syncs, idempotent.  Called by the builder
    on steady finalize steps only (the ``cur % page == 1`` settled gate)."""
    if n_req == 0:
        return 0
    if _TAIL_GRAPH:
        from .rki4_tailgraph import tail_graph_run
        r = tail_graph_run(st, K_layers, block_table, seq_lens, n_req, rows)
        if r is not None:
            return r
    L = len(K_layers)
    blocks = _tail_blocks(block_table, seq_lens, n_req, st.page, rows)
    blocks = _mask_blocks(st, K_layers[0], blocks)
    if blocks.numel() == 0:
        return 0
    n = blocks.shape[0]
    # (LOCKS_TAIL_BATCH, the TP4 Round C index_select-batched gather, was
    # removed 2026-07-21, cleanup Wave 3a: pair-refuted at -0.020 ms vs the
    # 0.05 falsifier because the removed host launches were already hidden;
    # SELECT_KERNEL_CAMPAIGN.md section 12 + ours_doc/REFUTED_ARMS_INDEX.md.)
    Kp_raw = torch.stack([K[blocks] for K in K_layers])      # (L,n,page,n_kv,d)
    lidx, bidx = _scatter_index(L, 0, blocks)
    _rki4_write(st, Kp_raw.reshape(L * n, st.page, st.n_kv, st.d), lidx, bidx)
    return L * n


def rki4_build_bulk(st: Rki4State, K_layers, block_table: torch.Tensor,
                    seq_lens: torch.Tensor, n_req: int, max_fin: int,
                    chunk_pairs: int = 8192) -> int:
    """Rebuild ALL finalized pages for ALL layers, zero syncs; rki4 twin of
    :func:`locks.selection.build.r8_build_bulk` (rationale there)."""
    if n_req == 0 or max_fin <= 0:
        return 0
    blocks = _bulk_blocks(block_table, seq_lens, n_req, page=st.page,
                          max_fin=max_fin)
    blocks = _mask_blocks(st, K_layers[0], blocks)
    return _build_blocks(_rki4_write, st, K_layers, blocks, chunk_pairs)


def rki4_build_delta(st: Rki4State, K_layers, block_table: torch.Tensor,
                     seq_lens: torch.Tensor, n_req: int, max_fin: int,
                     chunk_pairs: int = 8192) -> int:
    """Composition-change rebuild: only blocks whose content tag went stale;
    rki4 twin of :func:`locks.selection.build.r8_build_delta`."""
    if n_req == 0 or max_fin <= 0:
        return 0
    blocks = _delta_select(st.pp_tag, K_layers, block_table, seq_lens,
                           n_req, st.page, max_fin, st.tagw)
    blocks = _mask_blocks(st, K_layers[0], blocks)
    return _build_blocks(_rki4_write, st, K_layers, blocks, chunk_pairs)


# --------------------------------------------------------------------------- #
def rki4_build_refresh(st: Rki4State, layer: int, K: torch.Tensor,
                       block_table: torch.Tensor, seq_lens: torch.Tensor,
                       n_req: int, *, force: bool = False) -> int:
    """(Re)build the rki4 summary for every STALE finalized page of the batch.

    K            per-layer key view (NB, page, n_kv, d), the engine K half.
    block_table  (n_req, max_blocks) int32, logical page -> physical block.
    seq_lens     (n_req,) int32 SETTLED lengths.
    force        ignore tags and rebuild all finalized pages (tests).

    Returns the number of physical blocks rebuilt (0 in the steady state).
    Same tag-gated, off-hot-path contract as ``r8_build_refresh``.
    """
    device = K.device
    page, tagw = st.page, st.tagw
    if n_req == 0:
        return 0
    sl = seq_lens[:n_req].to(torch.int64)
    n_fin = sl // page                                    # finalized pages/req
    max_fin = int(n_fin.max().item())
    if max_fin == 0:
        return 0

    pidx = torch.arange(max_fin, device=device)
    fin_mask = pidx[None, :] < n_fin[:, None]             # (n_req, max_fin)
    bt = block_table[:n_req, :max_fin].to(torch.int64)    # (n_req, max_fin)
    blocks_flat = bt[fin_mask]                            # (n_valid,)
    blocks_flat = _mask_blocks(st, K, blocks_flat)
    if blocks_flat.numel() == 0:
        return 0

    if force:
        stale = torch.ones(blocks_flat.numel(), dtype=torch.bool, device=device)
    else:
        # content tag = leading TAGW channels of the page-final token's key.
        tag_cur = K[blocks_flat, page - 1, :, :tagw].float()   # (n_valid,n_kv,W)
        tag_old = st.pp_tag[layer, blocks_flat].float()
        stale = (tag_cur != tag_old).any(dim=(1, 2))           # NaN-init -> True

    sb = torch.unique(blocks_flat[stale])                 # physical blocks
    if sb.numel() == 0:
        return 0
    lidx = torch.full((sb.numel(),), layer, device=device, dtype=torch.int64)
    _rki4_write(st, K[sb], lidx, sb)
    return int(sb.numel())

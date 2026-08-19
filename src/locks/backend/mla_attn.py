"""LOCKS MLA backend -- the MLA (DeepSeek V2/V3/V3.2) decode path, folded into
the ``src/locks`` package so a single ``locks.register()`` seamlessly serves both
GQA and MLA models. The MLA branch of ``register.get_attn_backend_cls`` swaps
whatever MLA backend vLLM picks for :class:`LocksMLABackend` (same seam vLLM uses
to separate MLA from GQA).

MLA keeps ONE latent KV head: the paged cache is ``(num_blocks, block_size,
kv_lora_rank + qk_rope_head_dim)`` (e.g. 512 latent + 64 rope). vLLM's
``MLAAttention.forward_impl`` absorbs the decode query into the latent space and
calls ``self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)`` with
``mqa_q = (q_c, q_rope)`` -- exactly the input of
``locks.selection.mla_select.mla_select_pages``. We override ``forward_mqa`` only,
so PREFILL is byte-identical to stock MLA (TTFT parity).

Config comes from :class:`LocksConfig` (``_runtime.get_config()``), NOT the
environment. Supported: the ``fast`` variant with a fixed page budget
(``budget_pages`` or ``budget``). The mem tier and adaptive per-head coverage are
GQA-only for now (coverage on the latent union is a follow-up). Selection uses the
local rank-``r8_rank`` summary (the deployed method); FullKV = ``LOCKS_DISABLE=1``
(register() stays inert -> stock MLA backend).
"""
from __future__ import annotations

import math
import os

import torch

from vllm.v1.attention.backends.mla.triton_mla import (TritonMLABackend,
                                                       TritonMLAImpl)

from ..selection.mla_select import (build_page_factors, build_page_summary,
                                    mla_select_pages)
from ..selection.mla_score_torch import topb_middle
from . import _runtime

try:  # the batched factored kernel is the latency path; import-safe
    from ..selection.mla_score_triton import mla_page_score_factored_batched
except Exception:  # pragma: no cover
    mla_page_score_factored_batched = None

NEG = float("-inf")
# (id(layer), block0, prompt_pages) -> (Lf[n,PAGE,R], Rf[n,R,Lkv], k_rope, valid, n)
_C8CACHE = {}
_ANNOUNCED = False
# Optional validation aid (NOT config): write a one-line attended-fraction
# diagnostic on the first real masking decode. Proves selection is active.
_DIAG = os.environ.get("LOCKS_MLA_DIAG", "")
_DIAG_DONE = False
# Fused tf32 page-score kernel (7.7x over eager, ranking-identical). Default ON;
# LOCKS_MLA_SCORE=eager forces the torch reference (A/B for the real-data gate).
_SCORE_TRITON = os.environ.get("LOCKS_MLA_SCORE", "triton") != "eager"


def _budget_pages(cfg, n_pages: int) -> int:
    """Per-step page budget from the LocksConfig. ``budget_pages`` (absolute)
    wins; else ``budget`` (fraction) x n_pages. Coverage-adaptive selection on
    the MLA latent is not implemented yet -- a fixed budget must be set."""
    if cfg.budget_pages is not None and cfg.budget_pages > 0:
        return int(cfg.budget_pages)
    if cfg.budget is not None:
        return max(1, math.ceil(cfg.budget * n_pages))
    raise RuntimeError(
        "LOCKS MLA path requires a fixed budget: set 'budget_pages' (pages) or "
        "'budget' (fraction) in LOCKS_CONFIG. Adaptive coverage on MLA is a "
        "follow-up (the GQA flagship coverage arm is not ported to the latent "
        "union).")


class LocksMLAImpl(TritonMLAImpl):
    """MLA decode with LOCKS page selection over the latent cache. Prefill =
    stock MLA (parent). Eager decode restricted to the selected pages (quality
    path; the fused Triton scorer is available but off by default)."""

    def __init__(self, *args, **kwargs):
        # DeepSeek-V3.2 (DSA) builds its MLA attention impl with the extra
        # ``topk_indices_buffer`` kwarg (the sparse indexer's shared selection
        # buffer). The native FlashMLASparseImpl consumes it; the dense
        # TritonMLAImpl base forwards **kwargs to MLACommonImpl, which rejects
        # it (TypeError). LOCKS does its OWN page selection over the latent and
        # IGNORES the trained indexer, so we swallow the sparse-only kwarg and
        # construct the dense base normally. Harmless on V2/V3 (kwarg absent).
        kwargs.pop("topk_indices_buffer", None)
        super().__init__(*args, **kwargs)

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        cfg = _runtime.get_config()
        assert cfg is not None, "LocksMLAImpl reached with no LocksConfig"
        assert getattr(self, "kv_lora_rank", None), \
            "LocksMLAImpl requires a latent (MLA) attention (kv_lora_rank)"
        assert mla_page_score_factored_batched is not None, \
            "LOCKS MLA fast path requires the Triton factored score kernel"
        global _ANNOUNCED
        if not _ANNOUNCED:
            _ANNOUNCED = True
            print(f"[locks] ACTIVE (MLA batched) budget_pages={cfg.budget_pages} "
                  f"sink={cfg.sink_pages} window={cfg.window_pages} "
                  f"rank={cfg.r8_rank}", flush=True)

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)                        # [B, H, Lkv+Dr]
        B, H, HS = q.shape                                  # B=decode tokens, H=heads
        Lkv = self.kv_lora_rank
        scale = float(self.scale)
        dec = attn_metadata.decode
        bt = dec.block_table
        seq_lens = dec.seq_lens
        cache = kv_c_and_k_pe_cache                         # [nblk, PAGE, HS]
        PAGE = cache.shape[1]
        dev = q.device

        o = torch.zeros(B, H, Lkv, dtype=q.dtype, device=dev)
        lse = torch.zeros(B, H, dtype=q.dtype, device=dev)

        # BATCHED + FACTORED decode. The score for ALL rows' cached pages runs in
        # ONE Triton launch (the launch-overhead fix: 1 kernel/layer, not B), over
        # the COMPACT rank-r factors Lf/Rf (~4x smaller cache than the full C8 --
        # the OOM fix). Selection stays exact (rank-preserving factored score,
        # gated); attention gathers ONLY the selected pages, batched.
        seq_cpu = seq_lens.tolist()
        blk0 = bt[:, 0].tolist()
        # Batched content fingerprint of each row's FIRST and LAST prompt block
        # (start+end content -- stable across a request's decode, ~never colliding
        # across requests even when they share an instruction prefix). ONE sync/
        # forward (not the per-step syncs the old fingerprint cost). Keys the
        # factor cache so a sequence that REUSES a finished request's block ids
        # (vLLM recycles them) rebuilds instead of reading STALE factors -- the
        # correctness bug the block-id-only key missed.
        np_all = (seq_lens.clamp(min=1) + PAGE - 1) // PAGE
        last_blk = torch.gather(bt, 1, (np_all - 1).clamp(min=0)[:, None]).squeeze(1)
        fp_f = cache.index_select(0, bt[:, 0]).float()[:, :, :8].sum(dim=(1, 2))
        fp_l = cache.index_select(0, last_blk).float()[:, :, :8].sum(dim=(1, 2))
        fp_all = (fp_f + 1.0009 * fp_l).tolist()
        sink, window = int(cfg.sink_pages), int(cfg.window_pages)

        Lf_l, Rf_l, kr_l, val_l, row_l = [], [], [], [], []
        rows = []                                           # (bi, L, n_pages, bp, ncache)
        for bi in range(B):
            L = seq_cpu[bi]
            if L <= 0:
                continue
            n_pages = (L + PAGE - 1) // PAGE
            bp = _budget_pages(cfg, n_pages)
            if n_pages > bp:
                Lf, Rf, kr, val, ncache = self._factors(
                    cfg, cache, bt, bi, L, PAGE, n_pages, layer,
                    blk0[bi], fp_all[bi])
                if ncache > 0:
                    Lf_l.append(Lf); Rf_l.append(Rf); kr_l.append(kr); val_l.append(val)
                    row_l.append(torch.full((ncache,), bi, dtype=torch.int32, device=dev))
                rows.append((bi, L, n_pages, bp, ncache))
            else:
                rows.append((bi, L, n_pages, bp, 0))

        score_all = None
        if Lf_l:
            q_c = q[:, :, :Lkv]                             # [B, H, Lkv]
            q_rope = q[:, :, Lkv:]                          # [B, H, Dr]
            blk_h = 16 if H % 16 == 0 else H
            score_all = mla_page_score_factored_batched(
                q_c, q_rope, torch.cat(Lf_l), torch.cat(Rf_l), torch.cat(kr_l),
                torch.cat(val_l), torch.cat(row_l), scale, blk_h=blk_h)  # [totalP]

        # per-row topb -> kept page indices (cheap ops, no per-row score kernels)
        keep_l, off = [], 0
        for (bi, L, n_pages, bp, ncache) in rows:
            if ncache > 0:
                ps = q.new_full((n_pages,), NEG, dtype=torch.float32)
                ps[:ncache] = score_all[off:off + ncache]; off += ncache
                idx = topb_middle(ps, bp, sink, window).nonzero(
                    as_tuple=False).flatten()
            else:
                idx = torch.arange(n_pages, device=dev)
            keep_l.append((bi, L, idx))
        if not keep_l:
            return o, lse

        # batched gather of the selected pages + batched attention
        Kmax = max(idx.numel() for (_, _, idx) in keep_l)
        Bv = len(keep_l)
        pad_idx = torch.zeros(Bv, Kmax, dtype=torch.long, device=dev)
        valid_pg = torch.zeros(Bv, Kmax, dtype=torch.bool, device=dev)
        bidx = torch.empty(Bv, dtype=torch.long, device=dev)
        Ls = torch.empty(Bv, dtype=torch.long, device=dev)
        for j, (bi, L, idx) in enumerate(keep_l):
            k = idx.numel()
            pad_idx[j, :k] = idx; valid_pg[j, :k] = True
            bidx[j] = bi; Ls[j] = L
        ar = torch.arange(PAGE, device=dev)
        gblk = torch.gather(bt.index_select(0, bidx), 1, pad_idx).long()   # [Bv,Kmax]
        sel = cache.index_select(0, gblk.reshape(-1)).reshape(Bv, Kmax * PAGE, HS)
        tok_pos = (pad_idx[:, :, None] * PAGE + ar[None, None, :]).reshape(Bv, Kmax * PAGE)
        tok_ok = (valid_pg[:, :, None].expand(Bv, Kmax, PAGE).reshape(Bv, Kmax * PAGE)
                  & (tok_pos < Ls[:, None]))
        qv = q.index_select(0, bidx).float()               # [Bv, H, HS]
        selk = sel.float()
        scores = torch.bmm(qv, selk.transpose(1, 2)) * scale           # [Bv, H, T]
        scores = scores.masked_fill(~tok_ok[:, None, :], NEG)
        attn = torch.softmax(scores, dim=-1)
        o.index_copy_(0, bidx, torch.bmm(attn, selk[:, :, :Lkv]).to(q.dtype))
        lse.index_copy_(0, bidx, torch.logsumexp(scores, dim=-1).to(q.dtype))
        if _DIAG:
            global _DIAG_DONE
            if not _DIAG_DONE:
                _DIAG_DONE = True
                with open(_DIAG, "a") as fh:
                    fh.write(f"LOCKS-MLA batched: B={B} Kmax={Kmax} "
                             f"kept_tok~{int(tok_ok.sum().item())} "
                             f"rank={cfg.r8_rank} score={'triton' if _SCORE_TRITON else 'x'}\n")
        return o, lse

    def _factors(self, cfg, cache, bt, bi, L, PAGE, n_pages, layer, blk0, fp):
        """Build/lookup the COMPACT rank-r factors (Lf[n,PAGE,R], Rf[n,R,Lkv]) +
        rope keys for a sequence. Keyed by (layer, block0) so the cache is BOUNDED
        (one entry per block slot, OVERWRITTEN when the slot is reused) -- but the
        entry stores a content fingerprint, and a mismatch (or a shrink) forces a
        rebuild. That both prevents the unbounded-growth OOM AND the stale-factor
        correctness bug (block-id reuse across requests). Full KV read + SVD run
        only on a miss (first decode); steady-state decode reads nothing here."""
        Lkv = self.kv_lora_rank
        HS = cache.shape[2]
        key = (id(layer), blk0)
        entry = _C8CACHE.get(key)
        if entry is None or entry[5] != fp or (L // PAGE) < entry[4]:
            n_cache = L // PAGE
            blks = bt[bi, :n_pages].long()
            kvb = cache.index_select(0, blks).reshape(-1, HS)[:L]   # full read, ONCE
            kv_c = kvb[:n_cache * PAGE, :Lkv].float()
            Lf, Rf = build_page_factors(kv_c, n_cache, PAGE, cfg.r8_rank)
            if n_cache > 0:
                k_rope = kvb[:n_cache * PAGE, Lkv:].float().view(
                    n_cache, PAGE, -1).contiguous()
                valid = torch.full((n_cache,), PAGE, dtype=torch.long, device=cache.device)
            else:
                k_rope = kvb.new_zeros(0, PAGE, HS - Lkv, dtype=torch.float32)
                valid = torch.zeros(0, dtype=torch.long, device=cache.device)
            entry = (Lf, Rf, k_rope, valid, n_cache, fp)
            _C8CACHE[key] = entry
        return entry[:5]


class LocksMLABackend(TritonMLABackend):
    """Triton MLA backend with the LOCKS MLA impl swapped in. get_name is
    inherited (-> "TRITON_MLA") so AttentionBackendEnum resolves; the impl comes
    from THIS subclass (mirrors LocksAttentionBackend)."""

    @staticmethod
    def get_impl_cls():
        return LocksMLAImpl

from __future__ import annotations
import math
import os
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
    TritonMLAMetadataBuilder,
)
from ..selection.mla_select import build_page_factors, topb_middle
from . import _runtime
from ..selection.mla_score_triton import mla_page_score_factored_batched

NEG = float("-inf")
_C8CACHE = {}
_ANNOUNCED = False
_DIAG = os.environ.get("LOCKS_MLA_DIAG", "")
_DIAG_DONE = False
_SCORE_TRITON = True


def _budget_pages(cfg, n_pages: int) -> int:
    if cfg.budget_pages is not None and cfg.budget_pages > 0:
        return int(cfg.budget_pages)
    if cfg.budget is not None:
        return max(1, math.ceil(cfg.budget * n_pages))
    raise RuntimeError(
        "LOCKS MLA path requires a fixed budget: set 'budget_pages' (pages) or 'budget' (fraction) in LOCKS_CONFIG. Adaptive coverage on MLA is a follow-up (the GQA flagship coverage arm is not ported to the latent union)."
    )


class LocksMLAImpl(TritonMLAImpl):
    def __init__(self, *args, **kwargs):
        kwargs.pop("topk_indices_buffer", None)
        super().__init__(*args, **kwargs)

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if kv_c_and_k_pe_cache.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("LOCKS MLA requires original BF16/FP16 latent KV")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MLA factor-cache orchestration requires eager attention execution"
            )
        cfg = _runtime.get_config()
        assert cfg is not None, "LocksMLAImpl reached with no LocksConfig"
        assert getattr(self, "kv_lora_rank", None), (
            "LocksMLAImpl requires a latent (MLA) attention (kv_lora_rank)"
        )
        assert mla_page_score_factored_batched is not None, (
            "LOCKS MLA fast path requires the Triton factored score kernel"
        )
        global _ANNOUNCED
        if not _ANNOUNCED:
            _ANNOUNCED = True
            print(
                f"[locks] ACTIVE (MLA batched) budget_pages={cfg.budget_pages} sink={cfg.sink_pages} window={cfg.window_pages} rank={cfg.r8_rank}",
                flush=True,
            )
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        (B, H, HS) = q.shape
        Lkv = self.kv_lora_rank
        scale = float(self.scale)
        dec = attn_metadata.decode
        bt = dec.block_table
        seq_lens = dec.seq_lens
        cache = kv_c_and_k_pe_cache
        PAGE = cache.shape[1]
        dev = q.device
        o = torch.zeros(B, H, Lkv, dtype=q.dtype, device=dev)
        lse = torch.zeros(B, H, dtype=q.dtype, device=dev)
        seq_cpu = seq_lens.tolist()
        blk0 = bt[:, 0].tolist()
        np_all = (seq_lens.clamp(min=1) + PAGE - 1) // PAGE
        last_blk = torch.gather(bt, 1, (np_all - 1).clamp(min=0)[:, None]).squeeze(1)
        fp_f = cache.index_select(0, bt[:, 0]).float()[:, :, :8].sum(dim=(1, 2))
        fp_l = cache.index_select(0, last_blk).float()[:, :, :8].sum(dim=(1, 2))
        fp_all = (fp_f + 1.0009 * fp_l).tolist()
        (sink, window) = (int(cfg.sink_pages), int(cfg.window_pages))
        (Lf_l, Rf_l, kr_l, val_l, row_l) = ([], [], [], [], [])
        rows = []
        for bi in range(B):
            L = seq_cpu[bi]
            if L <= 0:
                continue
            n_pages = (L + PAGE - 1) // PAGE
            bp = _budget_pages(cfg, n_pages)
            if n_pages > bp:
                (Lf, Rf, kr, val, ncache) = self._factors(
                    cfg, cache, bt, bi, L, PAGE, n_pages, layer, blk0[bi], fp_all[bi]
                )
                if ncache > 0:
                    Lf_l.append(Lf)
                    Rf_l.append(Rf)
                    kr_l.append(kr)
                    val_l.append(val)
                    row_l.append(
                        torch.full((ncache,), bi, dtype=torch.int32, device=dev)
                    )
                rows.append((bi, L, n_pages, bp, ncache))
            else:
                rows.append((bi, L, n_pages, bp, 0))
        score_all = None
        if Lf_l:
            q_c = q[:, :, :Lkv]
            q_rope = q[:, :, Lkv:]
            blk_h = 16 if H % 16 == 0 else H
            score_all = mla_page_score_factored_batched(
                q_c,
                q_rope,
                torch.cat(Lf_l),
                torch.cat(Rf_l),
                torch.cat(kr_l),
                torch.cat(val_l),
                torch.cat(row_l),
                scale,
                blk_h=blk_h,
            )
        (keep_l, off) = ([], 0)
        for bi, L, n_pages, bp, ncache in rows:
            if ncache > 0:
                ps = q.new_full((n_pages,), NEG, dtype=torch.float32)
                ps[:ncache] = score_all[off : off + ncache]
                off += ncache
                idx = (
                    topb_middle(ps, bp, sink, window).nonzero(as_tuple=False).flatten()
                )
            else:
                idx = torch.arange(n_pages, device=dev)
            keep_l.append((bi, L, idx))
        if not keep_l:
            return (o, lse)
        Kmax = max((idx.numel() for (_, _, idx) in keep_l))
        Bv = len(keep_l)
        pad_idx = torch.zeros(Bv, Kmax, dtype=torch.long, device=dev)
        valid_pg = torch.zeros(Bv, Kmax, dtype=torch.bool, device=dev)
        bidx = torch.empty(Bv, dtype=torch.long, device=dev)
        Ls = torch.empty(Bv, dtype=torch.long, device=dev)
        for j, (bi, L, idx) in enumerate(keep_l):
            k = idx.numel()
            pad_idx[j, :k] = idx
            valid_pg[j, :k] = True
            bidx[j] = bi
            Ls[j] = L
        ar = torch.arange(PAGE, device=dev)
        gblk = torch.gather(bt.index_select(0, bidx), 1, pad_idx).long()
        sel = cache.index_select(0, gblk.reshape(-1)).reshape(Bv, Kmax * PAGE, HS)
        tok_pos = (pad_idx[:, :, None] * PAGE + ar[None, None, :]).reshape(
            Bv, Kmax * PAGE
        )
        tok_ok = valid_pg[:, :, None].expand(Bv, Kmax, PAGE).reshape(
            Bv, Kmax * PAGE
        ) & (tok_pos < Ls[:, None])
        qv = q.index_select(0, bidx).float()
        selk = sel.float()
        scores = torch.bmm(qv, selk.transpose(1, 2)) * scale
        scores = scores.masked_fill(~tok_ok[:, None, :], NEG)
        attn = torch.softmax(scores, dim=-1)
        o.index_copy_(0, bidx, torch.bmm(attn, selk[:, :, :Lkv]).to(q.dtype))
        lse.index_copy_(0, bidx, torch.logsumexp(scores, dim=-1).to(q.dtype))
        if _DIAG:
            global _DIAG_DONE
            if not _DIAG_DONE:
                _DIAG_DONE = True
                with open(_DIAG, "a") as fh:
                    fh.write(
                        f"LOCKS-MLA batched: B={B} Kmax={Kmax} kept_tok~{int(tok_ok.sum().item())} rank={cfg.r8_rank} score={('triton' if _SCORE_TRITON else 'x')}\n"
                    )
        return (o, lse)

    def _factors(self, cfg, cache, bt, bi, L, PAGE, n_pages, layer, blk0, fp):
        Lkv = self.kv_lora_rank
        HS = cache.shape[2]
        key = (id(layer), blk0)
        entry = _C8CACHE.get(key)
        if entry is None or entry[5] != fp or L // PAGE < entry[4]:
            n_cache = L // PAGE
            blks = bt[bi, :n_pages].long()
            kvb = cache.index_select(0, blks).reshape(-1, HS)[:L]
            kv_c = kvb[: n_cache * PAGE, :Lkv].float()
            (Lf, Rf) = build_page_factors(kv_c, n_cache, PAGE, cfg.r8_rank)
            if n_cache > 0:
                k_rope = (
                    kvb[: n_cache * PAGE, Lkv:]
                    .float()
                    .view(n_cache, PAGE, -1)
                    .contiguous()
                )
                valid = torch.full(
                    (n_cache,), PAGE, dtype=torch.long, device=cache.device
                )
            else:
                k_rope = kvb.new_zeros(0, PAGE, HS - Lkv, dtype=torch.float32)
                valid = torch.zeros(0, dtype=torch.long, device=cache.device)
            entry = (Lf, Rf, k_rope, valid, n_cache, fp)
            _C8CACHE[key] = entry
        return entry[:5]


class LocksMLAMetadataBuilder(TritonMLAMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER


class LocksMLABackend(TritonMLABackend):
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_impl_cls():
        return LocksMLAImpl

    @staticmethod
    def get_builder_cls():
        return LocksMLAMetadataBuilder

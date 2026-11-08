from __future__ import annotations
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from ..attention.decode import ResidentVSource
from . import _runtime

_NOROPE = False
_ANNOUNCED = False
_DELEG_SAID = False
_ACT = {
    "decode_rows_sparse": 0,
    "decode_rows_dense": 0,
    "dense_reasons": {},
    "decode_rows_stock_proj": 0,
}


def get_activation():
    s = _ACT["decode_rows_sparse"]
    d = _ACT["decode_rows_dense"]
    if s + d == 0:
        return None
    return {
        "sparse": s,
        "dense": d,
        "frac": s / (s + d),
        "dense_reasons": dict(_ACT["dense_reasons"]),
        "stock_proj": _ACT["decode_rows_stock_proj"],
    }


def _locks_activation_report():
    a = get_activation()
    if a is None:
        return
    ok = a["frac"] >= 1.0
    print(
        f"[locks] SPARSE ACTIVATION {a['sparse']}/{a['sparse'] + a['dense']} decode rows = {100 * a['frac']:.2f}%  [{('OK' if ok else 'INVALID -- part of this cell is FullKV')}]",
        flush=True,
    )
    if a["dense_reasons"]:
        print(
            f"[locks]   decode rows served DENSE by gate: {a['dense_reasons']}",
            flush=True,
        )
    if a["stock_proj"]:
        print(
            f"[locks]   ROPE-ABSORBING lane INACTIVE for {a['stock_proj']}/{a['sparse']} sparse decode rows (their step had nt>1, so the single-row co op ran the stock fused projection + rope). Correct, and fully sparse -- but those rows did NOT exercise the CO/FIX-C kernels.",
            flush=True,
        )


__import__("atexit").register(_locks_activation_report)


class LocksAttentionImpl(FlashAttentionImpl):
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        m = attn_metadata
        cfg = _runtime.get_config()
        if cfg is not None and cfg.is_mem:
            assert _runtime.mem_stageb_wired(), (
                "locks mem: _MEM_STAGEB_WIRED is False -- the mem variant has no feature-off decode path (set it True, or run LOCKS_DISABLE=1 for the stock-FA/FullKV line)"
            )
            return self._forward_mem(
                layer,
                query,
                key,
                value,
                kv_cache,
                m,
                output,
                output_scale,
                output_block_scale,
            )
        st = getattr(m, "locks", None) if m is not None else None
        _reasons = (
            ("no_cfg", cfg is None),
            ("variant!=fast", cfg is not None and cfg.variant != "fast"),
            ("no_state", st is None),
            ("no_metadata", m is None),
            ("output_is_None", output is None),
            ("max_query_len!=1", m is not None and m.max_query_len != 1),
            ("kv_cache_empty", kv_cache.numel() == 0),
            ("use_cascade", bool(getattr(m, "use_cascade", False))),
            ("not_causal", m is not None and m.causal is not True),
            ("attn_type!=DECODER", self.attn_type != AttentionType.DECODER),
            ("sliding_window", self.sliding_window != (-1, -1)),
            ("alibi", self.alibi_slopes is not None),
            ("sinks", self.sinks is not None),
            ("dcp>1", getattr(self, "dcp_world_size", 1) > 1),
            ("quantized_kv", bool(is_quantized_kv_cache(self.kv_cache_dtype))),
        )
        if any((v for (_, v) in _reasons)):
            if (
                st is None
                and cfg is not None
                and (cfg.variant == "fast")
                and (m is not None)
                and (output is not None)
                and (m.max_query_len == 1)
                and (kv_cache.numel() != 0)
            ):
                global _DELEG_SAID
                if not _DELEG_SAID:
                    _DELEG_SAID = True
                    print(
                        "[locks] FALLBACK no-state: pure-decode step with NO Stage-A state -> stock FA (FullKV) for this step. If this line appears after boot the cell is NOT measuring locks.",
                        flush=True,
                    )
            import torch as _t

            _only_mql = (
                st is not None
                and m is not None
                and (output is not None)
                and (kv_cache.numel() != 0)
                and (cfg is not None)
                and (cfg.variant == "fast")
                and (m.max_query_len != 1)
                and all((not v for (n, v) in _reasons if n != "max_query_len!=1"))
                and (not _t.cuda.is_current_stream_capturing())
            )
            if _only_mql:
                return self._forward_fast_batchsplit(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    m,
                    output,
                    st,
                    cfg,
                    output_scale,
                    output_block_scale,
                )
            if m is not None and kv_cache.numel() != 0 and (output is not None):
                qsl = getattr(m, "query_start_loc", None)
                if qsl is not None:
                    try:
                        n_dec = int((__import__("torch").diff(qsl) == 1).sum())
                    except Exception:
                        n_dec = 0
                    if n_dec:
                        _ACT["decode_rows_dense"] += n_dec
                        key = ",".join((n for (n, v) in _reasons if v))
                        _ACT["dense_reasons"][key] = (
                            _ACT["dense_reasons"].get(key, 0) + n_dec
                        )
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                m,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        (_State, _derive, select_pages, _graph_safe) = _runtime.resolve_selection()
        n_req = m.seq_lens.shape[0]
        _ACT["decode_rows_sparse"] += n_req
        out3 = (
            output
            if output.dim() == 3
            else output.view(output.shape[0], self.num_heads, -1)
        )
        st_use = st
        select_pages(
            query, kv_cache, m.block_table, m.seq_lens, st_use, n_req, self.scale, cfg
        )
        _runtime.decode_dispatch(
            query,
            kv_cache,
            m.block_table,
            m.seq_lens,
            st_use,
            out3,
            ResidentVSource(kv_cache),
            self.scale,
            cfg,
        )
        global _ANNOUNCED
        if not _ANNOUNCED:
            _ANNOUNCED = True
            print(
                f"[locks] fast decode path ACTIVE (n_req={n_req} graph_safe={_graph_safe} budget={cfg.budget} coverage={cfg.coverage})",
                flush=True,
            )
        return output

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        cfg = _runtime.get_config()
        if cfg is None or not cfg.is_mem or (not _runtime.mem_stageb_wired()):
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        n_tokens = slot_mapping.shape[0]
        if n_tokens == 0 or kv_cache.numel() == 0:
            return
        tier = _runtime.get_tier()
        if tier is None:
            if _runtime.in_profiling():
                return
            if int(slot_mapping.max().item()) < 0:
                return
            raise RuntimeError("locks mem: KV write before the tier was built")
        lidx = _runtime.tier_layer_index(tier, kv_cache)
        if cfg.mem_k_only:
            tier.write_kv_konly(lidx, key, value, kv_cache, slot_mapping)
        else:
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
            tier.write_kv(lidx, value, n_tokens, key=key if cfg.offload_k else None)

    def _forward_fast_batchsplit(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        m,
        output,
        st,
        cfg,
        output_scale,
        output_block_scale,
    ):
        import torch

        assert not torch.cuda.is_current_stream_capturing(), (
            "locks fast: batch-split forward under CUDA-graph capture"
        )
        super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            m,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
        n_req = m.seq_lens.shape[0]
        qsl_cpu = getattr(m, "locks_qsl_cpu", None)
        sl_cpu = getattr(m, "locks_sl_cpu", None)
        if qsl_cpu is None or sl_cpu is None:
            qsl_cpu = m.query_start_loc[: n_req + 1].cpu()
            sl_cpu = m.seq_lens[:n_req].cpu()
        ql = (qsl_cpu[1 : n_req + 1] - qsl_cpu[:n_req]).tolist()
        sl = sl_cpu[:n_req].tolist()
        q0s = qsl_cpu[:n_req].tolist()
        dec_rows = [r for r in range(n_req) if ql[r] == 1 and sl[r] > 0]
        if not dec_rows:
            return output
        dev = query.device
        H = self.num_heads
        nt = m.num_actual_tokens
        qsrc = query
        q3 = qsrc[:nt] if qsrc.dim() == 3 else qsrc[:nt].view(nt, H, -1)
        out3 = output[:nt] if output.dim() == 3 else output[:nt].view(nt, H, -1)
        _bs = getattr(m, "_locks_fast_bs_idx", None)
        if _bs is None:
            _bs = {
                "rows": torch.tensor(dec_rows, dtype=torch.long, device=dev),
                "tok": torch.tensor(
                    [q0s[r] for r in dec_rows], dtype=torch.long, device=dev
                ),
            }
            m._locks_fast_bs_idx = _bs
        (rows_dev, tok_dev) = (_bs["rows"], _bs["tok"])
        n_dec = len(dec_rows)
        q_dec = q3.index_select(0, tok_dev)
        bt_dec = m.block_table.index_select(0, rows_dev)
        sl_dec = m.seq_lens.index_select(0, rows_dev)
        out_dec = torch.empty_like(q_dec)
        from ..selection import derive_page_params

        (_State, _derive, select_pages, _gs) = _runtime.resolve_selection()
        derive_page_params(st, sl_dec, n_dec)
        select_pages(q_dec, kv_cache, bt_dec, sl_dec, st, n_dec, self.scale, cfg)
        _runtime.decode_dispatch(
            q_dec,
            kv_cache,
            bt_dec,
            sl_dec,
            st,
            out_dec,
            ResidentVSource(kv_cache),
            self.scale,
            cfg,
        )
        out3.index_copy_(0, tok_dev, out_dec)
        _ACT["decode_rows_sparse"] += n_dec
        return output

    def _forward_mem(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        m,
        output,
        output_scale,
        output_block_scale,
    ):
        assert _runtime.mem_stageb_wired(), "locks mem path reached with the flag off"
        cfg = _runtime.get_config()
        st = getattr(m, "locks", None) if m is not None else None
        tier = getattr(m, "locks_tier", None) if m is not None else None
        if _runtime.in_profiling():
            return output
        if (
            m is None
            or output is None
            or kv_cache.numel() == 0
            or getattr(m, "use_cascade", False)
            or (m.causal is not True)
            or (self.attn_type != AttentionType.DECODER)
            or (self.sliding_window != (-1, -1))
            or (self.alibi_slopes is not None)
            or (self.sinks is not None)
            or (getattr(self, "dcp_world_size", 1) > 1)
            or is_quantized_kv_cache(self.kv_cache_dtype)
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                m,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        if st is None or tier is None:
            raise RuntimeError(
                "locks mem: forward reached with locks state/tier missing (st=%r tier=%r) -- the boot-time VRAM plan (memplan) or the builder failed; refusing a silent dense fallback"
                % (st is not None, tier is not None)
            )
        if m.max_query_len != 1:
            return self._forward_mem_batchsplit(
                layer, query, key, value, kv_cache, m, output, st, tier, cfg
            )
        from ..tier import mem_dynamic_decode

        lidx = _runtime.tier_layer_index(tier, kv_cache)
        n_req = m.seq_lens.shape[0]
        out3 = (
            output
            if output.dim() == 3
            else output.view(output.shape[0], self.num_heads, -1)
        )
        (_State, _derive, select_pages, _gs) = _runtime.resolve_selection()
        import os as _os

        _dbg = _os.environ.get("LOCKS_MEM_DEBUG") == "1"
        side_done = False
        st_use = st
        select_pages(
            query, kv_cache, m.block_table, m.seq_lens, st_use, n_req, self.scale, cfg
        )
        if _dbg:
            import torch as _t

            _t.cuda.synchronize()
            print(
                f"[dbg] L{lidx} select OK cnt0={st_use.page_cnt[0].tolist()} sl={m.seq_lens[:2].tolist()} nreq={n_req}",
                flush=True,
            )
        mem_dynamic_decode(
            query,
            kv_cache,
            m.block_table,
            m.seq_lens,
            st_use,
            out3,
            tier,
            lidx,
            scale=self.scale,
            fetch="kernel",
            impl="auto",
            tier_stepped=side_done,
        )
        if _dbg:
            import torch as _t

            _t.cuda.synchronize()
            print(f"[dbg] L{lidx} mem_decode OK", flush=True)
        global _ANNOUNCED
        if not _ANNOUNCED:
            _ANNOUNCED = True
            print(
                f"[locks] MEM decode path ACTIVE (variant={cfg.variant} n_req={n_req} k_only={cfg.mem_k_only} budget={cfg.budget} coverage={cfg.coverage})",
                flush=True,
            )
        return output

    def _forward_mem_batchsplit(
        self, layer, query, key, value, kv_cache, m, output, st, tier, cfg
    ):
        import torch

        assert not torch.cuda.is_current_stream_capturing(), (
            "locks mem: batch-split (prefill/mixed) forward under CUDA-graph capture -- capture batches must be uniform single-token decode"
        )
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        qsl_cpu = getattr(m, "locks_qsl_cpu", None)
        sl_cpu = getattr(m, "locks_sl_cpu", None)
        n_req = m.seq_lens.shape[0]
        if qsl_cpu is None or sl_cpu is None:
            qsl_cpu = m.query_start_loc[: n_req + 1].cpu()
            sl_cpu = m.seq_lens[:n_req].cpu()
        ql = (qsl_cpu[1 : n_req + 1] - qsl_cpu[:n_req]).tolist()
        sl = sl_cpu[:n_req].tolist()
        q0s = qsl_cpu[:n_req].tolist()
        lidx = _runtime.tier_layer_index(tier, kv_cache)
        H = self.num_heads
        nt = m.num_actual_tokens
        q3 = query[:nt] if query.dim() == 3 else query[:nt].view(nt, H, -1)
        q3_dec_src = q3
        out3 = output[:nt] if output.dim() == 3 else output[:nt].view(nt, H, -1)
        dev = query.device
        lease = getattr(m, "locks_lease", None)
        lease_row = int(getattr(m, "locks_lease_row", -1))
        dec_rows = [r for r in range(n_req) if ql[r] == 1 and sl[r] > 0]
        pre_rows = [r for r in range(n_req) if ql[r] > 1]
        _bs = getattr(m, "_locks_bs_idx", None)
        if _bs is None:
            _bs = {
                "cu_q": {
                    r: torch.tensor([0, ql[r]], dtype=torch.int32, device=dev)
                    for r in pre_rows
                },
                "cu_k": {
                    r: torch.tensor([0, sl[r]], dtype=torch.int32, device=dev)
                    for r in pre_rows
                    if r == lease_row
                },
                "rows_dev": torch.tensor(dec_rows, dtype=torch.long, device=dev)
                if dec_rows
                else None,
                "tok_dev": torch.tensor(
                    [q0s[r] for r in dec_rows], dtype=torch.long, device=dev
                )
                if dec_rows
                else None,
            }
            m._locks_bs_idx = _bs
        for r in pre_rows:
            (t0, t1) = (sl[r] - ql[r], sl[r])
            (q0, q1) = (q0s[r], q0s[r] + ql[r])
            cu_q = _bs["cu_q"][r]
            if r == lease_row:
                lease.fill_layer(lidx, m.block_table[r])
                cu_k = _bs["cu_k"][r]
                flash_attn_varlen_func(
                    q=q3[q0:q1],
                    k=lease.k[lidx][:t1],
                    v=lease.v[lidx][:t1],
                    out=out3[q0:q1],
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=ql[r],
                    max_seqlen_k=t1,
                    softmax_scale=self.scale,
                    causal=True,
                    fa_version=self.vllm_flash_attn_version,
                )
            elif t0 == 0:
                flash_attn_varlen_func(
                    q=q3[q0:q1],
                    k=key[q0:q1],
                    v=value[q0:q1],
                    out=out3[q0:q1],
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_q,
                    max_seqlen_q=ql[r],
                    max_seqlen_k=ql[r],
                    softmax_scale=self.scale,
                    causal=True,
                    fa_version=self.vllm_flash_attn_version,
                )
            else:
                raise RuntimeError(
                    f"locks mem: continuation prefill row {r} (ctx={t0}) has no lease (lease_row={lease_row}) -- builder lease plan bug or max_num_partial_prefills > 1"
                )
        if dec_rows:
            from ..selection import derive_page_params
            from ..tier import mem_dynamic_decode

            rows_dev = _bs["rows_dev"]
            tok_dev = _bs["tok_dev"]
            n_dec = len(dec_rows)
            q_dec = q3_dec_src.index_select(0, tok_dev)
            bt_dec = m.block_table.index_select(0, rows_dev)
            sl_dec = m.seq_lens.index_select(0, rows_dev)
            out_dec = torch.empty_like(q_dec)
            (_State, _derive, select_pages, _gs) = _runtime.resolve_selection()
            derive_page_params(st, sl_dec, n_dec)
            select_pages(q_dec, kv_cache, bt_dec, sl_dec, st, n_dec, self.scale, cfg)
            mem_dynamic_decode(
                q_dec,
                kv_cache,
                bt_dec,
                sl_dec,
                st,
                out_dec,
                tier,
                lidx,
                scale=self.scale,
                fetch="kernel",
                impl="auto",
            )
            out3.index_copy_(0, tok_dev, out_dec)
        return output

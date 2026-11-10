from __future__ import annotations
import os
import traceback
import torch

_NVTX = os.environ.get("LOCKS_NVTX", "0") == "1"
_nvtx_push = torch.cuda.nvtx.range_push
_nvtx_pop = torch.cuda.nvtx.range_pop
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
from ..attention.decode import _split_kv, ensure_stage_b_buffers
from . import _runtime


class _TierKLayer:
    __slots__ = ("_tier", "_lidx")

    def __init__(self, tier, lidx: int):
        (self._tier, self._lidx) = (tier, lidx)

    @property
    def shape(self):
        t = self._tier
        return (t.NB, t.page, t.n_kv, t.d)

    @property
    def device(self):
        return self._tier.device

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if (
                len(idx) == 4
                and isinstance(idx[1], int)
                and (idx[2] == slice(None))
                and isinstance(idx[3], slice)
                and (idx[3].start in (None, 0))
                and (idx[3].step in (None, 1))
                and isinstance(idx[3].stop, int)
                and (0 <= idx[1] < self._tier.page)
                and (0 <= idx[3].stop <= self._tier.d)
            ):
                return self._tier.gather_k_tags(self._lidx, idx[0], idx[1], idx[3].stop)
            pages = self._tier.gather_k_blocks(self._lidx, idx[0])
            return pages[(slice(None),) + tuple(idx[1:])]
        return self._tier.gather_k_blocks(self._lidx, idx)


def _build_record_arena(layer_kv, L: int):
    t0 = layer_kv[0]
    per = t0.numel()
    per_bytes = per * t0.element_size()
    base = t0.data_ptr()
    storage = t0.untyped_storage()
    ok = storage.nbytes() >= L * per_bytes
    for i, t in enumerate(layer_kv):
        if (
            t.numel() != per
            or not t.is_contiguous()
            or t.data_ptr() != base + i * per_bytes
            or (t.untyped_storage().data_ptr() != storage.data_ptr())
        ):
            ok = False
            break
    if not ok:
        print(
            "[locks] summary_cache: engine KV layers are SEPARATE storages -> QuadState keeps a separate record arena (aliasing fold-in deferred; K still leaves VRAM)",
            flush=True,
        )
        return None
    full = t0.new_empty(0)
    full.set_(storage, t0.storage_offset(), (L * per,))
    return full


def _seq_lens_host(cam):
    sl = getattr(cam, "seq_lens_cpu_upper_bound", None)
    if sl is None:
        sl = getattr(cam, "_seq_lens_cpu", None)
    return sl


_SPLIT = 128
_OV_DELAY_MS = float("0")
_OV_DELAY_CYCLES = int(0.0 * 0.001 * 1500000000.0)


class _PrefillOverlap:
    __slots__ = (
        "_dev",
        "_stream",
        "_event",
        "_pending",
        "_plan",
        "_seen",
        "_hooked",
        "_impls",
        "_write_fn",
        "_layer_k",
        "_state",
        "_lg",
        "_grouped",
        "n_steps",
        "n_fires",
        "n_pages_built",
        "n_rebound",
    )

    def __init__(self, device):
        self._dev = device
        self._stream = torch.cuda.Stream(device=device)
        self._event = torch.cuda.Event()
        self._pending = False
        self._plan = None
        self._seen = None
        self._hooked = False
        self._impls = []
        self._write_fn = None
        self._layer_k = None
        self._state = None
        self._lg = 1
        self._grouped = True
        self.n_steps = 0
        self.n_fires = 0
        self.n_pages_built = 0
        self.n_rebound = 0

    def bind(self, state, layer_k, write_fn):
        (self._state, self._layer_k, self._write_fn) = (state, layer_k, write_fn)
        self._seen = [False] * len(layer_k)

    def install(self, sfc, layer_names) -> int:
        if self._hooked:
            return len(self._impls) + self.n_rebound
        for lidx, name in enumerate(layer_names):
            mod = sfc.get(name)
            impl = getattr(mod, "impl", None)
            if impl is None:
                continue
            if getattr(impl, "_locks_ov_wrapped", False):
                impl._locks_ov_target = self
                self.n_rebound += 1
                continue
            orig = impl.forward

            def _wrapped(*args, _orig=orig, _l=lidx, _impl=impl, **kw):
                out = _orig(*args, **kw)
                tgt = getattr(_impl, "_locks_ov_target", None)
                if tgt is not None:
                    tgt._layer_done(_l)
                return out

            impl.forward = _wrapped
            impl._locks_ov_wrapped = True
            impl._locks_ov_target = self
            self._impls.append((impl, orig))
        self._hooked = True
        return len(self._impls) + self.n_rebound

    def uninstall(self) -> None:
        for impl, orig in self._impls:
            try:
                del impl.forward
            except AttributeError:
                impl.forward = orig
            impl._locks_ov_wrapped = False
            impl._locks_ov_target = None
        self._impls = []
        self._hooked = False

    def plan(self, blocks) -> None:
        self._plan = blocks
        P = int(blocks.shape[0])
        self._grouped = P <= 8192
        self._lg = max(1, 8192 // P) if self._grouped else 1
        if self._seen is not None:
            for i in range(len(self._seen)):
                self._seen[i] = False
        self.n_steps += 1

    def disarm(self) -> None:
        self._plan = None

    @property
    def armed(self) -> bool:
        return self._plan is not None

    def _layer_done(self, lidx: int) -> None:
        blocks = self._plan
        if blocks is None:
            return
        seen = self._seen
        L = len(seen)
        if lidx >= L or seen[lidx]:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        seen[lidx] = True
        lg = self._lg
        l0 = lidx // lg * lg
        end = min(l0 + lg, L)
        if lidx != end - 1:
            return
        if _NVTX:
            _nvtx_push("locks.ov_build")
        side = self._stream
        side.wait_stream(torch.cuda.current_stream())
        from ..selection.build import _scatter_index, _stack_key_pages

        (st, write_fn) = (self._state, self._write_fn)
        P = blocks.shape[0]
        with torch.cuda.stream(side):
            if self._grouped:
                Ks = self._layer_k[l0:end]
                Kp = _stack_key_pages(Ks, blocks)
                (li, bi) = _scatter_index(len(Ks), l0, blocks)
                write_fn(st, Kp.reshape(len(Ks) * P, st.page, st.n_kv, st.d), li, bi)
            else:
                K = self._layer_k[l0]
                for s0 in range(0, P, 8192):
                    bs_ = blocks[s0 : s0 + 8192]
                    (li, bi) = _scatter_index(1, l0, bs_)
                    write_fn(st, K[bs_], li, bi)
        blocks.record_stream(side)
        self._event.record(side)
        self._pending = True
        self.n_fires += 1
        self.n_pages_built += P * (end - l0)
        if _NVTX:
            _nvtx_pop()

    def fence(self) -> bool:
        if not self._pending:
            return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "locks prefill-overlap: side-stream build pending while the current stream is CAPTURING -- a plan leaked into a cudagraph capture batch. Refusing to pollute the graph."
            )
        self._pending = False
        torch.cuda.current_stream().wait_event(self._event)
        return True

    def drain(self) -> None:
        if self._pending and (not torch.cuda.is_current_stream_capturing()):
            self._stream.synchronize()
            self._pending = False


def _host_prev_cur(cam, n_req):
    qs = getattr(cam, "query_start_loc_cpu", None)
    if qs is None or qs.shape[0] < n_req + 1:
        return (None, None)
    qlen = (qs[1 : n_req + 1] - qs[:n_req]).to(torch.int64)
    sl = _seq_lens_host(cam)
    if sl is not None and sl.shape[0] >= n_req:
        cur = sl[:n_req].to(torch.int64)
        return ((cur - qlen).clamp_min(0), cur)
    nct = getattr(cam, "_num_computed_tokens_cpu", None)
    if nct is not None and nct.shape[0] >= n_req:
        prev = nct[:n_req].to(torch.int64)
        return (prev, prev + qlen)
    return (None, None)


def _finalized_block_plan(block_table, prev, cur, n_req, page, device):
    lo = prev // page
    hi = cur // page
    cnt = (hi - lo).clamp_min_(0)
    total = int(cnt.sum())
    if total == 0:
        return None
    rows = torch.repeat_interleave(torch.arange(n_req, dtype=torch.int64), cnt)
    starts = torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    pages = torch.repeat_interleave(lo, cnt) + (torch.arange(total) - starts)
    stride = block_table.shape[1]
    flat = (rows * stride + pages).to(device, non_blocking=True)
    return block_table.reshape(-1)[flat].to(torch.int64)


class LocksMetadataBuilder(FlashAttentionMetadataBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        cfg = _runtime.get_config()
        (_State, _derive, _select, graph_safe) = _runtime.resolve_selection()
        if (
            cfg is not None
            and graph_safe
            and (not cfg.force_eager)
            and (cfg.variant == "fast" or cfg.is_mem)
        ):
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._cfg = _runtime.get_config()
        self._device = device
        self._vllm_config = vllm_config
        self._layer_names = list(layer_names)
        self._state = None
        self._derive = None
        self._r8 = False
        self._qwen_decode_fast_build = False
        self._capturing = False
        self._layer_kv = None
        self._layer_k = None
        self._prev_seq_lens = None
        self._prev_settled = None
        self._built_once = False
        self._tier = None
        self._tier_tried = False
        self._ov = None
        self._lease = None
        self._rows_pin = None
        self._rows_dev = None
        self._rows_par = 0
        if self._cfg is not None and self._cfg.is_mem:
            sched = vllm_config.scheduler_config
            mpp = int(getattr(sched, "max_num_partial_prefills", 1) or 1)
            if mpp != 1:
                raise RuntimeError(
                    f"locks mem: max_num_partial_prefills={mpp} unsupported (the prefill lease covers exactly one mid-prefill request)"
                )
            if bool(getattr(vllm_config.cache_config, "enable_prefix_caching", False)):
                raise RuntimeError(
                    "locks mem: enable_prefix_caching is unsupported (a cache hit resumes with prior context the lease never saw)"
                )
        if self._cfg is None:
            return
        if self._cfg.is_mem:
            print(
                "[locks] mem tier WIRED; Tier + Stage-A state deferred to first build (num_gpu_blocks not set yet)",
                flush=True,
            )
        (State, derive, _select, graph_safe) = _runtime.resolve_selection()
        self._r8 = graph_safe
        self._derive = derive
        self._qwen_decode_fast_build = bool(
            graph_safe and not self._cfg.is_mem and self._cfg.r8_rank == 4
            and vllm_config.scheduler_config.max_num_seqs == 1
            and self.num_heads_q == 32 and self.num_heads_kv == 8
            and self.headdim == 128 and self.block_size == 16
            and 16384 <= vllm_config.model_config.max_model_len <= 131072
        )
        if self._r8:
            print(
                "[locks] r8 selection WIRED (graph-safe); R8State + layer map deferred to first build (num_gpu_blocks not set yet)",
                flush=True,
            )
            return
        raise RuntimeError("LOCKS requires the graph-safe optimized selector")

    def _ensure_state(self):
        if self._state is not None or not self._r8:
            return self._state
        if _runtime.in_profiling():
            return None
        nb = self._vllm_config.cache_config.num_gpu_blocks
        if not nb:
            return None
        sfc = self._vllm_config.compilation_config.static_forward_context
        layer_kv = []
        ptr2layer = {}
        for lidx, name in enumerate(self._layer_names):
            mod = sfc.get(name)
            kvc = getattr(mod, "kv_cache", None) if mod is not None else None
            if not isinstance(kvc, torch.Tensor) or kvc.numel() == 0:
                return None
            layer_kv.append(kvc)
            ptr2layer[kvc.data_ptr()] = lidx
        from ..selection.rki4_state import Rki4State as _Rki4Geom

        _Rki4Geom.validate_geometry(
            head_dim=self.headdim,
            G=self.num_heads_q // self.num_heads_kv,
            page=self.block_size,
        )
        try:
            cfg = self._cfg
            from ..selection import Rki4State as _State
            from ..selection.rki4_state import RNK as rank

            n_kv = self.num_heads_kv
            G = self.num_heads_q // n_kv
            max_reqs = self._vllm_config.scheduler_config.max_num_seqs
            max_model_len = self._vllm_config.model_config.max_model_len
            page = self.block_size
            max_pages = (max_model_len + page - 1) // page
            budget = cfg.budget if cfg.budget is not None else 0.1
            extra_state = {}
            extra_state["combine"] = cfg.quad_combine
            extra_state["zsplit"] = getattr(cfg, "rki4_zsplit", None)
            st = _State(
                self._device,
                num_layers=len(self._layer_names),
                num_blocks=int(nb),
                n_kv=n_kv,
                G=G,
                head_dim=self.headdim,
                page=page,
                max_reqs=max_reqs,
                max_pages=max_pages,
                rank=rank,
                budget=budget,
                budget_pages=cfg.budget_pages,
                sink_pages=cfg.sink_pages,
                window_pages=cfg.window_pages,
                resident_low_context=not cfg.is_mem,
                **extra_state,
            )
            st.D = st.d
            st.scale = None
            ensure_stage_b_buffers(st, self._device, 128)
            st._layer_of = ptr2layer
            self._layer_kv = layer_kv
            if getattr(cfg, "mem_summary_cache", False):
                self._layer_k = None
            else:
                self._layer_k = [_split_kv(kvc)[0] for kvc in layer_kv]

            self._state = st
            if st.low_context and os.environ.get("LOCKS_DEBUG_GATE", "0") != "1":
                self._maybe_refresh = self._refresh_single
            if not getattr(cfg, "mem_summary_cache", False):
                self._ensure_overlap(sfc)
            print(
                f"[locks] {_State.__name__} ALLOCATED (score=rki4): layers={st.L} num_blocks={st.NB} reqs={max_reqs} n_kv={n_kv} G={G} d={self.headdim} page={page} max_pages={max_pages} rank={st.r} v_bits={getattr(st, 'v_bits', '-')} combine={getattr(st, 'combine', '-')} coords={getattr(st, 'coords', '-')} rec={(getattr(st, 'rec_bytes', 0) if getattr(st, 'rec', None) is not None else 0)} budget={budget} sink={cfg.sink_pages} window={cfg.window_pages} split={128} selector_MiB={st.bytes_per_layer() * st.L / 2**20:.1f}",
                flush=True,
            )
        except Exception as e:
            print(
                "[locks] R8State alloc FAILED -- refusing the silent stock-FA/FullKV delegation:",
                flush=True,
            )
            traceback.print_exception(type(e), e, e.__traceback__)
            raise RuntimeError(
                f"locks: Stage-A state allocation failed ({type(e).__name__}: {e}). There is no FullKV fallback under the locks tag -- lower gpu_memory_utilization / fix the geometry, or run the reference line with LOCKS_DISABLE=1."
            ) from e
        return self._state

    def _ensure_overlap(self, sfc) -> None:
        if self._ov is not None:
            return
        try:
            from ..selection.rki4_build import _rki4_write as write_fn

            ov = _PrefillOverlap(self._device)
            ov.bind(self._state, self._layer_k, write_fn)
            n = ov.install(sfc, self._layer_names)
            if n == 0:
                print(
                    "[locks] prefill overlap: NO layer hooks installed -> serialized first-decode build",
                    flush=True,
                )
                return
            self._ov = ov
            print(
                f"[locks] prefill overlap ACTIVE: side-stream summary build hooked on {n}/{len(self._layer_names)} layers (rebound={ov.n_rebound}, delay_ms={0.0} fence={'on'})",
                flush=True,
            )
        except Exception as e:
            print(
                "[locks] prefill overlap install FAILED (serialized build):", flush=True
            )
            traceback.print_exception(type(e), e, e.__traceback__)
            self._ov = None

    def _ov_schedule(self, md, cam, st) -> None:
        ov = self._ov
        if ov is None or st is None or self._capturing:
            if ov is not None:
                ov.disarm()
            return
        n_req = int(getattr(cam, "num_reqs", 0) or 0)
        n_req = min(n_req, md.block_table.shape[0])
        if n_req <= 0:
            ov.disarm()
            return
        (prev, cur) = _host_prev_cur(cam, n_req)
        if prev is None:
            ov.disarm()
            return
        try:
            blocks = _finalized_block_plan(
                md.block_table, prev, cur, n_req, self.block_size, self._device
            )
        except Exception as e:
            print(
                "[locks] prefill overlap plan FAILED (this step falls back to the first-decode build):",
                flush=True,
            )
            traceback.print_exception(type(e), e, e.__traceback__)
            ov.disarm()
            return
        if blocks is None:
            ov.disarm()
            return
        ov.plan(blocks)
        self._built_once = True

    def _ensure_tier(self):
        if self._tier is not None or self._tier_tried:
            return self._tier
        if _runtime.in_profiling():
            return None
        nb = self._vllm_config.cache_config.num_gpu_blocks
        if not nb:
            return None
        sfc = self._vllm_config.compilation_config.static_forward_context
        layer_kv = []
        ptr2layer = {}
        for lidx, name in enumerate(self._layer_names):
            mod = sfc.get(name)
            kvc = getattr(mod, "kv_cache", None) if mod is not None else None
            if not isinstance(kvc, torch.Tensor) or kvc.numel() == 0:
                return None
            layer_kv.append(kvc)
            ptr2layer[kvc.data_ptr()] = lidx
        self._tier_tried = True
        try:
            from ..tier import Tier

            cfg = self._cfg
            sched = self._vllm_config.scheduler_config
            comp = self._vllm_config.compilation_config
            max_reqs = sched.max_num_seqs
            max_model_len = self._vllm_config.model_config.max_model_len
            page = self.block_size
            max_pages = (max_model_len + page - 1) // page
            max_tokens = (
                max(
                    int(getattr(sched, "max_num_batched_tokens", 0) or 0),
                    int(getattr(comp, "max_cudagraph_capture_size", 0) or 0),
                    max_reqs,
                )
                + 8
            )
            dtype = layer_kv[0].dtype
            self._tier = Tier.from_config(
                cfg,
                num_layers=len(self._layer_names),
                num_blocks=int(nb),
                n_kv=self.num_heads_kv,
                page=page,
                d=self.headdim,
                max_reqs=max_reqs,
                max_pages=max_pages,
                max_tokens=max_tokens,
                dtype=dtype,
                device=self._device,
            )
            self._tier._layer_of = ptr2layer
            self._layer_kv = layer_kv
            _runtime.set_tier(self._tier)
            print(
                f"[locks] Tier ALLOCATED ({cfg.variant}): {self._tier.bytes_report()}",
                flush=True,
            )
        except Exception as e:
            print("[locks] Tier alloc FAILED (mem-v -> stock FA):", flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            self._tier = None
        return self._tier

    def _ensure_lease(self):
        if self._lease is not None:
            return self._lease
        from ..tier.lease import PrefillLease

        page = self.block_size
        max_model_len = self._vllm_config.model_config.max_model_len
        t = (max_model_len + page - 1) // page * page
        self._lease = PrefillLease(self._tier, t)
        print(
            f"[locks] prefill lease ALLOCATED: {self._lease.gib:.2f} GiB (K+V, {t} tokens, transient)",
            flush=True,
        )
        return self._lease

    def _lease_plan(self, md, cam, n_req) -> None:
        md.locks_lease = None
        md.locks_lease_row = -1
        qs = getattr(cam, "query_start_loc_cpu", None)
        sl = _seq_lens_host(cam)
        if qs is None or qs.shape[0] < n_req + 1 or sl is None or (sl.shape[0] < n_req):
            qs = md.query_start_loc[: n_req + 1].cpu()
            sl = md.seq_lens[:n_req].cpu()
        md.locks_qsl_cpu = qs
        md.locks_sl_cpu = sl
        ql = (qs[1 : n_req + 1] - qs[:n_req]).tolist()
        sls = sl[:n_req].tolist()
        cont = [r for r in range(n_req) if ql[r] > 1 and sls[r] - ql[r] > 0]
        fresh = [r for r in range(n_req) if ql[r] > 1 and sls[r] - ql[r] == 0]
        if len(cont) > 1:
            raise RuntimeError(
                f"locks mem: {len(cont)} continuation prefill rows in one step (max_num_partial_prefills must be 1)"
            )
        row = cont[0] if cont else max(fresh, key=lambda r: ql[r]) if fresh else -1
        if row < 0:
            return
        lease = self._ensure_lease()
        if not cont:
            lease.wm = 0
        lease.plan_step(sls[row] - ql[row], sls[row])
        md.locks_lease = lease
        md.locks_lease_row = row

    def build_for_cudagraph_capture(self, common_attn_metadata):
        if self._ov is not None:
            self._ov.disarm()
            self._ov.drain()
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if _NVTX:
            _nvtx_push("locks.fa_build")
        fast_build = fast_build or (
            self._qwen_decode_fast_build and common_attn_metadata.max_query_len == 1
        )
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if _NVTX:
            _nvtx_pop()
        if self._cfg is not None and self._cfg.is_mem:
            tier = self._ensure_tier()
            if tier is not None:
                tier.begin_step(
                    md.block_table,
                    md.seq_lens,
                    md.query_start_loc,
                    md.max_query_len,
                    md.slot_mapping.shape[0],
                    capture=self._capturing,
                )
                md.locks_tier = tier
        st = self._ensure_state()
        if st is None or md.max_query_len != 1:
            if (
                st is not None
                and (not self._capturing)
                and (self._cfg is not None)
                and (not self._cfg.is_mem)
            ):
                n_req_x = md.seq_lens.shape[0]
                qs_x = getattr(common_attn_metadata, "query_start_loc_cpu", None)
                sl_x = _seq_lens_host(common_attn_metadata)
                if (
                    qs_x is None
                    or qs_x.shape[0] < n_req_x + 1
                    or sl_x is None
                    or (sl_x.shape[0] < n_req_x)
                ):
                    qs_x = md.query_start_loc[: n_req_x + 1].cpu()
                    sl_x = md.seq_lens[:n_req_x].cpu()
                ql_x = qs_x[1 : n_req_x + 1] - qs_x[:n_req_x]
                if bool((ql_x == 1).any()):
                    md.locks_qsl_cpu = qs_x
                    md.locks_sl_cpu = sl_x
                    if self._ov is not None:
                        self._ov.disarm()
                        self._ov.fence()
                    if self._r8:
                        self._maybe_refresh(st, md, common_attn_metadata, n_req_x)
                    md.locks = st
            self._ov_schedule(md, common_attn_metadata, st)
            if (
                st is not None
                and self._cfg is not None
                and self._cfg.is_mem
                and (getattr(md, "locks_tier", None) is not None)
                and (not self._capturing)
            ):
                n_req = md.seq_lens.shape[0]
                if os.environ.get("LOCKS_MEM_PREFILL_REFRESH", "1") != "0":
                    self._refresh_prefill(st, md, common_attn_metadata, n_req)
                self._lease_plan(md, common_attn_metadata, n_req)
                md.locks = st
            return md
        if self._ov is not None:
            self._ov.disarm()
            self._ov.fence()
        if self._lease is not None and (not self._capturing):
            self._lease.release()
            self._lease = None
        n_req = md.seq_lens.shape[0]
        if self._r8:
            need_derive = True
            if not self._capturing:
                if _NVTX:
                    _nvtx_push("locks.refresh")
                need_derive = self._maybe_refresh(st, md, common_attn_metadata, n_req)
                if _NVTX:
                    _nvtx_pop()
            if need_derive:
                from ..selection import derive_page_params

                if _NVTX:
                    _nvtx_push("locks.derive")
                derive_page_params(st, md.seq_lens, n_req)
                if _NVTX:
                    _nvtx_pop()
        else:
            self._derive(md.seq_lens, st, self._cfg)
        md.locks = st
        return md

    def _maybe_refresh(self, st, md, cam, n_req):
        page = self.block_size
        sl_cpu = _seq_lens_host(cam)
        qs_cpu = getattr(cam, "query_start_loc_cpu", None)
        if sl_cpu is None or qs_cpu is None or qs_cpu.shape[0] < n_req + 1:
            self._prev_seq_lens = None
            self._prev_settled = None
            self._r8_refresh(st, md, n_req)
            return True
        cur = sl_cpu[:n_req]
        cur_st = cur - (qs_cpu[1 : n_req + 1] - qs_cpu[:n_req]).to(cur.dtype)
        cur_st = cur_st.clamp_(min=0)
        prev = self._prev_seq_lens
        prev_st = self._prev_settled
        full = True
        crossed = None
        if (
            prev is not None
            and prev.shape == cur.shape
            and (prev_st is not None)
            and (prev_st.shape == cur_st.shape)
        ):
            adv = (cur == prev + 1) & (cur_st == prev_st + 1)
            pad = (cur == 0) & (prev == 0)
            frozen = (cur == prev) & (cur_st == prev_st)
            if bool(torch.all(adv | pad | frozen)):
                full = False
                crossed = (cur_st % page == 0) & adv
        if os.environ.get("LOCKS_DEBUG_GATE", "0") == "1":
            print(
                f"[locks gate] full={full} crossed={(crossed.sum().item() if crossed is not None else '-')} n_req={n_req} cur={cur[:4].tolist()} prev={(prev[:4].tolist() if prev is not None else None)} cur_st={cur_st[:4].tolist()} prev_st={(prev_st[:4].tolist() if prev_st is not None else None)}",
                flush=True,
            )
        self._prev_seq_lens = cur.clone()
        self._prev_settled = cur_st.clone()
        need_derive = True
        if full:
            max_fin = int(cur_st.max().item()) // page
            settled = self._settled_dev(md, n_req)
            if self._built_once:
                self._delta_refresh(st, md, n_req, max_fin, settled=settled)
            else:
                self._bulk_refresh(st, md, n_req, max_fin, settled=settled)
                self._built_once = True
        else:
            fired = bool(crossed.any())
            if fired:
                rows = self._rows_to_device(torch.nonzero(crossed).squeeze(-1))
                self._tail_refresh(st, md, n_req, rows)
            need_derive = fired
        return need_derive

    def _refresh_single(self, st, md, cam, n_req):
        sl_cpu = _seq_lens_host(cam)
        qs_cpu = getattr(cam, "query_start_loc_cpu", None)
        if (n_req != 1 or sl_cpu is None or qs_cpu is None or qs_cpu.shape[0] < 2
                or sl_cpu.device.type != "cpu" or qs_cpu.device.type != "cpu"):
            self._single_previous = None
            return type(self)._maybe_refresh(self, st, md, cam, n_req)
        cur = int(sl_cpu[0])
        settled = max(0, cur - (int(qs_cpu[1]) - int(qs_cpu[0])))
        previous = getattr(self, "_single_previous", None)
        self._single_previous = (cur, settled)
        if previous is not None:
            old_cur, old_settled = previous
            advanced = cur == old_cur + 1 and settled == old_settled + 1
            padded = cur == old_cur == 0
            frozen = cur == old_cur and settled == old_settled
            if advanced or padded or frozen:
                crossed = advanced and settled % self.block_size == 0
                if crossed:
                    rows = getattr(self, "_single_row_cpu", None)
                    if rows is None:
                        rows = self._single_row_cpu = torch.tensor([0], dtype=torch.int64)
                    self._tail_refresh(st, md, n_req, self._rows_to_device(rows))
                return crossed
        settled_dev = self._settled_dev(md, n_req)
        if self._built_once:
            self._delta_refresh(st, md, n_req, settled // self.block_size, settled=settled_dev)
        else:
            self._bulk_refresh(st, md, n_req, settled // self.block_size, settled=settled_dev)
            self._built_once = True
        return True

    def _rows_to_device(self, idx):
        k = idx.shape[0]
        pin = self._rows_pin
        if pin is None or pin[0].shape[0] < k:
            cap = max(k, int(self._vllm_config.scheduler_config.max_num_seqs))
            self._rows_pin = pin = [
                torch.empty(cap, dtype=torch.int64, pin_memory=True) for _ in range(2)
            ]
            self._rows_dev = [
                torch.empty(cap, dtype=torch.int64, device=self._device)
                for _ in range(2)
            ]
        p = self._rows_par = self._rows_par ^ 1
        pin[p][:k].copy_(idx)
        self._rows_dev[p][:k].copy_(pin[p][:k], non_blocking=True)
        return self._rows_dev[p][:k]

    def _r8_refresh(self, st, md, n_req):
        if getattr(self._cfg, "mem_summary_cache", False):
            raise RuntimeError(
                "locks mem_summary_cache: refresh fallback without host seq lens reached -- the engine page holds records, not K; cannot build summaries from it"
            )
        from ..selection import rki4_build_refresh as _refresh

        bt = md.block_table
        sl = self._settled_dev(md, n_req)
        for lidx, kvc in enumerate(self._layer_kv):
            (K, _) = _split_kv(kvc)
            _refresh(st, lidx, K, bt, sl, n_req)

    def _k_layers(self):
        cfg = self._cfg
        if getattr(cfg, "mem_summary_cache", False):
            tier = _runtime.get_tier()
            if tier is None:
                raise RuntimeError(
                    "mem_summary_cache: summary build reached before the tier was built (no K source)"
                )
            return [_TierKLayer(tier, l) for l in range(len(self._layer_names))]
        return self._layer_k

    def _tail_refresh(self, st, md, n_req, rows):
        from ..selection import rki4_build_tail as _tail

        _tail(
            st,
            self._k_layers(),
            md.block_table,
            (md.seq_lens - 1).clamp(min=0),
            n_req,
            rows,
        )

    def _settled_dev(self, md, n_req):
        qsl = md.query_start_loc
        ql = qsl[1 : n_req + 1] - qsl[:n_req]
        return (md.seq_lens[:n_req] - ql).clamp(min=0)

    def _bulk_refresh(self, st, md, n_req, max_fin, settled=None):
        from ..selection import rki4_build_bulk as _bulk

        if settled is None:
            settled = (md.seq_lens - 1).clamp(min=0)
        _bulk(st, self._k_layers(), md.block_table, settled, n_req, max_fin)

    def _delta_refresh(self, st, md, n_req, max_fin, settled=None):
        from ..selection import rki4_build_delta as _delta

        if settled is None:
            settled = (md.seq_lens - 1).clamp(min=0)
        _delta(st, self._k_layers(), md.block_table, settled, n_req, max_fin)

    def _refresh_prefill(self, st, md, cam, n_req):
        (prev, _cur) = _host_prev_cur(cam, n_req)
        if prev is None:
            return
        max_fin = int(prev.max().item()) // self.block_size
        if max_fin <= 0:
            return
        qsl = md.query_start_loc
        ql_dev = qsl[1 : n_req + 1] - qsl[:n_req]
        settled = (md.seq_lens[:n_req] - ql_dev).clamp(min=0)
        from ..selection import rki4_build_bulk as _bulk, rki4_build_delta as _delta

        if self._built_once:
            _delta(st, self._k_layers(), md.block_table, settled, n_req, max_fin)
        else:
            _bulk(st, self._k_layers(), md.block_table, settled, n_req, max_fin)
            self._built_once = True

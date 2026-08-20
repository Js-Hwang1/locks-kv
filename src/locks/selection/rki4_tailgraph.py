"""LOCKS_TAIL_GRAPH -- CUDA-graph the bs=1 page-crossing tail refresh.

graphprof (results/nsys_gt_qf4, 2026-07-19) measured the eager tail-refresh
chain at 3.6-4.0 ms inter-graph wall on every 16th decode step (2.7-3.1 ms
GPU-idle, 176 launches): ~30 python-driven ATen ops across 40 layers plus
the cusolver eigh host sync.  At bs=1 the chain is fixed-shape (ONE tail
block x L layers x n_kv = L*n_kv 16x16 grams), so everything EXCEPT the
eigh captures into two CUDA graphs replayed with static buffers:

    [A: derive tail block from static (bt row, settled sl) copies ->
        gather K -> center -> grams]        eigh (cusolver, eager)
    [B: eigh consumers -> int4/int8 quantize -> slab scatters]

V2 (same session): the block-id derivation (``_tail_blocks`` math, n=1,
row 0) moved INSIDE graph A reading two static input buffers refreshed by
plain ``copy_`` per crossing -- the per-crossing eager work is now 2 copies
+ replay A + eigh + 2 copies + replay B (was ~7 extra eager launches).

Stage math = ``rki4_build._write_pre`` / ``_write_post`` -- the SAME
functions the eager path runs, so the replayed kernels are bit-identical
(standalone gates G1-G3 in scratch_qgemv/gate_tailgraph.py incl. the
changed-block replay; in-engine: suite both flag states + PTAB step-40
same-arm EXACT with the capture sentinel verified).

MEM-ARM K SOURCE: ``tier.gather_k_blocks`` is host-branchy (``.any()``
syncs, boolean-mask shapes, CPU ``index_select`` on the pinned pool) ->
NOT capturable.  The graph path uses a dual-source gather instead: read
BOTH the staged slot (``staging.k_pool[l][vbo]``, clamped) AND the pinned
page (``pool_k`` via a real int16 CUDA tensor aliasing the UVA device
pointer) and ``torch.where``-select on ``vbo >= 0`` -- pure device ops,
byte-identical output.  (The deployed mem-kv hands plain resident K-half
views, so this path is a guard for the spec-v2 records config.)

GUARDS (any miss -> caller's eager path, never wrong): n_req == 1, one
crossing row, stable block-table width, no active stream capture, not the
memory-profile pass; block ids are clamped in-graph (identity for real
tables, the ``_mask_blocks`` invariant); capture failure latches OFF
loudly once.  Rebuilds are idempotent, so a spurious extra build is
harmless by the same contract as the eager tail.
"""
from __future__ import annotations

import traceback

import torch

from .rki4_build import _eigh_chunked, _write_post, _write_pre

_WRAP_EXT = None
_INPROF = [None]          # resolved lazily; [None] -> unresolved


def _in_profiling() -> bool:
    # NO-FALLBACK RULE (2026-07-22): the import is NOT guarded.  A swallowed
    # ImportError silently disabled the profile-window guard (the graph would
    # then capture against the throwaway profiling cache).
    if _INPROF[0] is None:
        from ..backend._runtime import in_profiling
        _INPROF[0] = in_profiling
    return _INPROF[0]()


def _wrap_dev_ptr(ptr: int, shape, device: torch.device) -> torch.Tensor:
    """Real int16 CUDA tensor aliasing a device-mapped (UVA) host pool.

    ``pool.DevPtrTensor`` is a duck-typed pointer carrier (kernels only);
    torch advanced indexing needs a genuine tensor, so wrap the same device
    pointer via ``torch::from_blob`` (no copy, no ownership)."""
    global _WRAP_EXT
    if _WRAP_EXT is None:
        from torch.utils.cpp_extension import load_inline
        _WRAP_EXT = load_inline(
            name="locks_devptr_wrap",
            cpp_sources=r"""
#include <torch/extension.h>
torch::Tensor wrap_i16(int64_t ptr, std::vector<int64_t> shape,
                       int64_t dev) {
  auto opts = torch::TensorOptions().dtype(torch::kInt16)
      .device(torch::kCUDA, (int)dev);
  return torch::from_blob(reinterpret_cast<void*>(ptr), shape,
                          [](void*) {}, opts);
}
""",
            functions=["wrap_i16"], with_cuda=True, verbose=False)
    return _WRAP_EXT.wrap_i16(int(ptr), list(shape),
                              device.index if device.index is not None else 0)


class _TailGraph:
    """Lazy per-state two-graph replay engine (see module docstring)."""

    def __init__(self):
        self.ready = False
        self.dead = False

    # ---- in-graph tail-block derivation (V2) ------------------------------ #
    def _derive_blocks(self):
        """``_tail_blocks`` math for n=1, row 0, on the static copies; VALUES
        identical to the eager helper (then clamped, the _mask_blocks
        invariant).  Captured into graph A so replays re-read the statics."""
        fin1 = (self.sl_st.to(torch.int64) // self.page - 1).clamp_(min=0)
        blk = self.bt_st.to(torch.int64).gather(0, fin1)
        self.blocks.copy_(blk.clamp_(0, self.nb - 1))

    # ---- capturable gathers (byte-identical to the eager sources) -------- #
    def _gather_fast(self):
        Kp = torch.stack([K[self.blocks] for K in self._k_layers])
        return Kp.reshape(self.L, self.page, self.n_kv, self.d)

    def _gather_mem(self):
        b = self.blocks                                       # (1,) int64
        vb = self.vbo[b].to(torch.int64)                      # staging slot
        stg = (vb >= 0).view(1, 1, 1, 1, 1)
        ks = self.kpool[:, vb.clamp(min=0)]                   # (L,1,pg,kv,d)
        kp = self.kpin[:, b].view(torch.bfloat16)             # (L,1,kv,pg,d)
        kp = kp.permute(0, 1, 3, 2, 4)                        # -> (L,1,pg,kv,d)
        sel = torch.where(stg, ks, kp)                        # bit-select
        return sel.reshape(self.L, self.page, self.n_kv, self.d)

    def _stage_a(self, st):
        self._derive_blocks()
        return _write_pre(st, self._gather())

    # ---- lazy capture ----------------------------------------------------- #
    def _capture(self, st):
        mem = self._mem
        # warmup (side stream; executes the build -- idempotent by contract)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                tag, mu, dc, Gm = self._stage_a(st)
                S2, U = _eigh_chunked(Gm.reshape(-1, self.page, self.page))
                _write_post(st, mu, dc, S2, U, tag,
                            self.lidx, self.blocks.repeat(self.L))
        torch.cuda.current_stream().wait_stream(side)

        self.gA = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.gA):
            self.tag, self.mu, self.dc, self.Gm = self._stage_a(st)
        self.gA.replay()                     # materialize real grams
        self.Gm_flat = self.Gm.reshape(-1, self.page, self.page)
        S2, U = _eigh_chunked(self.Gm_flat)
        self.S2 = S2.clone()                 # static eigh landing
        self.U = U.clone()
        self.gB = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.gB):
            _write_post(st, self.mu, self.dc, self.S2, self.U, self.tag,
                        self.lidx, self.blocks.repeat(self.L))
        self.gB.replay()                     # perform THIS step's write
        self.ready = True
        print(f"[locks] TAIL_GRAPH captured (v2): {'mem' if mem else 'fast'} "
              f"L={self.L} page={self.page} n_kv={self.n_kv} nb={self.nb} "
              f"W={self.W}", flush=True)

    def _init_static(self, st, K_layers, block_table, seq_lens):
        dev = seq_lens.device
        self.L, self.page = st.L, st.page
        self.n_kv, self.d = st.n_kv, st.d
        self.nb = (st.pp_rec.shape[1]
                   if getattr(st, "pp_rec", None) is not None
                   else st.pp_v4.shape[1])
        self.W = int(block_table.shape[1])
        self.bt_st = torch.zeros(self.W, dtype=torch.int32, device=dev)
        self.sl_st = torch.zeros(1, dtype=torch.int32, device=dev)
        self.blocks = torch.zeros(1, dtype=torch.int64, device=dev)
        self.lidx = torch.arange(st.L, device=dev, dtype=torch.int64)
        self._mem = not torch.is_tensor(K_layers[0])
        if self._mem:
            tier = K_layers[0]._tier
            self.vbo = tier.staging.vbo
            self.kpool = tier.staging.k_pool
            self.kpin = _wrap_dev_ptr(tier.pool_k.dev_view.data_ptr(),
                                      tier.pool_k.host.shape, dev)
            self._src_key = id(tier)
            self._gather = self._gather_mem
        else:
            self._src_key = id(K_layers[0])
            self._k_layers = list(K_layers)
            self._gather = self._gather_fast

    # ---- per-crossing entry ------------------------------------------------ #
    def run(self, st, K_layers, block_table, seq_lens, n_req, rows):
        if self.dead or n_req != 1 or len(K_layers) == 0:
            return None
        if rows is not None and rows.numel() != 1:
            return None
        try:
            if torch.cuda.is_current_stream_capturing() or _in_profiling():
                return None
            if not self.ready:
                self._init_static(st, K_layers, block_table, seq_lens)
                self.bt_st.copy_(block_table[0, :self.W])
                self.sl_st.copy_(seq_lens[0:1])
                self._capture(st)
                return st.L
            src = (id(K_layers[0]) if not self._mem
                   else id(K_layers[0]._tier))
            if src != self._src_key or int(block_table.shape[1]) != self.W:
                # LEGAL capability boundary (the captured graph's statics are
                # keyed to one K source + block-table width), but it must
                # never be quiet: the run continues on the EAGER tail from
                # here, so every later crossing step is a different
                # configuration than the LOCKS_TAIL_GRAPH=1 label implies.
                self.dead = True
                print("[locks] FALLBACK TAIL_GRAPH: K source/block-table "
                      f"width changed (W={self.W} -> "
                      f"{int(block_table.shape[1])}) -- the captured graph is "
                      "invalid; EAGER tail refresh for the rest of this run",
                      flush=True)
                return None
            self.bt_st.copy_(block_table[0, :self.W])
            self.sl_st.copy_(seq_lens[0:1])
            self.gA.replay()
            S2, U = _eigh_chunked(self.Gm_flat)
            self.S2.copy_(S2)
            self.U.copy_(U)
            self.gB.replay()
            return st.L
        except Exception as e:
            # NO-FALLBACK RULE (2026-07-22): a capture/replay failure used to
            # latch `dead` and degrade every later crossing step to the eager
            # chain (3.6-4.0 ms inter-graph wall) for the REST OF THE RUN,
            # under the LOCKS_TAIL_GRAPH=1 label -- a silently different
            # measured configuration.  The legal shape guards are the
            # early-returns above (n_req != 1, multi-row, capture, profiling);
            # anything reaching here is a defect.
            self.dead = True
            print("[locks] TAIL_GRAPH failed -- refusing the silent eager "
                  "degrade (kill switch: LOCKS_TAIL_GRAPH=0):", flush=True)
            traceback.print_exc()
            raise RuntimeError(
                "locks TAIL_GRAPH: tail-refresh graph capture/replay failed "
                f"({type(e).__name__}: {e}); no silent eager fallback -- set "
                "LOCKS_TAIL_GRAPH=0 to select the eager tail explicitly."
            ) from e


def tail_graph_run(st, K_layers, block_table, seq_lens, n_req, rows):
    """Entry used by ``rki4_build_tail`` under LOCKS_TAIL_GRAPH=1.
    Returns the rebuilt count, or None -> caller runs the eager path."""
    tg = getattr(st, "_tailgraph", None)
    if tg is None:
        tg = st._tailgraph = _TailGraph()
    return tg.run(st, K_layers, block_table, seq_lens, n_req, rows)
